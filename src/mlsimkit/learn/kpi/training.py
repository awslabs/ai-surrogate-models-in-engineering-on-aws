# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import time

import mlsimkit.learn.common.tracking as tracking
from accelerate import Accelerator
import numpy as np

import torch
import torch_geometric.loader


from mlsimkit.common.logging import getLogger
import mlsimkit.learn.common.training
import mlsimkit.learn.networks.mgn as mgn
from mlsimkit.learn.common.utils import (
    save_prediction_results,
)
from mlsimkit.learn.common.schema.training import GlobalConditionMethod

from .schema.training import TrainingSettings
from .inference import get_predictions
from .data import KPIDataset

log = getLogger(__name__)


def run_train(config: TrainingSettings, accelerator: Accelerator):
    log.info(f"Training configuration: {json.dumps(config.dict(), indent=2)}")
    os.makedirs(config.training_output_dir, exist_ok=True)

    t_start = time.time()

    device = mlsimkit.learn.common.training.initialize(config, accelerator)

    dataset_train = KPIDataset(
        config.train_manifest_path, config.output_kpi_indices, config.global_condition_method, device
    )
    dataset_validation = KPIDataset(
        config.validation_manifest_path, config.output_kpi_indices, config.global_condition_method, device
    )

    if dataset_train.kpi_indices != dataset_validation.kpi_indices:
        raise RuntimeError(
            f"Training and validation KPI indices differ: {dataset_train.kpi_indices} != {dataset_validation.kpi_indices}"
        )

    model_name = "model"
    kpi_indices = dataset_train.kpi_indices
    if hasattr(dataset_train[0], "global_condition"):
        normalize_keys = ("x", "edge_attr", "y", "global_condition")
        data_scaler = mgn.DataScaler(data_list=dataset_train, normalize_keys=normalize_keys, device=device)
    else:
        data_scaler = mgn.DataScaler(data_list=dataset_train, device=device)

    train_data_loader = torch_geometric.loader.DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=config.shuffle_data_each_epoch,
        drop_last=config.drop_last,
        follow_batch=["x", "coarse_x"],
    )

    validation_data_loader = torch_geometric.loader.DataLoader(
        dataset_validation,
        batch_size=config.batch_size,
        shuffle=config.shuffle_data_each_epoch,
        drop_last=config.drop_last,
        follow_batch=["x", "coarse_x"],
    )

    # get graph shape from the dataset
    node_input_size = dataset_train[0].x.shape[1]
    edge_input_size = dataset_train[0].edge_attr.shape[1]
    num_classes = dataset_train[0].y.shape[1]
    global_condition_size = (
        dataset_train[0].global_condition.shape[1]
        if config.global_condition_method == GlobalConditionMethod.MODEL
        else 0
    )

    # use model IO class provided by MGN so generic train function can save/load/create models
    model_loader = mgn.ModelIO(
        config,
        data_scaler,
        graph_shape=(node_input_size, edge_input_size, num_classes),
        accelerator=accelerator,
        global_condition_size=global_condition_size,
    )

    # use the loss function from MGN
    def loss_func(pred, inputs):
        return mgn.calc_loss(pred, inputs, config.loss_metric)

    (
        validation_losses,
        train_losses,
        best_model,
        last_model,
        best_validation_loss,
        validation_loader,
    ) = mlsimkit.learn.common.training.train(
        model_loader,
        train_data_loader,
        validation_data_loader,
        loss_func,
        device,
        config,
        model_name,
        data_scaler,
        accelerator,
    )

    if config.save_predictions_vs_actuals:
        for model in [best_model, last_model]:
            # resetting seeds before getting predictions
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

            data_scaler.inplace = False

            train_pred_dict, train_actual_dict, train_mesh_paths = get_predictions(
                dataset_train,
                model,
                kpi_indices,
                data_scaler,
                device,
            )
            validation_pred_dict, validation_actual_dict, validation_mesh_paths = get_predictions(
                dataset_validation,
                model,
                kpi_indices,
                data_scaler,
                device,
            )
            model_name_predictions = f"best_{model_name}" if model == best_model else f"last_{model_name}"
            predictions_dir = os.path.join(
                config.training_output_dir, f"{model_name_predictions}_predictions"
            )
            os.makedirs(predictions_dir, exist_ok=True)
            if accelerator.is_local_main_process:
                save_prediction_results(
                    kpi_indices,
                    predictions_dir,
                    actual_dicts=[train_actual_dict, validation_actual_dict],
                    pred_dicts=[train_pred_dict, validation_pred_dict],
                    labels=["train", "validation"],
                    mesh_path_lists=[train_mesh_paths, validation_mesh_paths],
                    model_name=model_name_predictions,
                )

    log.info("Training Completed")

    t_end = time.time()
    log.info(f"Total training time: {(t_end - t_start):.3f} seconds / {((t_end - t_start) / 60):.3f} minutes")
    tracking.log_metric("total_processing_time", t_end - t_start)

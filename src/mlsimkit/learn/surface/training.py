# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
import json

from accelerate import Accelerator
import numpy as np

import torch
import torch_geometric.loader
import torch.distributed

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.manifest.manifest import write_manifest_file
from mlsimkit.learn.common.training import is_distributed
import mlsimkit.learn.common.training
import mlsimkit.learn.networks.mgn as mgn
from mlsimkit.learn.common.schema.training import LossMetric

from .schema.training import TrainingSettings
from .inference import get_predictions
from .data import SurfaceDataset


log = getLogger(__name__)


def run_train(config: TrainingSettings, accelerator: Accelerator):
    log.info(f"Training configuration: {json.dumps(config.dict(), indent=2)}")
    os.makedirs(config.training_output_dir, exist_ok=True)

    t_start = time.time()

    device = mlsimkit.learn.common.training.initialize(config, accelerator)

    dataset_train = SurfaceDataset(config.train_manifest_path, device=device)
    dataset_validation = SurfaceDataset(config.validation_manifest_path, device=device)

    model_name = "model"
    data_scaler = mgn.DataScaler(data_list=dataset_train, device=device, normalize_keys=("edge_attr", "y"))

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

    # store the variables with the model so predict command can name variables in .vtp files
    variables = dataset_train.surface_variables()

    # use model IO class provided by MGN so generic train function can save/load/create models
    model_loader = mgn.ModelIO(
        config,
        data_scaler,
        graph_shape=(node_input_size, edge_input_size, num_classes),
        accelerator=accelerator,
        graph_level_prediction=False,
        metadata={"model_type": "surface", "variables": variables},
    )

    # use the loss function from MGN
    def loss_func(pred, inputs):
        node_loss = mgn.calc_loss(pred, inputs, config.loss_metric)

        actual = inputs.y.reshape(pred.size()[0], -1)
        actual = actual.to(pred.dtype)

        predicted_aggr_x = []
        actual_aggr_x = []
        predicted_aggr_y = []
        actual_aggr_y = []
        predicted_aggr_z = []
        actual_aggr_z = []
        graphs = torch.unique(inputs.x_batch)
        for graph in graphs:
            selected = inputs.x_batch == graph
            pred_graph = pred[selected, :]
            actual_graph = actual[selected, :]
            normal_x_graph = inputs.x[selected, 0].reshape(-1, 1)
            normal_y_graph = inputs.x[selected, 1].reshape(-1, 1)
            normal_z_graph = inputs.x[selected, 2].reshape(-1, 1)
            node_weights_graph = inputs.node_weights[selected].reshape(-1, 1)

            predicted_aggr_x.append(torch.sum(pred_graph * normal_x_graph * node_weights_graph, dim=0))
            actual_aggr_x.append(torch.sum(actual_graph * normal_x_graph * node_weights_graph, dim=0))
            predicted_aggr_y.append(torch.sum(pred_graph * normal_y_graph * node_weights_graph, dim=0))
            actual_aggr_y.append(torch.sum(actual_graph * normal_y_graph * node_weights_graph, dim=0))
            predicted_aggr_z.append(torch.sum(pred_graph * normal_z_graph * node_weights_graph, dim=0))
            actual_aggr_z.append(torch.sum(actual_graph * normal_z_graph * node_weights_graph, dim=0))

        aggr_loss_x = torch.nn.MSELoss()(torch.stack(predicted_aggr_x), torch.stack(actual_aggr_x))
        aggr_loss_y = torch.nn.MSELoss()(torch.stack(predicted_aggr_y), torch.stack(actual_aggr_y))
        aggr_loss_z = torch.nn.MSELoss()(torch.stack(predicted_aggr_z), torch.stack(actual_aggr_z))

        if config.loss_metric == LossMetric.RMSE.value:
            aggr_loss_x = torch.sqrt(aggr_loss_x)
            aggr_loss_y = torch.sqrt(aggr_loss_y)
            aggr_loss_z = torch.sqrt(aggr_loss_z)

        aggr_loss_x = aggr_loss_x * config.strength_x
        aggr_loss_y = aggr_loss_y * config.strength_y
        aggr_loss_z = aggr_loss_z * config.strength_z

        return torch.stack([node_loss, aggr_loss_x, aggr_loss_y, aggr_loss_z])

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

    # When distributed, torch will timeout non-main processes after waiting on the
    # main process. This could happen for long-running post-training steps like
    # getting predictions. Instead of setting a longer timeout, we keep the
    # other processes active by checking the status broadcast by the main process.
    def broadcast_status(done):
        if not is_distributed():
            return
        assert accelerator.is_main_process
        status = (
            torch.ones(1, device=accelerator.device) if done else torch.zeros(1, device=accelerator.device)
        )
        torch.distributed.broadcast(status, src=0)
        accelerator.wait_for_everyone()

    def wait_until_done():
        if not is_distributed():
            return
        assert not accelerator.is_main_process
        done = False
        while not done:
            status = torch.zeros(1, device=accelerator.device)
            torch.distributed.broadcast(status, src=0)
            done = bool(status)
            accelerator.wait_for_everyone()

    if config.save_predictions_vs_actuals and not accelerator.is_main_process:
        # Avoid timeouts by ensuring non-main processes receive updates from
        # the main process.
        wait_until_done()  # train predictions
        wait_until_done()  # validation predictions
    elif config.save_predictions_vs_actuals and accelerator.is_main_process:
        model = best_model

        # reset seeds before getting predictions
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        data_scaler.inplace = False

        model_name_predictions = f"best_{model_name}"
        predictions_dir = os.path.join(config.training_output_dir, f"{model_name_predictions}_predictions")
        train_predictions_dir = os.path.join(predictions_dir, "training")
        validation_predictions_dir = os.path.join(predictions_dir, "validation")

        os.makedirs(train_predictions_dir, exist_ok=True)
        os.makedirs(validation_predictions_dir, exist_ok=True)

        log.info("Get predictions on the training set")
        get_predictions(
            dataset_train,
            model,
            num_classes,
            data_scaler,
            train_predictions_dir,
            device,
            variables=variables,
            save_screenshots=config.save_prediction_screenshots,
            screenshot_size=config.screenshot_size,
            on_status_update=broadcast_status,
        )
        log.info("Get predictions on the validation set")
        get_predictions(
            dataset_validation,
            model,
            num_classes,
            data_scaler,
            validation_predictions_dir,
            device,
            variables=variables,
            save_screenshots=config.save_prediction_screenshots,
            screenshot_size=config.screenshot_size,
            on_status_update=broadcast_status,
        )

        # ensure the predicted files are saved to the manifest files
        write_manifest_file(dataset_train.manifest, config.train_manifest_path)
        write_manifest_file(dataset_validation.manifest, config.validation_manifest_path)

    log.info("Training Completed")

    t_end = time.time()
    log.info(f"Total training time: {(t_end - t_start):.3f} seconds / {((t_end - t_start) / 60):.3f} minutes")

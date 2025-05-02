# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import click
import mlsimkit.learn.common.training
import mlsimkit.learn.networks.autoencoder as ae
import mlsimkit.learn.networks.mgn as mgn
import torch.utils.data
import torch_geometric.loader
import mlsimkit.learn.common.tracking as tracking
from accelerate import Accelerator
from mlsimkit.common.logging import getLogger
from mlsimkit.learn.manifest.manifest import read_manifest_file

from .data import GraphDataset, SlicesDataset, add_noise_and_flip
from .schema.training import TrainAESettings, TrainMGNSettings

log = getLogger(__name__)


def run_train_ae(config: TrainAESettings, accelerator: Accelerator):
    log.debug(f"Training AE configuration: {json.dumps(config.dict(), indent=2)}")
    os.makedirs(config.training_output_dir, exist_ok=True)

    t_start = time.time()

    device = mlsimkit.learn.common.training.initialize(config, accelerator)

    transform = None
    if config.add_noise_and_flip:
        transform = add_noise_and_flip

    dataset_train = SlicesDataset(config.train_manifest_path, transform=transform)
    dataset_validation = SlicesDataset(config.validation_manifest_path, transform=transform)
    data_scaler = ae.DataScaler()

    if config.batch_size > len(dataset_train) or config.batch_size > len(dataset_validation):
        raise click.BadOptionUsage(
            option_name="--batch-size",
            message=f"Batch size '{config.batch_size}' is smaller than dataset sizes "
            f"(train={len(dataset_train)}, validation={len(dataset_validation)})",
        )

    train_data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=config.shuffle_data_each_epoch, drop_last=True
    )

    validation_data_loader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=config.batch_size,
        shuffle=config.shuffle_data_each_epoch,
        drop_last=True,
    )

    # ensure we save the number of frames to the model config by getting the frame count from the manifest (first row)
    config.frame_count = read_manifest_file(config.train_manifest_path).loc[0, "slices_data_frame_count"]

    # use model IO class provided by AE so generic train function can save/load/create models
    model_loader = ae.ModelIO(config, data_scaler, accelerator)

    # use the loss function from AE
    def loss_func(recon_x, x):
        return ae.get_loss(recon_x, x)

    model_name = "model"
    mlsimkit.learn.common.training.train(
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

    log.info("Training Completed")

    t_end = time.time()
    log.info(f"Total training time: {(t_end - t_start):.3f} seconds / {((t_end - t_start) / 60):.3f} minutes")


def run_train_mgn(config: TrainMGNSettings, accelerator: Accelerator):
    log.debug(f"Training configuration: {json.dumps(config.dict(), indent=2)}")
    os.makedirs(config.training_output_dir, exist_ok=True)

    device = mlsimkit.learn.common.training.initialize(config, accelerator)

    dataset_train = GraphDataset(config.train_manifest_path, key="encoding_uri", device=device)
    dataset_validation = GraphDataset(config.validation_manifest_path, key="encoding_uri", device=device)
    data_scaler = mgn.DataScaler(
        data_list=dataset_train,
        dimensions={"y": 2},
        shapes={"y": []},  # flatten
        device=device,
    )

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
    node_input_size = dataset_train[1].x.shape[1]
    edge_input_size = dataset_train[1].edge_attr.shape[1]
    num_classes = dataset_train[1].y.shape[2]

    # use model IO class provided by MGN so generic train function can save/load/create models
    model_loader = mgn.ModelIO(
        config,
        data_scaler,
        graph_shape=(node_input_size, edge_input_size, num_classes),
        accelerator=accelerator,
    )

    # use the loss function from MGN
    def loss_func(pred, inputs):
        return mgn.calc_loss(pred, inputs, config.loss_metric)

    t_start = time.time()

    model_name = "model"
    mlsimkit.learn.common.training.train(
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

    log.info("Training Completed")

    t_end = time.time()
    log.info(f"Total training time: {(t_end - t_start):.3f} seconds / {((t_end - t_start) / 60):.3f} minutes")
    tracking.log_metric("total_processing_time", t_end - t_start)

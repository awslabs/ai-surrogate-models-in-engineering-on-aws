# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import time
import copy
import shutil
import json
import numpy as np
import pandas as pd

import click

from pathlib import Path
from tqdm import tqdm
from typing import Callable, Tuple, List, Protocol

from accelerate import Accelerator, utils
from accelerate.utils import DistributedType
from accelerate.state import PartialState

import torch_geometric.data

import mlsimkit.learn.common.tracking as tracking

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.common.schema.training import BaseTrainSettings, Device
from mlsimkit.learn.common.utils import get_optimizer, save_loss_plots

log = getLogger(__name__)


class ModelIO(Protocol):
    """
    Protocol for defining the interface for creating, loading, and saving models.

    This protocol defines the methods that must be implemented by any class that conforms to the ModelIO interface.
    The ModelIO interface is used to abstract the model creation, loading, and saving processes, allowing different
    model architectures to be used without modifying the core training logic.

    Attributes:
        None

    Methods:
        new(): Create a new instance of the model.
        load(config): Load a saved model checkpoint.
        save(model, model_path, train_loss, validation_loss, optimizer, epoch): Save the model checkpoint.
    """

    def new(self):
        """
        Create a new instance of the model.

        Returns:
            torch.nn.Module: The new model instance.
        """
        ...

    def load(self, config):
        """
        Load a saved model checkpoint.

        Args:
            config (mlsimkit.learn.common.schema.training.BaseTrainSettings): The training configuration settings.

        Returns:
            Tuple[torch.nn.Module, torch.optim.Optimizer, int, list, list, float, int, torch.nn.Module, pd.DataFrame]:
                - model: The loaded model.
                - optimizer: The loaded optimizer.
                - start_epoch: The starting epoch for training.
                - train_losses: A list of training losses.
                - validation_losses: A list of validation losses.
                - best_validation_loss: The best validation loss value.
                - best_validation_loss_epoch: The epoch with the best validation loss.
                - best_model: The model with the best validation loss.
                - losses_df: A DataFrame containing the training and validation losses.
        """
        ...

    def save(self, model, model_path, train_loss, validation_loss, optimizer, epoch):
        """
        Save the model checkpoint.

        Args:
            model (torch.nn.Module): The model to save.
            model_path (str or Path): The path to save the model checkpoint.
            train_loss (torch.Tensor): The training loss.
            validation_loss (torch.Tensor): The validation loss.
            optimizer (torch.optim.Optimizer): The optimizer used during training.
            epoch (int): The current epoch number.
        """
        ...


def make_accelerator(config: BaseTrainSettings):
    """
    Create an instance of the HuggingFace Accelerator for efficient training on various hardware configurations.

    The HuggingFace Accelerator is a utility that simplifies the process of training deep learning models
    on different hardware configurations, including CPUs, GPUs, and multi-GPU setups. It handles device
    management, distributed training, mixed precision, and other performance optimizations automatically.

    By using the Accelerator, MLSimKit can leverage efficient training on a wide range of hardware setups
    without the need for extensive manual configuration and optimization. 

    Args:
        config (BaseTrainSettings): The training configuration settings.

    Returns:
        Accelerator: The HuggingFace Accelerator instance.
    """
    return Accelerator(mixed_precision=config.mixed_precision.value, cpu=config.device == Device.CPU)


def is_distributed():
    """
    Check if the current execution environment is distributed.

    Returns:
        bool: True if the execution environment is distributed, False otherwise.
    """
    return PartialState._shared_state.get("distributed_type", DistributedType.NO) != DistributedType.NO


def validate_training_settings(config: BaseTrainSettings, ctx: click.Context):
    """
    Validate the training settings configuration.

    Args:
        config (BaseTrainSettings): The training configuration settings.
        ctx (click.Context): The Click context object.

    Raises:
        click.UsageError: If the configuration is invalid (e.g., using CPU with `accelerate launch`).
    """
    if config.device == Device.CPU and ctx.obj.get("invoked_with_accelerate", False):
        raise click.UsageError(
            "Cannot use --device cpu with 'accelerate launch'. For CPU-only, call 'mlsimkit-learn' directly or 'accelerate launch --no-python --cpu'"
        )


def initialize(config: BaseTrainSettings, accelerator):
    """
    Initialize the training environment based on the provided configuration and accelerator.

    Args:
        config (BaseTrainSettings): The training configuration settings.
        accelerator (Accelerator): The Accelerator instance.

    Returns:
        torch.device: The device to be used for training.
    """
    utils.set_seed(config.seed)

    if config.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        log.debug("Setting the CUBLAS_WORKSPACE_CONFIG environment variable to ':4096:8'.")
        torch.use_deterministic_algorithms(True)

    device = accelerator.device

    return device


def load_checkpoint_model(modelio, config):
    """
    Load a checkpoint model from the provided configuration.

    Args:
        modelio (ModelIO): The ModelIO instance for creating and loading models.
        config (BaseTrainSettings): The training configuration settings.

    Raises:
        Exception: If any of the required checkpoint paths are None.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, int, list, list, float, int, torch.nn.Module, pd.DataFrame]:
            - model: The loaded model.
            - optimizer: The loaded optimizer.
            - start_epoch: The starting epoch for training.
            - train_losses: A list of training losses.
            - validation_losses: A list of validation losses.
            - best_validation_loss: The best validation loss value.
            - best_validation_loss_epoch: The epoch with the best validation loss.
            - best_model: The model with the best validation loss.
            - losses_df: A DataFrame containing the training and validation losses.
    """
    if (
        config.load_checkpoint.checkpoint_path is None
        or config.load_checkpoint.best_checkpoint_path is None
        or config.load_checkpoint.loss_path is None
    ):
        raise Exception(
            "To start model training from a checkpoint the following arguments are required: "
            "checkpoint_path, best_checkpoint_path, loss_path"
        )

    model = modelio.new()
    best_model = modelio.new()

    optimizer = get_optimizer(model.parameters(), config.optimizer)

    checkpoint_model = torch.load(config.load_checkpoint.checkpoint_path)
    checkpoint_best_model = torch.load(config.load_checkpoint.best_checkpoint_path)
    checkpoint_losses_df = pd.read_csv(config.load_checkpoint.loss_path)
    model.load_state_dict(checkpoint_model["model_state_dict"])
    optimizer.load_state_dict(checkpoint_model["optimizer_state_dict"])

    start_epoch = checkpoint_model["epoch"] + 1
    losses_df = checkpoint_losses_df.iloc[:start_epoch]
    train_losses = losses_df["train_loss"].tolist()
    validation_losses = losses_df["validation_loss"].tolist()
    best_validation_loss = checkpoint_best_model["validation_loss"]
    best_validation_loss_epoch = checkpoint_best_model["epoch"]
    best_model.load_state_dict(checkpoint_best_model["model_state_dict"])

    log.info(f"Checkpoint loaded best validation loss={best_validation_loss}")
    return (
        model,
        optimizer,
        start_epoch,
        train_losses,
        validation_losses,
        best_validation_loss,
        best_validation_loss_epoch,
        best_model,
        losses_df,
    )


def validate(loader, device, model, data_scaler, calc_loss):
    """
    Calculate the validation loss of the model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for validation data.
        device (torch.device): The device to use for validation.
        model (torch.nn.Module): The model to validate.
        data_scaler (DataScaler): The data scaler for normalizing data.
        calc_loss (callable): The function to calculate the loss.

    Returns:
        float: The total validation loss.
    """
    loss = 0
    model.eval()
    for data in loader:
        data = data.to(device)  # move before normalize, fix for `accelerate launch`
        data = data_scaler.normalize_all(data)
        with torch.no_grad():
            pred = model(data)
            loss += calc_loss(pred, data)
    return loss / len(loader)


def fmt_state(accelerator):
    """
    Format the state of the Accelerator instance.

    Args:
        accelerator (Accelerator): The Accelerator instance.

    Returns:
        dict: A dictionary containing the formatted state information.
    """
    # Note that str(accelerate.state) adds yucky formatting
    return {
        "Distributed": accelerator.state.distributed_type.value.lower(),
        "Num processes": accelerator.state.num_processes,
        "Process index": accelerator.state.process_index,
        "Local process index": accelerator.state.local_process_index,
        "Device": str(accelerator.device),
        "Mixed precision": accelerator.state.mixed_precision,
    }


def train(
    modelio: ModelIO,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    calc_loss: Callable[[torch.Tensor, torch_geometric.data.Data], torch.Tensor],
    device: torch.device,
    config: BaseTrainSettings,
    model_name: str,
    data_scaler,
    accelerator: Accelerator,
) -> Tuple[List[float], List[float], torch.nn.Module, torch.nn.Module, float, torch.utils.data.DataLoader]:
    """Train a model using the provided data loaders, loss function, and configuration.

    This function encapsulates the core training loop for various machine learning models in MLSimKit.
    It is designed to be generic and reusable across different use cases, such as KPI prediction,
    surface variable prediction, and slice prediction.

    The `train` function operates in conjunction with the `ModelIO` interface, which abstracts the
    creation, loading, and saving of models. The `ModelIO` interface allows different model architectures
    to be used for training without modifying the core training logic.

    The training data and validation data are provided as PyTorch data loaders, which abstract
    the data loading and preprocessing steps. This design allows for different data types and
    preprocessing pipelines to be used for training, as long as they conform to the data loader
    interface.

    The training process involves the following steps:

    1. Prepare the training and validation data loaders.
    2. Initialize or load a model using the `ModelIO` interface.
    3. Perform the training loop, iterating over epochs and updating the model weights.
    4. Validate the model on the validation data loader after each epoch.
    5. Save the best model and checkpoint models using the `ModelIO` interface.
    6. Return the training and validation losses, the best model, and other relevant information.

    By leveraging the `ModelIO` interface, the `train` function can be used with different model
    architectures without modifying its core implementation. The specific model architecture is
    provided through the `modelio` argument, which must conform to the `ModelIO` protocol.

    Args:
        modelio (ModelIO): The ModelIO instance for creating and loading models.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        validation_loader (torch.utils.data.DataLoader): The data loader for validation data.
        calc_loss (Callable[[torch.Tensor, torch_geometric.data.Data], torch.Tensor]): The function to calculate the loss.
        device (torch.device): The device to use for training.
        config (mlsimkit.learn.common.schema.training.BaseTrainSettings): The training configuration settings.
        model_name (str): The name of the model.
        data_scaler (DataScaler): The data scaler for normalizing data.
        accelerator (accelerate.Accelerator): The HuggingFace Accelerator instance.

    Returns:
        Tuple[list, list, torch.nn.Module, torch.nn.Module, float, torch.utils.data.DataLoader]:
            - validation_losses (list): A list of validation losses for each epoch.
            - train_losses (list): A list of training losses for each epoch.
            - best_model (torch.nn.Module): The model with the best validation loss.
            - model (torch.nn.Module): The final trained model.
            - best_validation_loss (float): The best validation loss achieved during training.
            - validation_loader (torch.utils.data.DataLoader): The data loader for validation data.
    """
    t_start_train = time.time()
    tracking.log_params(config.dict())
    tracking.log_artifact(config.train_manifest_path, "manifest")
    tracking.log_artifact(config.validation_manifest_path, "manifest")
    log.info(f"Training state configuration: {json.dumps(fmt_state(accelerator))}")

    log.info(f"Training started for '{model_name}'")
    log.info(f"Train dataset size: {len(train_loader.dataset)}")
    log.info(f"Validation dataset size: {len(validation_loader.dataset)}")

    model = None
    optimizer = None

    # accelerate is batch size *per* process
    # https://huggingface.co/docs/accelerate/v0.29.3/en/concept_guides/performance#observed-batch-sizes
    if (config.batch_size * accelerator.num_processes) > len(train_loader.dataset):
        raise ValueError(
            f"Batch size '{config.batch_size}' is too large for the *training* dataset ({len(train_loader.dataset)}) and the "
            f"number of processes ({accelerator.num_processes}). Use a smaller batch size "
            f"(<={int(len(train_loader.dataset)/accelerator.num_processes)}) or increase the dataset size "
            f"(>={config.batch_size*accelerator.num_processes})"
        )

    # note: validation is single process
    if config.batch_size > len(validation_loader.dataset):
        raise ValueError(
            f"Batch size '{config.batch_size}' is too large for the *validation* dataset ({len(validation_loader.dataset)}). "
            f"Use a smaller batch size (<={int(len(validation_loader.dataset))}) or increase the dataset size "
            f"(>={config.batch_size})"
        )

    # load checkpoint
    # TODO: error if config parameters do not match checkpoint
    if config.load_checkpoint.checkpoint_path is not None:
        log.info(f"Loading checkpoint '{config.load_checkpoint.checkpoint_path}'")
        (
            model,
            optimizer,
            start_epoch,
            train_losses,
            validation_losses,
            best_validation_loss,
            best_validation_loss_epoch,
            best_model,
            losses_df,
        ) = modelio.load(config)

        # restore the epoch number corresponding to the best validation loss -- helps accurate logging/tracking.
        minimum_validation_loss_epoch = best_validation_loss_epoch

        # copy model into output directory if output directories have changed, this keeps the new output
        # directory self-contained
        new_best_model_path = os.path.join(config.training_output_dir, f"best_{model_name}.pt")
        if Path(new_best_model_path).resolve() != Path(config.load_checkpoint.best_checkpoint_path).resolve():
            shutil.copy(config.load_checkpoint.best_checkpoint_path, new_best_model_path)
            log.debug(
                "Copied checkpoint best model '{config.load_checkpoint.best_checkpoint_path}' to new output directory '{new_best_model_path}'"
            )

    else:
        model = modelio.new()
        optimizer = get_optimizer(model.parameters(), config.optimizer)
        start_epoch = 0
        minimum_validation_loss_epoch = start_epoch
        train_losses = []
        validation_losses = []
        best_validation_loss = np.inf
        best_model = None

        losses_df = pd.DataFrame(columns=["epoch", "train_loss", "validation_loss"])

    assert model is not None
    assert optimizer is not None

    end_epoch = start_epoch + config.epochs

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    data_scaler.to(device)  # fix for `accelerate launch`

    # manually update progress bar (tqdm) to log to file, stream, etc. We are not using this
    # for gui/interactive sessions. Later, we could accept a cli option for notebooks.
    pbar = {
        "total": end_epoch,
        "initial": start_epoch,
        "prefix": "Training",
        "unit": "epochs",
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]",
        "ncols": 100,
    }

    start_time = time.time()
    for epoch in range(start_epoch, end_epoch):
        pbar["n"] = epoch
        pbar["elapsed"] = time.time() - start_time
        pbar_string = tqdm.format_meter(**pbar)
        log.info(pbar_string)

        total_training_losses = 0
        model.train()
        for batch in train_loader:
            batch = data_scaler.normalize_all(batch)
            optimizer.zero_grad()
            pred = model(batch)
            losses = calc_loss(pred, batch)
            loss = torch.sum(losses)
            accelerator.backward(loss)
            optimizer.step()
            total_training_losses += losses
            if config.empty_cache:
                torch.cuda.empty_cache()

        total_training_losses /= len(train_loader)
        training_loss = torch.sum(total_training_losses).item()
        train_losses.append(training_loss)
        tracking.log_metric("Training_loss", training_loss)
        tracking.log_metric("Epoch", epoch)

        if epoch % config.validation_loss_save_interval == 0:
            total_validation_losses = validate(validation_loader, device, model, data_scaler, calc_loss)
            validation_loss = torch.sum(total_validation_losses).item()
            validation_losses.append(validation_loss)

            log_message_train_loss = f"Epoch {epoch}: train loss = {training_loss:.5f}"
            log_message_validation_loss = f"; validation loss = {validation_loss:.5f}"
            log_message_best_validation_loss = f"; best validation loss = {best_validation_loss:.5f}"

            if total_training_losses.ndim == 1:
                log_message_train_loss = (
                    log_message_train_loss
                    + " ("
                    + ", ".join([f"{loss:.5f}" for loss in total_training_losses])
                    + ")"
                )
                log_message_validation_loss = (
                    log_message_validation_loss
                    + " ("
                    + ", ".join([f"{loss:.5f}" for loss in total_validation_losses])
                    + ")"
                )

            log_message_loss = (
                log_message_train_loss + log_message_validation_loss + log_message_best_validation_loss
            )

            log.info(log_message_loss)
            tracking.log_metric("Validation_loss", validation_loss)
            tracking.log_metric("Best_validation_loss", best_validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model = copy.deepcopy(model)
                minimum_validation_loss_epoch = epoch
                best_model_path = os.path.join(config.training_output_dir, f"best_{model_name}.pt")
                modelio.save(best_model, best_model_path, loss, best_validation_loss, optimizer, epoch)
            else:
                log.info(f"Epoch {minimum_validation_loss_epoch} had the minimum validation loss.")

        else:
            validation_losses.append(validation_losses[-1])

        losses_df.loc[len(losses_df.index)] = {
            "epoch": epoch,
            "train_loss": train_losses[-1],
            "validation_loss": validation_losses[-1],
        }

        # saving checkpoint model and loss plot
        if epoch % config.checkpoint_save_interval == 0 or epoch == end_epoch - 1:
            checkpoint_model = copy.deepcopy(model)
            if epoch == end_epoch - 1:
                model_path = os.path.join(config.training_output_dir, f"last_{model_name}.pt")
            else:
                os.makedirs(
                    os.path.join(config.training_output_dir, "checkpoint_models"),
                    exist_ok=True,
                )
                model_path = os.path.join(
                    config.training_output_dir,
                    f"checkpoint_models/{model_name}_epoch{epoch}.pt",
                )
            if accelerator.is_local_main_process:
                modelio.save(checkpoint_model, model_path, loss, best_validation_loss, optimizer, epoch)
                save_loss_plots(config, train_losses, validation_losses, model_name)
                save_loss_plots(config, train_losses, validation_losses, model_name, plot_log=True)

        # saving loss
        if accelerator.is_local_main_process:
            model_loss_path = os.path.join(config.training_output_dir, model_name + "_loss.csv")
            losses_df.to_csv(model_loss_path, index=False)

        accelerator.wait_for_everyone()

    pbar["n"] = end_epoch
    pbar["elapsed"] = time.time() - start_time
    pbar_string = tqdm.format_meter(**pbar)
    log.info(pbar_string)

    t_end_train = time.time()
    log.info(
        f"Training time for {model_name}: {(t_end_train - t_start_train):.3f} seconds / {((t_end_train - t_start_train) / 60):.3f} minutes"
    )
    tracking.log_metric("Training_time_sec", (t_end_train - t_start_train))

    log.info(f"Minimum validation loss: {min(validation_losses):.5f}")
    log.info(f"Minimum train loss: {min(train_losses):.5f}")
    tracking.log_metric("Minimum_validation_loss", min(validation_losses))
    tracking.log_metric("Minimum_training_loss", min(train_losses))

    return (
        validation_losses,
        train_losses,
        best_model,
        model,
        best_validation_loss,
        validation_loader,
    )

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os

from typing import Tuple, Dict

import mlsimkit.learn.common.tracking as tracking
import numpy as np
import pandas as pd
import sklearn.metrics
import torch.optim as optim
import torch

from accelerate import utils
from mlsimkit.common.logging import getLogger
from mlsimkit.learn.common.schema.optimizer import LearningRateScheduler, OptimizerAlgorithm

log = getLogger(__name__)


def calculate_mean_stddev(
    dataset: torch.utils.data.Dataset,
    keys: Tuple[str] = ("x"),
    dims: Dict[str, int] = None,
    shapes: Dict[str, Tuple[int, ...]] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Calculate the mean and std deviation for each specified key in a dataset using an online algorithm.

    This function avoids keeping the entire dataset in memory by computing the cumulative mean and standard deviation
    for each key using "Parallel Algorithm" from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm).

    By default, the first dimension is used for calculations and the second dimension determines
    the output shape. You may override the default behavior for a specific key by
    setting ``dims`` and/or ``shape`` respectively.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to normalize.
        keys (Tuple[str], optional): The keys to normalize, defaults to ("x").
        dims (Dict[str, int], optional): Optional dictionary specifying the dimension along
            which to compute the statistics for each key. If not provided or if a key is
            not present in the dictionary, the default dimension is 0.
        shapes (Dict[str, Union[[], Tuple[int]], optional): Optional dictionary specifying
            the target shape for each key after computing the statistics. The default shape
            is the second dimension (i.e., `dataset[0][key].shape[1]`). If an empty
            tuple `()` or an empty list `[]` is specified for a key, the tensor associated
            with that key will be flattened to a scalar (len=0) before returning
            the statistics.
        device: The torch device to conduct the calculations.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple containing two dictionaries:
            1. The cumulative means for each key, with the specified target shape.
            2. The cumulative standard deviations for each key, with the specified target shape.

    Examples:
        >>> # Compute mean and std dev for 'x' key with default dimensions and shapes
        >>> means, stds = calculate_mean_stddev(dataset, keys=('x'))

        >>> # Compute mean and std dev for 'y' along the third dimension (index 2)
        >>> means, stds = calculate_mean_stddev(dataset, keys=('x', 'y'), dims={'y': 2})

        >>> # Flatten 'y' before computing mean and std dev
        >>> means, stds = calculate_mean_stddev(dataset, keys=('x', 'y'), shapes={'y': []})

        >>> # Compute mean and std dev for 'y' along the third dimension (index 2), and return a flattened tensor
        >>> means, stds = calculate_mean_stddev(dataset, keys=('x', 'y'), dims={'y': 2}, shapes={'y': []})
    """
    dims = dims or {}
    shapes = shapes or {}
    shapes = {key: shapes.get(key, dataset[0][key].shape[1]) for key in keys}

    cumulative_means: Dict[str, torch.Tensor] = {key: torch.zeros(shapes[key], device=device) for key in keys}
    cumulative_stds: Dict[str, torch.Tensor] = {}
    cumulative_m2: Dict[str, torch.Tensor] = {key: torch.zeros(shapes[key], device=device) for key in keys}
    total_samples: Dict[str, int] = {key: 0 for key in keys}

    # Each tensor (data) in the dataset may have many samples so we use the
    # parallel algorithm that works for combining sets of A (cumulative) and B (next data).
    # We get get the mean from B and add this to the cumulative totals. Because this is
    # parallel algorithm, we could multi-process this if need be for larger datasets.
    for data in dataset:
        for key in keys:
            if key in data:
                sample_n = data[key].size(dims.get(key, 0))
                current_n = total_samples[key]
                total_samples[key] += sample_n
                N = total_samples[key]

                value = data[key].to(device)
                dim = dims.get(key, 0)
                shape = shapes.get(key)
                mean = value.mean(dim=dim).reshape(shape)
                m2_batch = ((value - mean) ** 2).sum(dim=dim).reshape(shape)

                delta = mean - cumulative_means[key]
                cumulative_means[key] += delta * (sample_n / N)
                cumulative_m2[key] += m2_batch + delta**2 * current_n * sample_n / N

    for key in keys:
        cumulative_stds[key] = (cumulative_m2[key] / (total_samples[key] - 1)) ** 0.5

    return cumulative_means, cumulative_stds


def get_optimizer(model_parameters, config):
    optimizers = {
        OptimizerAlgorithm.ADAM.value: optim.Adam,
        OptimizerAlgorithm.ADAMW.value: optim.AdamW,
        OptimizerAlgorithm.SGD.value: optim.SGD,
        OptimizerAlgorithm.RMSPROP.value: optim.RMSprop,
        OptimizerAlgorithm.ADAGRAD.value: optim.Adagrad,
    }

    optimizer_fn = optimizers.get(config.algorithm)
    if (
        config.algorithm == OptimizerAlgorithm.SGD.value
        or config.algorithm == OptimizerAlgorithm.RMSPROP.value
    ):
        optimizer = optimizer_fn(
            model_parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optimizer_fn(
            model_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    return optimizer


def get_lr_scheduler(optimizer, config):
    if config.lr_scheduler == LearningRateScheduler.STEP.value:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.decay_rate
        )
    elif config.lr_scheduler == LearningRateScheduler.REDUCDE_LR_ON_PLATEAU.value:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.tracking_metric,
            factor=config.decay_rate,
            patience=config.patience_epochs,
            min_lr=config.min_lr,
        )
    elif config.lr_scheduler is None:
        return None
    return lr_scheduler


def save_loss_plots(config, train_losses, validation_losses, model_name, plot_log=False):
    # lazy import as matplotlib is slow
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.title("Model Loss")
    plt.plot(train_losses, label="train loss")
    plt.plot(validation_losses, label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = os.path.join(config.training_output_dir, f"{model_name}_loss.png")
    if plot_log:
        plt.yscale("log")
        plot_path = os.path.join(config.training_output_dir, f"{model_name}_loss_log.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close()
    # FIXME: multiple calls for the same tracking context will overwrite this file
    tracking.log_artifact(plot_path)
    return None


def save_pred_vs_actual_plot(actuals, preds, labels, kpi_idx, plot_path, model_name=""):
    # lazy import as matplotlib is slow
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    for actual, pred, label, color in zip(actuals, preds, labels, ["#0173B2", "#DE8F05"]):
        plt.scatter(pred, actual, marker="x", color=color, label=f"{label} data")

    flattened_list = np.fromiter(itertools.chain.from_iterable(preds + actuals), float)
    data_min = np.min(flattened_list)
    data_max = np.max(flattened_list)
    data_range = abs(data_max - data_min)

    plt.axline((data_min, data_min), slope=1, c="grey")
    plt.ylim(data_min - 0.1 * data_range, data_max + 0.1 * data_range)
    plt.xlim(data_min - 0.1 * data_range, data_max + 0.1 * data_range)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if len(model_name) == 0:
        plt.title(f"Predicted vs Actual - kpi {kpi_idx}")
    else:
        plt.title(f"Predicted vs Actual - kpi {kpi_idx} - {model_name.replace('_', ' ')}")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    # FIXME: multiple calls for the same tracking context will overwrite this file
    tracking.log_artifact(plot_path)
    return None


def calculate_error_metrics(actual, pred, kpi_idx, label, model_name):
    metrics_dict = {}
    metrics_dict["mape"] = sklearn.metrics.mean_absolute_percentage_error(actual, pred)
    metrics_dict["mae"] = sklearn.metrics.mean_absolute_error(actual, pred)
    metrics_dict["mse"] = sklearn.metrics.mean_squared_error(actual, pred)
    suffix = ".".join(filter(None, (str(kpi_idx), model_name or None, label)))
    tracking.log_metrics({k + "." + suffix: v for k, v in metrics_dict.items()})
    model_name = f"{model_name}: " if model_name else ""
    log.info(
        f"{model_name}prediction error for {label} data, kpi {kpi_idx}: MAPE = {metrics_dict['mape']:.5f}, MAE = {metrics_dict['mae']:.5f}, MSE = {metrics_dict['mse']:.5f}"
    )
    return metrics_dict


def calculate_directional_correctness(prediction_results_df, kpi_idx):
    n = len(prediction_results_df)
    if n <= 1:
        return np.nan
    ground_truth = prediction_results_df["actual"].tolist()
    prediction = prediction_results_df["prediction"].tolist()
    pairs = itertools.combinations(range(n), 2)
    directionally_correct_list = [
        (ground_truth[i] - ground_truth[j]) * (prediction[i] - prediction[j]) > 0
        for i, j in pairs
    ]
    directional_correctness = np.mean(directionally_correct_list)
    log.info(
        f"KPI {kpi_idx} predictions are directionally correct "
        f"{np.round(directional_correctness * 100, 1)}% of the time "
        f"among {len(directionally_correct_list)} run pairs."
    )
    return directional_correctness


def save_prediction_results(
    kpi_indices,
    predictions_dir,
    mesh_path_lists,
    actual_dicts,
    pred_dicts,
    labels,
    model_name="",
    ground_truth_exist=True,
):
    error_metrics_list = []
    prediction_results_df = pd.DataFrame()

    for kpi_idx in kpi_indices:
        if ground_truth_exist:
            plot_path = os.path.join(predictions_dir, f"predicted_vs_actual_kpi{kpi_idx}.png")
            save_pred_vs_actual_plot(
                actuals=[actual[kpi_idx] for actual in actual_dicts],
                preds=[pred[kpi_idx] for pred in pred_dicts],
                labels=labels,
                kpi_idx=kpi_idx,
                plot_path=plot_path,
                model_name=model_name,
            )

        for pred_dict, actual_dict, mesh_path_list, label in zip(
            pred_dicts, actual_dicts, mesh_path_lists, labels
        ):
            df_temp = pd.DataFrame(
                {
                    "kpi_index": kpi_idx,
                    "label": label,
                    "model_name": model_name,
                    "actual": actual_dict[kpi_idx],
                    "prediction": pred_dict[kpi_idx],
                    "mesh_path": mesh_path_list,
                }
            )

            if ground_truth_exist:
                error_metrics_dict = calculate_error_metrics(
                    actual_dict[kpi_idx], pred_dict[kpi_idx], kpi_idx, label, model_name
                )
                error_metrics_dict["directional_correctness"] = calculate_directional_correctness(
                    df_temp, kpi_idx
                )
                error_metrics_list.append(
                    {
                        "kpi_index": kpi_idx,
                        "label": label,
                        "model_name": model_name,
                        "plot_path": plot_path,
                        **error_metrics_dict,
                    }
                )

            prediction_results_df = pd.concat([prediction_results_df, df_temp])

    if ground_truth_exist:
        error_metrics_df = pd.DataFrame(error_metrics_list)
        error_path = os.path.join(predictions_dir, "dataset_prediction_error_metrics.csv")
        error_metrics_df.to_csv(error_path, index=False)
        log.info(f"Saved error metrics: {error_path}")
        tracking.log_artifact(error_path)

    results_filepath = os.path.join(predictions_dir, "prediction_results.csv")
    prediction_results_df.to_csv(results_filepath, index=False)
    log.info(f"Saved results: {results_filepath}")
    tracking.log_artifact(results_filepath)

    return None


def save_dataset(dataset, path, indices=None):
    if indices is not None:
        dataset = [dataset[i] for i in indices]
    utils.save(dataset, path)

    return dataset

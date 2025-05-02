# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time

import mlsimkit.learn.common.tracking as tracking
import numpy as np
import torch

from pathlib import Path

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.common.schema.training import Device
from mlsimkit.learn.common.utils import save_prediction_results
from mlsimkit.learn.manifest.manifest import make_manifest
from mlsimkit.learn.networks.mgn import load_model

from .schema.inference import InferenceSettings
from .schema.preprocessing import PreprocessingSettings
from .preprocessing import process_mesh_files, add_preprocessed_files

from .data import KPIDataset


log = getLogger(__name__)


def get_predictions(
    dataset, model, kpi_indices, data_scaler, device=Device.CPU.value, ground_truth_exist=True
):
    pred_list = []
    actual_list = []
    mesh_path_list = []

    model.eval()
    for data in dataset:
        data = data.to(device)
        model = model.to(device)
        data_scaler.to(device)
        mesh_path_list.append(data.mesh_path)
        if ground_truth_exist:
            actual_list.append(data.y[0].tolist())
        with torch.no_grad():
            pred = model(data_scaler.normalize_all(data))
            pred_list.append(data_scaler.unnormalize(pred, "y")[0].tolist())
    pred_dict = {
        kpi_idx: [pred_list[n_data][i] for n_data in range(len(pred_list))]
        for i, kpi_idx in enumerate(kpi_indices)
    }
    if ground_truth_exist:
        actual_dict = {
            kpi_idx: [actual_list[n_data][i] for n_data in range(len(actual_list))]
            for i, kpi_idx in enumerate(kpi_indices)
        }
    else:
        actual_dict = {kpi_idx: [None] * len(pred_list) for kpi_idx in kpi_indices}

    return pred_dict, actual_dict, mesh_path_list


def run_predict(config: InferenceSettings, compare_groundtruth: bool = True, mesh_files=None):
    log.info(f"Inference configuration: {config}")
    os.makedirs(config.inference_results_dir, exist_ok=True)

    t_start = time.time()

    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(5)

    model, model_dict = load_model(config.model_path)

    data_scaler = model_dict["data_scaler"]
    num_classes = model_dict["num_classes"]
    global_condition_method = model_dict["global_condition_method"]

    if not mesh_files:
        log.info("Predicting using input manifest '%s'", config.manifest_path)
        tracking.log_artifact(config.manifest_path, "manifest")
        dataset = KPIDataset(
            config.manifest_path, config.output_kpi_indices, global_condition_method, device="cpu"
        )
    else:
        log.info("Predicting using input mesh files, preprocessing first...")
        # let the user specify mesh files, but we need to preprocess first and write to file (for now)
        preprocess_settings = PreprocessingSettings(
            output_dir=os.path.join(config.inference_results_dir, "preprocessed_data"),
            num_processes=config.num_processes,
        )
        supported_mesh_formats = [".stl", ".vtp"]
        manifest = make_manifest(([f] for f in mesh_files if Path(f).suffix in supported_mesh_formats))
        manifest = process_mesh_files(preprocess_settings, manifest, use_labels=False)
        manifest = add_preprocessed_files(manifest, (f for f in mesh_files if Path(f).suffix == ".pt"))
        dataset = KPIDataset(manifest, config.output_kpi_indices, global_condition_method, device="cpu")

    log.info(f"Inference dataset size: {len(dataset)}")

    ground_truth_exist = dataset.kpi_indices
    if compare_groundtruth and not ground_truth_exist:
        raise RuntimeError("Cannot compare ground truth, KPIs missing from dataset")

    kpi_indices = dataset.kpi_indices if dataset.kpi_indices else list(range(num_classes))
    data_scaler.inplace = False

    t_data_model_loaded = time.time()
    log.info(f"Time to load dataset and model: {(t_data_model_loaded - t_start):.3f} seconds")
    pred_dict, actual_dict, mesh_path_list = get_predictions(
        dataset, model, kpi_indices, data_scaler, ground_truth_exist=compare_groundtruth
    )

    t_inference_done = time.time()
    log.info(
        f"Inference time for each data point: {((t_inference_done - t_data_model_loaded) / len(dataset)):.3f} seconds"
    )
    predictions_dir = config.inference_results_dir
    save_prediction_results(
        kpi_indices=kpi_indices,
        predictions_dir=predictions_dir,
        mesh_path_lists=[mesh_path_list],
        actual_dicts=[actual_dict],
        pred_dicts=[pred_dict],
        labels=["inference"],
        ground_truth_exist=compare_groundtruth,
    )

    t_end = time.time()
    log.info(f"Total inference time: {(t_end - t_start):.3f} seconds")
    tracking.log_metric("total_processing_time", t_end - t_start)

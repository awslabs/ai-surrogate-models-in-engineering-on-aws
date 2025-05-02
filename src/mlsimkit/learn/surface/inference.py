# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time

import numpy as np
import pandas as pd
import pyvista as pv
import torch

from pathlib import Path

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.networks.mgn import load_model
from .schema.inference import InferenceSettings, Device
from .data import SurfaceDataset
from .visualize import Viewer

log = getLogger(__name__)


def create_polydata_from_data(data):
    pv_polydata = pv.PolyData()
    points = data.x[:, -3:].cpu().numpy()
    if "middle_point" in data.keys():
        points = points / data.scale_factor.cpu().numpy() + data.middle_point.cpu().numpy()
    pv_polydata.points = points
    if "cells" in data.keys():
        pv_polydata.faces = data.cells.cpu().numpy()
    else:
        edges = data.edge_index.cpu().numpy()
        n_edges = edges.shape[-1]
        lines = np.empty(int(n_edges * 3 / 2), dtype=edges.dtype)
        lines[0::3] = 2
        lines[1::3] = edges[0, : int(n_edges / 2)]
        lines[2::3] = edges[1, : int(n_edges / 2)]
        pv_polydata.lines = lines

    return pv_polydata


def add_prediction_to_polydata(pv_polydata, pred, variable_names):
    for i in range(len(variable_names)):
        pv_polydata.point_data[variable_names[i]] = pred[:, i].cpu().numpy()

    return pv_polydata


def add_difference_to_polydata(pv_polydata, data, pred, variable_names):
    for i in range(len(variable_names)):
        pv_polydata.point_data[variable_names[i] + "_error"] = (
            data.y[:, i].cpu().numpy() - pred[:, i].cpu().numpy()
        )

    return pv_polydata


def convert_to_pv_polydata(data, pred, save_prediction, save_difference, variable_names=None):
    pv_polydata = create_polydata_from_data(data)

    if not variable_names:
        variable_names = [str(i) for i in range(pred.shape[-1])]

    if save_prediction:
        pv_polydata = add_prediction_to_polydata(pv_polydata, pred, variable_names)

    if save_difference:
        pv_polydata = add_difference_to_polydata(pv_polydata, data, pred, variable_names)

    return pv_polydata


def get_predictions(
    dataset,
    model,
    num_classes,
    data_scaler,
    predictions_dir,
    device,
    save_prediction=True,
    save_difference=True,
    ground_truth_exist=True,
    variables=[],
    save_screenshots=False,
    screenshot_size=[2000, 800],
    on_status_update=None,
):
    mesh_path_list = []

    # format variable names for VTP files *and* file-friendly
    variable_names = []
    variable_filenames = []
    for v in variables:
        name = v["name"]
        if "dimension" in v:
            variable_names.append(f"{name}[{v['dimension']}]")
            variable_filenames.append(f"{name}_idx{v['dimension']}")
        else:
            variable_names.append(name)
            variable_filenames.append(name)

    metrics_dict = {v: [] for v in variable_names}
    model.eval()

    if save_screenshots:
        viewer = Viewer(dataset, interactive=False)
        screenshot_dir = Path(predictions_dir) / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    data_scaler.to(device)
    for i, data in enumerate(dataset):
        log.info(f"Run inference on geometry {i+1}")
        data = data.to(device)
        # TODO: use run names/ids to replace mesh paths
        mesh_path = data.mesh_path[0]
        mesh_path = Path(mesh_path[0]) if isinstance(mesh_path, list) else Path(mesh_path)
        mesh_path_list.append(mesh_path.with_suffix(".vtp").name)
        with torch.no_grad():
            pred = model(data_scaler.normalize_all(data))
            pred = data_scaler.unnormalize(pred, "y")

        if ground_truth_exist:
            for j in range(num_classes):
                predicted = pred[:, j]
                actual = data.y[:, j]
                mse = torch.nn.MSELoss()(predicted, actual)
                rmse = torch.sqrt(mse)
                ae = torch.nn.L1Loss(reduction="none")(predicted, actual)
                mae = torch.mean(ae)
                range_actual = torch.max(actual) - torch.min(actual)
                range_actual_1st_99th = torch.quantile(actual, 0.99) - torch.quantile(actual, 0.01)
                wmape = mae / torch.mean(torch.abs(actual))
                mae_by_range_1st_99th = mae / range_actual_1st_99th
                perc_99th_ae = torch.quantile(ae, 0.99)
                worst_1_perc_ae_by_range = torch.mean(ae[ae > perc_99th_ae]) / range_actual
                metrics_one_geometry = {
                    "rmse": rmse.item(),
                    "mae": mae.item(),
                    "wmape": wmape.item(),
                    "mae_by_range_1st_99th": mae_by_range_1st_99th.item(),
                    "worst_1_perc_ae_by_range": worst_1_perc_ae_by_range.item(),
                }
                metrics_dict[variable_names[j]].append(metrics_one_geometry)

        save_difference = save_difference and ground_truth_exist
        if save_prediction or save_difference:
            pv_polydata = convert_to_pv_polydata(
                data,
                pred,
                save_prediction=save_prediction,
                save_difference=save_difference,
                variable_names=variable_names,
            )
            individual_predicted_files_dir = Path(predictions_dir) / "results"
            individual_predicted_files_dir.mkdir(parents=True, exist_ok=True)

            predicted_mesh_filename = "predicted_" + mesh_path.stem + ".vtp"
            predicted_mesh_filepath = individual_predicted_files_dir / predicted_mesh_filename

            dataset.set_predicted_file(i, predicted_mesh_filepath.resolve().as_posix())
            pv_polydata.save(predicted_mesh_filepath)

            if save_screenshots:
                for j in range(num_classes):
                    filename = viewer.take_screenshot(
                        i, variables[j], screenshot_dir, screenshot_size[0], screenshot_size[1]
                    )
                    log.info(f"Screenshot written '{filename}'")

        if on_status_update:
            on_status_update(done=False)

    if ground_truth_exist:
        metrics_list = []
        for varname, varfilename in zip(variable_names, variable_filenames):
            metrics_per_variable_df = pd.DataFrame(metrics_dict[varname], index=mesh_path_list)
            metrics_per_variable_df.index.name = "mesh"
            geometry_level_error_file_path = os.path.join(
                predictions_dir, f"{varfilename}_errors_by_geometry.csv"
            )
            metrics_per_variable_df.to_csv(geometry_level_error_file_path)

            metrics_per_variable = metrics_per_variable_df.mean(axis=0)
            metrics_list.append(metrics_per_variable)
            log.info(
                f"Prediction error for surface variable '{varname}': "
                f"RMSE (root mean squared error) = {metrics_per_variable['rmse']:.10f}, "
                f"MAE (mean absolute error) = {metrics_per_variable['mae']:.5f}, "
                f"WMAPE (weighted mean absolute percentage error) = {metrics_per_variable['wmape']:.3f}, "
                f"MAE normalized by 1%-99% ground truth range = {metrics_per_variable['mae_by_range_1st_99th']:.3f}, "
                f"Average largest 1% absolute deviation normalized by ground truth range = {metrics_per_variable['worst_1_perc_ae_by_range']:.3f}, "
            )
        error_metric_file_path = os.path.join(predictions_dir, "error_metrics.csv")
        all_metrics_df = pd.DataFrame(metrics_list, index=variable_names)
        all_metrics_df.index.name = "variable"
        all_metrics_df.to_csv(error_metric_file_path)

    if on_status_update:
        on_status_update(done=True)

    return None


def run_predict(config: InferenceSettings):
    log.info(f"Inference configuration: {config}")
    os.makedirs(config.inference_results_dir, exist_ok=True)

    t_start = time.time()

    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(5)

    device = config.device
    if device == Device.CUDA.value and not torch.cuda.is_available():
        device = Device.CPU.value
        log.warning("CUDA device is not available. Use CPU instead.")

    dataset = SurfaceDataset(config.manifest_path, device=device)

    log.info(f"Inference dataset size: {len(dataset)}")

    model, model_dict = load_model(config.model_path)

    data_scaler = model_dict["data_scaler"]
    num_classes = model_dict["num_classes"]
    variables = model_dict.get("metadata", {}).get("variables", [])

    save_prediction = save_difference = False
    if "prediction" in config.save_vtp_output:
        save_prediction = True
    if "difference" in config.save_vtp_output:
        save_difference = True

    ground_truth_exist = bool(dataset[0].y is not None)
    data_scaler.inplace = False

    t_data_model_loaded = time.time()
    log.info(f"Time to load dataset and model: {(t_data_model_loaded - t_start):.3f} seconds")
    predictions_dir = config.inference_results_dir
    get_predictions(
        dataset,
        model,
        num_classes,
        data_scaler,
        predictions_dir,
        device,
        save_prediction,
        save_difference,
        ground_truth_exist,
        variables=variables,
        save_screenshots=config.save_prediction_screenshots,
        screenshot_size=config.screenshot_size,
    )
    t_inference_done = time.time()
    log.info(
        f"Inference time for each data point: {((t_inference_done - t_data_model_loaded) / len(dataset)):.3f} seconds"
    )

    t_end = time.time()
    log.info(f"Total inference time: {(t_end - t_start):.3f} seconds")

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources  # nosemgrep: python37-compatibility-importlib2
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import click
import cv2
import mlsimkit.image
import mlsimkit.learn.common.tracking as tracking
import mlsimkit.learn.networks.mgn as mgn
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch_geometric
from mlsimkit.common.logging import getLogger
from mlsimkit.learn.manifest.manifest import (
    make_working_manifest,
    read_manifest_file,
    resolve_file_path,
    write_manifest_file,
)
from mlsimkit.learn.networks import autoencoder
from mlsimkit.learn.common.mesh import load_mesh_files, as_torch_data
from torchmetrics import MeanSquaredLogError
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)

from .data import SlicesDataset
from .schema.inference import EncoderInferenceSettings, InferenceSettings, PredictionSettings

log = getLogger(__name__)


def calculate_metrics(original, prediction):
    results = {}
    prediction = torch.from_numpy(prediction.transpose(0, 3, 1, 2) / 255)
    original = torch.from_numpy(original.transpose(0, 3, 1, 2) / 255)
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    mape = MeanAbsolutePercentageError()
    msle = MeanSquaredLogError()
    psnr = PeakSignalNoiseRatio()
    results["mse"] = mse(prediction, original).item()
    results["mae"] = mae(prediction, original).item()
    results["mape"] = mape(prediction, original).item()
    results["msle"] = msle(prediction, original).item()
    results["psnr"] = psnr(prediction, original).item()
    log.info(f"Result metrics: {results}")
    return results


def load_model(model_path, device="cpu"):
    model_dict = torch.load(model_path)
    model = autoencoder.ConvAutoencoder(model_dict["config"].autoencoder_settings)
    model.load_state_dict(model_dict["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_dict


class Inference:
    def __init__(self, manifest_path, model_path, device):
        self.device = device
        self.model, self.model_dict = load_model(model_path, self.device)
        self.data_scaler = self.model_dict["data_scaler"]

        # TODO: unify manifest & dataset interface?
        self.manifest = read_manifest_file(manifest_path)
        self.dataset = SlicesDataset(manifest_path)

    @torch.no_grad
    def encodings(self):
        """
        Get encodings for dataset with loaded model, returns a generator
        """

        def get_coding(data):
            return self.model.encode(self.data_scaler.normalize_all(data))

        self.model.to(self.device)
        self.model.eval()
        return torch.stack([get_coding(ds.unsqueeze(0).to(self.device)).cpu() for ds in self.dataset])

    @torch.no_grad
    def full_codings(self):
        """
        Get full codings for dataset with loaded model, returns a generator
        """

        def get_coding(data):
            return self.data_scaler.unnormalize_all(self.model(self.data_scaler.normalize_all(data)))

        self.model.to(self.device)
        self.model.eval()
        return torch.stack([get_coding(ds.unsqueeze(0).to(self.device)).cpu() for ds in self.dataset])


def process_individual_mesh_data(path_list, label=None):
    mesh = load_mesh_files(*path_list)
    return as_torch_data(mesh)


def run_inference_encoder(config: EncoderInferenceSettings):
    os.makedirs(config.inference_results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    package_root_dir = importlib.resources.files("mlsimkit")

    for manifest_path in config.manifest_paths:
        log.info("Processing manifest '%s'", manifest_path)
        inference = Inference(manifest_path, config.model_path, device)
        for i, encoding in enumerate(inference.encodings()):
            geom_files = [
                resolve_file_path(geom_file, package_root_dir)
                for geom_file in inference.dataset.rows.iloc[i]["geometry_files"]
            ]
            base_graph = process_individual_mesh_data(geom_files)
            base_graph["y"] = encoding.to("cpu")
            graph = torch_geometric.data.Data(**base_graph)
            encoding_filepath = Path(config.inference_results_dir) / (
                "geometry-group-" + str(inference.dataset.rows.iloc[i]["id"]) + ".pt"
            )
            torch.save(graph, encoding_filepath)

            # update manifest with new data, use absolute paths
            inference.manifest.loc[i, "encoding_uri"] = urljoin(
                "file://", encoding_filepath.resolve().as_posix()
            )
            log.info("Encoding written '%s'", encoding_filepath)

        write_manifest_file(inference.manifest, manifest_path)


def run_prediction(config: PredictionSettings, project_root: Path, compare_groundtruth: bool = True):
    os.makedirs(config.results_dir, exist_ok=True)

    device = "cpu"  # TODO: always cpu?  "cuda" if torch.cuda.is_available() else "cpu"
    working_manifest_path = make_working_manifest(config.manifest_path, project_root)
    manifest_frame = read_manifest_file(working_manifest_path)
    # Done in case the user provides their own 'id' column we need to check to make sure we don't overwrite it
    # TODO: check for unique `id` if user provided
    if "id" not in manifest_frame.columns:
        manifest_frame["id"] = manifest_frame.index
        write_manifest_file(manifest_frame, working_manifest_path)
    tracking.log_artifact(working_manifest_path, "manifest")
    inference = Inference(working_manifest_path, config.ae_model_path, device)
    ae_model = inference.model
    ae_scaler = inference.data_scaler

    log.debug(f"ae_model config: {inference.model_dict['config']}")

    mgn_model, mgn_model_dict = mgn.load_model(config.mgn_model_path)
    mgn_scaler = mgn_model_dict["data_scaler"]
    mgn_scaler.to("cpu")
    mgn_model.eval()

    log.debug(f"mgn_model_dict node_size: {mgn_model_dict['node_input_size']}")
    log.debug(f"mgn_model_dict edge_size: {mgn_model_dict['edge_input_size']}")
    log.debug(f"mgn_model_dict num_classes: {mgn_model_dict['num_classes']}")
    log.debug(f"mgn_model_dict model_config: {mgn_model_dict['config']}")

    # Get image shape from model file
    model_config = inference.model_dict["config"]
    nchannels = model_config.autoencoder_settings.image_size.depth
    nframes = model_config.frame_count
    xres, yres = (
        model_config.autoencoder_settings.image_size.width,
        model_config.autoencoder_settings.image_size.height,
    )

    log.info(f"Predicting images frames={nframes} channels={nchannels} size={(xres,yres)}")

    manifest = inference.manifest

    package_root_dir = importlib.resources.files("mlsimkit")

    image_dir = Path(config.results_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    results_filepath = Path(config.results_dir) / "results.jsonl"

    results = (
        open(Path(config.results_dir) / "results.jsonl", "w", encoding="utf-8")
        if compare_groundtruth
        else None
    )

    with torch.no_grad():
        for i, row in manifest.iterrows():
            geom_files = [
                resolve_file_path(geom_file, package_root_dir) for geom_file in row["geometry_files"]
            ]

            log.info("Predicting '%s'...", geom_files)

            graph = process_individual_mesh_data(geom_files)
            graph_normal = mgn_scaler.normalize_all(graph)
            encoding = mgn_model(graph_normal)
            encoding_denormalized = mgn_scaler.unnormalize(encoding, "y")
            log.debug(f"encoded shape: {encoding.shape}")
            log.debug(f"encoded_denormalized shape: {encoding_denormalized.shape}")
            decoded = ae_model.decode(encoding_denormalized)
            prediction = ae_scaler.unnormalize_all(decoded)

            # Reshape data as an array of BGR images
            predicted_images = (
                prediction.numpy()
                .reshape((nframes, nchannels, yres, xres), order="A")
                .transpose(0, 2, 3, 1)[:, :, :, ::-1]
            )

            # save prediction data
            output_path = f"{config.results_dir}/geometry-group-{row['id']}"
            np.save(output_path + "-prediction.npy", predicted_images)

            # save prediction images
            for frame in range(nframes):
                output_path = (image_dir / f"geometry-group-{row['id']}").as_posix()
                log.debug("Writing prediction image files: '%s' (frame=%s)", output_path, frame)
                cv2.imwrite(output_path + f"-prediction-{frame}.png", predicted_images[frame])

            if compare_groundtruth:
                if "slices_data_uri" not in row:
                    raise click.UsageError(
                        f"Key 'slices_data_uri' missing in manifest (row {i}), required to compare groundtruth"
                    )
                slice_data_uri = row["slices_data_uri"]
                slice_data_path = Path(urlparse(slice_data_uri).path)

                if not slice_data_path.exists():
                    raise click.FileError(filename=slice_data_path)

                log.info("Comparing predictions to '%s'", slice_data_path)
                original_images = (
                    np.load(slice_data_path)
                    .reshape((nframes, nchannels, yres, xres), order="A")
                    .transpose(0, 2, 3, 1)[:, :, :, ::-1]
                )
                error_images = np.sqrt(((original_images - predicted_images) ** 2).mean(axis=3)).astype(
                    np.uint8
                )

                for frame in range(nframes):
                    output_path = (image_dir / f"geometry-group-{row['id']}").as_posix()
                    log.debug("Writing original and error image files: '%s' (frame=%s)", output_path, frame)
                    cv2.imwrite(output_path + f"-original-{frame}.png", original_images[frame])
                    cv2.imwrite(output_path + f"-error-{frame}.png", error_images[frame])

                    combined_image = mlsimkit.image.combine_images(
                        original_images[frame],
                        predicted_images[frame],
                        cv2.cvtColor(error_images[frame], cv2.COLOR_GRAY2RGB),
                        border_spacing=20,
                        resize_factor=1.0,
                    )
                    image_combined_path = output_path + f"-combined-{frame}.png"
                    cv2.imwrite(image_combined_path, combined_image)
                    tracking.log_artifact(image_combined_path)

                # stream results metadata to file as json lines
                metrics = calculate_metrics(original_images, predicted_images)
                result = json.dumps({"row": row.to_dict(), "metrics": metrics}) + "\n"
                results.write(result)

    log.info("Images written to '%s'", image_dir)

    if results:
        results.close()
        log.info("Ground truth results written to '%s'", results_filepath)
        tracking.log_artifact(results_filepath)
        tracking.log_metrics(
            pd.json_normalize(pd.read_json(results_filepath, lines=True)["metrics"]).mean(axis=0).to_dict()
        )


def run_inference_ae(config: InferenceSettings):
    os.makedirs(config.inference_results_dir, exist_ok=True)

    log.info(f"Running inference on model '{config.model_path}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = Inference(config.manifest_path, config.model_path, device)
    dataset = inference.dataset

    # Get image shape from model file
    model_config = inference.model_dict["config"]
    nchannels = model_config.autoencoder_settings.image_size.depth
    nframes = model_config.frame_count
    xres, yres = (
        model_config.autoencoder_settings.image_size.width,
        model_config.autoencoder_settings.image_size.height,
    )

    expected_image_size = (nchannels * dataset.nframes, xres, yres)
    actual_image_size = (dataset[0].shape[0], dataset[0].shape[2], dataset[0].shape[1])
    if actual_image_size != expected_image_size:
        raise RuntimeError(
            f"Dataset image shape does not match model shape: {actual_image_size} != {expected_image_size}"
        )

    log.info(f"Generating images frames={nframes} channels={nchannels} size={(xres,yres)}")

    #
    # For each slice group, reconstruct image data and write error metrics to results
    #
    image_dir = Path(config.inference_results_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    results_filepath = Path(config.inference_results_dir) / "results.jsonl"

    with open(results_filepath, "w", encoding="utf-8") as results:
        for slice_data_uri, coding in zip(dataset.slices, inference.full_codings()):
            slice_group_path = Path(urlparse(slice_data_uri).path)

            # convert into array of BGR images
            reconstructed_images = (
                coding.cpu()
                .contiguous()
                .numpy()
                .reshape((nframes, nchannels, yres, xres), order="A")
                .transpose(0, 2, 3, 1)[:, :, :, ::-1]
            )
            original_images = (
                np.load(slice_group_path)
                .reshape((nframes, nchannels, yres, xres), order="A")
                .transpose(0, 2, 3, 1)[:, :, :, ::-1]
            )
            error_images = np.sqrt(((original_images - reconstructed_images) ** 2).mean(axis=3)).astype(
                np.uint8
            )

            log.debug(f"shape of original images: {original_images.shape}")
            log.debug(f"shape of reconstructed images: {reconstructed_images.shape}")
            log.debug(f"shape of error images: {error_images.shape}")

            # save images
            for frame in range(nframes):
                output_path = (image_dir / f"{slice_group_path.stem}").as_posix()
                log.debug("Writing image files: '%s' (frame=%s)", output_path, frame)
                cv2.imwrite(output_path + f"-reconstructed-{frame}.png", reconstructed_images[frame])
                cv2.imwrite(output_path + f"-original-{frame}.png", original_images[frame])
                cv2.imwrite(output_path + f"-error-{frame}.png", error_images[frame])

                combined_image = mlsimkit.image.combine_images(
                    original_images[frame],
                    reconstructed_images[frame],
                    cv2.cvtColor(error_images[frame], cv2.COLOR_GRAY2RGB),
                    border_spacing=20,
                    resize_factor=1.0,
                )
                image_combined_path = output_path + f"-combined-{frame}.png"
                cv2.imwrite(image_combined_path, combined_image)
                tracking.log_artifact(image_combined_path)

            # save reconstructed torch data
            output_path = f"{config.inference_results_dir}/{slice_group_path.stem}"
            torch.save(coding.clone().detach().squeeze().cpu(), output_path + "-reconstructed.pt")

            # write result metadata as a json line
            metrics = calculate_metrics(original_images, reconstructed_images)
            result = json.dumps({"slice_data_uri": slice_data_uri, "metrics": metrics})
            results.write(result + "\n")

    log.info("Images written to '%s'", image_dir)
    log.info("Results written to '%s'", results_filepath)
    tracking.log_artifact(results_filepath)
    tracking.log_metrics(
        pd.json_normalize(pd.read_json(results_filepath, lines=True)["metrics"]).mean(axis=0).to_dict()
    )

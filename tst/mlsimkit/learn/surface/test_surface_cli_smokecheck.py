# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import importlib
import math
from pathlib import Path

import pandas as pd

#
# Files listed from a known working invocation based on dataset/ahmed-sample.
#
EXPECTED_OUTPUT_FILES = {
    "ahmed-sample": [
        "surface-copy.manifest",
        "train.manifest",
        "validate.manifest",
        "test.manifest",
        "preprocessed_data/preprocessed_run_00000.pt",
        "preprocessed_data/preprocessed_run_00001.pt",
        "preprocessed_data/preprocessed_run_00002.pt",
        "preprocessed_data/preprocessed_run_00003.pt",
        "preprocessed_data/preprocessed_run_00004.pt",
        "preprocessed_data/preprocessed_run_00005.pt",
        "preprocessed_data/preprocessed_run_00006.pt",
        "training_output/best_model.pt",
        "training_output/last_model.pt",
        "training_output/model_loss_log.png",
        "training_output/model_loss.csv",
        "training_output/model_loss.png",
        "training_output/checkpoint_models/model_epoch0.pt",
        "training_output/best_model_predictions/training/pMean_errors_by_geometry.csv",
        "training_output/best_model_predictions/training/error_metrics.csv",
        "training_output/best_model_predictions/training/results/predicted_boundary_3_mapped.vtp",
        "training_output/best_model_predictions/training/results/predicted_boundary_4_mapped.vtp",
        "training_output/best_model_predictions/training/results/predicted_boundary_5_mapped.vtp",
        "training_output/best_model_predictions/training/results/predicted_boundary_7_mapped.vtp",
        "training_output/best_model_predictions/validation/pMean_errors_by_geometry.csv",
        "training_output/best_model_predictions/validation/error_metrics.csv",
        "training_output/best_model_predictions/validation/results/predicted_boundary_6_mapped.vtp",
        "predictions/results/predicted_boundary_1_mapped.vtp",
        "predictions/results/predicted_boundary_2_mapped.vtp",
        "predictions/pMean_errors_by_geometry.csv",
        "predictions/error_metrics.csv",
    ]
}


def test_surface(tmp_path):
    dataset_name = "ahmed-sample"
    dataset_path = importlib.resources.files("mlsimkit") / "datasets" / dataset_name
    manifest_path = str(dataset_path.joinpath("surface.manifest"))
    output_dir = str(tmp_path / "output")

    # fmt: off
    command = [
        "mlsimkit-learn", "--output-dir", f"{output_dir}",
        "surface",
            "--manifest-uri", manifest_path,
        "preprocess",
            "--output-dir", f"{output_dir}/preprocessed_data/",
            "--manifest-base-relative-path", "ManifestRoot",
            "--random-seed", str(42),
            "--surface-variables", "name=pMean",
        "train",
            "--device", "cpu", # required with accelerate() to force test on cpu and one process
            "--training-output-dir", f"{output_dir}/training_output",
            "--epochs", "5",
            "--batch-size", "1",
            "--deterministic",
        "predict",
            "--model-path", f"{output_dir}/training_output/best_model.pt",
            "--inference-results-dir", f"{output_dir}/predictions",
    ]
    # fmt: on

    subprocess.check_output(command)  # nosemgrep: dangerous-subprocess-use-audit

    # Check expected files as a smokecheck for now. At least we know the
    # commands run end-to-run...
    assert os.path.exists(output_dir)

    for file_path in EXPECTED_OUTPUT_FILES.get(dataset_name, ["FILE_DOES_NOT_EXIST"]):
        assert (Path(output_dir) / file_path).exists(), f"File {file_path} does not exist"

    # Check model performance stays the same.
    inference_metric_file_path = Path(output_dir) / "predictions" / "error_metrics.csv"
    inference_metrics = pd.read_csv(inference_metric_file_path)
    # The expected_mae number needs to be adjusted when making changes that impact model performance.
    expected_mae = 0.0678220391273498
    assert math.isclose(inference_metrics["mae"][0], expected_mae, rel_tol=8e-04)

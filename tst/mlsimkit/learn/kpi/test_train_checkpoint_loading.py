# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import subprocess
import importlib
import tempfile


import torch


@pytest.mark.parametrize(
    "output_dirs",
    [
        # sample output directory across four training runs
        ["outputs1", "outputs1", "outputs1", "outputs1", "outputs1"],
        # different output directory across all four runs
        ["outputs1", "outputs2", "outputs3", "outputs4", "outputs5"],
    ],
)
def test_best_validation_loss_restored(output_dirs):
    """
    Test the behavior of the training process when loading checkpoints across multiple runs.

    This test verifies that the best model checkpoint is saved to any new output directory,
    and the minimum validation loss epoch is correctly restored from the checkpoint, even when the
    output directory differs from the checkpoint loading paths.

    The test simulates a scenario where the preprocessing is performed once, and then the training
    is performed multiple times with the same + different output directories. It loads the checkpoint
    from the previous iteration for each subsequent training run, ensuring that the training process can
    continue from the previous state.
    """
    dataset_name = "drivaer-sample"
    dataset_path = importlib.resources.files("mlsimkit") / "datasets" / dataset_name
    manifest_path = str(dataset_path.joinpath("kpi.manifest"))

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dirs = [os.path.join(temp_dir, dir_name) for dir_name in output_dirs]

        # Preprocess only once
        preprocess_output_dir = os.path.join(temp_dir, "preprocessed_data")
        os.makedirs(preprocess_output_dir, exist_ok=True)

        # fmt: off
        command = [
            "mlsimkit-learn", "--output-dir", f"{temp_dir}",
            "kpi", "--manifest-uri", manifest_path,
            "preprocess", 
                "--output-dir", f"{preprocess_output_dir}", 
                "--manifest-base-relative-path", "ManifestRoot",
                "--random-seed", str(42),
        ]

        subprocess.check_output(command)  # nosemgrep: dangerous-subprocess-use-audit
        # fmt: on

        model_name = "model"
        for i, output_dir in enumerate(output_dirs):
            os.makedirs(output_dir, exist_ok=True)

            # Set up the train command
            # fmt: off
            train_command = [
                "mlsimkit-learn", "--debug", "--output-dir", f"{temp_dir}", 
                "kpi", "--manifest-uri", manifest_path, 
                "train", 
                    "--device", "cpu", # required with accelerate() to force test on cpu and one process
                    "--train-manifest-path", f"{temp_dir}/train.manifest", 
                    "--validation-manifest-path", f"{temp_dir}/validate.manifest", 
                    "--training-output-dir", f"{output_dir}",
                    "--epochs", "2",
                    "--batch-size", "1",
                    "--deterministic",
            ]
            # fmt: on

            # Set up the load checkpoint configuration (if not the first iteration)
            if i > 0:
                prev_output_dir = output_dirs[i - 1]
                prev_checkpoint_path = os.path.join(prev_output_dir, f"last_{model_name}.pt")
                prev_best_checkpoint_path = os.path.join(prev_output_dir, f"best_{model_name}.pt")
                prev_loss_path = os.path.join(prev_output_dir, f"{model_name}_loss.csv")
                # fmt: off
                train_command.extend(
                    [
                        "--checkpointing.checkpoint-path", prev_checkpoint_path,
                        "--checkpointing.best-checkpoint-path", prev_best_checkpoint_path,
                        "--checkpointing.loss-path", prev_loss_path,
                    ]
                )
                # fmt: on

            # Run the train command, use run() for
            subprocess.check_output(train_command)  # nosemgrep: dangerous-subprocess-use-audit

            # Assert that the best model checkpoint is copied to the current output directory
            assert os.path.exists(os.path.join(output_dir, "best_model.pt"))

        # Load the best model from the last output directory
        best_model_path = os.path.join(output_dirs[-1], "best_model.pt")
        best_model_state = torch.load(best_model_path)

        # Assert that the minimum validation loss epoch is correctly restored
        expected_best_epoch = 5  # depends on sample datset and determinstic training
        assert best_model_state["epoch"] == expected_best_epoch

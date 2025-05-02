# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from enum import Enum


class VtpOutput(str, Enum):
    NO = "no"
    PREDICTION = "prediction"
    DIFFERENCE = "difference"
    BOTH = "prediction_and_difference"


class Device(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class InferenceSettings(BaseModel):
    model_path: Optional[str] = Field(None, description="path to the trained model")
    manifest_path: Optional[str] = Field(None, description="Path to the input manifest file")
    inference_results_dir: Optional[str] = Field(
        None, description="path of the directory where inference results are saved"
    )
    device: Device = Field(default=Device.CUDA, description="the device on which inference is performed")
    save_vtp_output: VtpOutput = Field(
        default=VtpOutput.BOTH,
        description="inference result saving options: 1. 'no' - only save metrics and no .vtp files; "
        "2. 'prediction' - save predicted surface variable values into .vtp files; "
        "3. 'difference' - save the difference between predicted surface variable values and the ground truth into .vtp files "
        "(nothing will be saved if ground truth doesn't exist); "
        "4. 'prediction_and_difference' - save both the predicted values and the difference.",
    )

    save_prediction_screenshots: bool = Field(
        default=False,
        description="save PNG images comparing ground truth vs predicted and error. Requires save_predictions_vs_actuals=True",
    )

    screenshot_size: List[int] = Field(
        [2000, 800],
        min_items=2,
        max_items=2,
        description="output resolution in pixels (width, height) for the screenshots. At least width=1500 and 2.5:1 ratio work well for the tutorial datasets with 3 split views",
    )

    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field, model_validator
from typing import Optional, List
from mlsimkit.learn.common.schema.training import BaseTrainSettings, LossMetric


class TrainingSettings(BaseTrainSettings):
    train_manifest_path: Optional[str] = Field(None, description="Path to the train manifest")
    validation_manifest_path: Optional[str] = Field(None, description="Path to the validation manifest")

    message_passing_steps: int = Field(
        default=10, ge=0, description="number of message passing steps for MGN"
    )
    hidden_size: int = Field(
        default=32,
        ge=1,
        description="size of the hidden layer in the multilayer perceptron (MLP) used in MGN",
    )
    loss_metric: LossMetric = Field(default=LossMetric.MSE, description="loss metric")
    save_predictions_vs_actuals: bool = Field(
        default=True,
        description="save the prediction output files and error metrics for training and validation datasets",
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

    strength_x: float = Field(
        default=0,
        ge=0,
        description="weight of the loss function at the x direction (a non-zero value is typically useful for scaler surface variables such as pressure)"
    )
    strength_y: float = Field(
        default=0,
        ge=0,
        description="weight of the loss function at the y direction (a non-zero value is typically useful for scaler surface variables such as pressure)"
    )
    strength_z: float = Field(
        default=0,
        ge=0,
        description="weight of the loss function at the z direction (a non-zero value is typically useful for scaler surface variables such as pressure)"
    )

    @model_validator(mode="after")
    def screenshots_require_save_predictions(self):
        if self.save_prediction_screenshots and not self.save_predictions_vs_actuals:
            raise ValueError("save_prediction_screenshots=True requires save_predictions_vs_actuals=True")
        return self

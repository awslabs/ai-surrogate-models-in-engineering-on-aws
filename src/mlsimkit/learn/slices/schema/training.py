# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from mlsimkit.common.schema.cli import CliExtras
from mlsimkit.learn.common.schema.training import BaseTrainSettings, LossMetric, PoolingType
from mlsimkit.learn.networks.schema.autoencoder import ConvAutoencoderSettings
from pydantic import Field


class TrainAESettings(BaseTrainSettings):
    autoencoder_settings: ConvAutoencoderSettings = Field(
        default=ConvAutoencoderSettings(), cli=CliExtras(prefix="ae")
    )

    train_manifest_path: Optional[str] = Field(None, description="Path to the train manifest")
    validation_manifest_path: Optional[str] = Field(None, description="Path to the validation manifest")
    add_noise_and_flip: bool = Field(True, description="Add noise and flip augmentations to the input data")
    flip_probability: float = Field(0.5, ge=0, le=1, description="Probability of flipping the input data")
    noise_factor: float = Field(0.01, ge=0.0, le=0.2, description="Factor for adding noise to the input data")

    # Override the batch_size field in BaseTrainSettings so the default value is better for slices
    batch_size: int = Field(
        default=4, ge=1, description="Batch size, recommended equal or greater than 4. See user guide."
    )

    # Storage for the number of frames to be kept with the model data. Written during training. Excluded from config/CLI.
    frame_count: int = Field(1, cli=CliExtras(exclude=True), description="Number of frames per slice group.")


class TrainMGNSettings(BaseTrainSettings):
    train_manifest_path: Optional[str] = Field(None, description="Path to the train manifest")
    validation_manifest_path: Optional[str] = Field(None, description="Path to the validation manifest")

    message_passing_steps: int = Field(
        default=10, ge=1, description="number of message passing steps for MGN"
    )
    hidden_size: int = Field(
        default=128,
        ge=1,
        description="size of the hidden layer in the multilayer perceptron (MLP) used in MGN",
    )
    dropout_prob: float = Field(default=0, ge=0, lt=1, description="probability of an element to be zeroed")
    loss_metric: LossMetric = Field(default=LossMetric.MSE, description="loss metric")
    pooling_type: PoolingType = Field(
        default=PoolingType.MEAN,
        description="Pooling type used in the MGN's model architecture",
    )
    shuffle_train_validation_split: bool = Field(
        default=True, description="shuffle data before train validation split"
    )
    save_predictions_vs_actuals: bool = Field(
        default=False,
        description="save the plots showing the predictions vs the actuals for train and validation datasets",
    )

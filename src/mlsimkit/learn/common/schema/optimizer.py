# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class LearningRateScheduler(str, Enum):
    STEP = "step"
    REDUCDE_LR_ON_PLATEAU = "reducelronplateau"


class OptimizerAlgorithm(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class TrackingMetric(str, Enum):
    MIN = "min"
    MAX = "max"


class OptimizerSettings(BaseModel):
    algorithm: OptimizerAlgorithm = Field(
        default=OptimizerAlgorithm.ADAMW, description="optimization algorithm"
    )
    weight_decay: float = Field(default=0.01, ge=0, description="weight decay coefficient")
    learning_rate: float = Field(default=0.001, gt=0, description="learning rate")
    momentum: float = Field(default=0.9, ge=0, le=1, description="momentum factor")
    lr_scheduler: Optional[LearningRateScheduler] = Field(default=None, description="learning rate scheduler")
    decay_rate: float = Field(default=0.7, gt=0, le=1, description="factor of learning rate decay")
    step_size: int = Field(default=1, ge=1, description="period of learning rate decay (for StepLR)")
    tracking_metric: TrackingMetric = Field(
        default=TrackingMetric.MIN, description="mode of the ReduceLROnPlateau learning rate scheduler"
    )
    patience_epochs: int = Field(
        default=100,
        ge=0,
        description="number of epochs with no improvement after which learning rate will be reduced (for ReduceLROnPlateau)",
    )
    min_lr: float = Field(
        default=5e-5,
        ge=0,
        description="a lower bound on the learning rate of all param groups (for ReduceLROnPlateau)",
    )

    class Config:
        title: str = "Training Optimizer Settings"

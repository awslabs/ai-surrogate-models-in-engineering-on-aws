# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field
from typing import Optional
from mlsimkit.learn.common.schema.training import (
    BaseTrainSettings,
    LossMetric,
    PoolingType,
    GlobalConditionMethod,
)


class TrainingSettings(BaseTrainSettings):
    train_manifest_path: Optional[str] = Field(None, description="Path to the train manifest")
    validation_manifest_path: Optional[str] = Field(None, description="Path to the validation manifest")

    output_kpi_indices: Optional[str] = Field(
        default=None,
        description="index(es) of desired KPIs to predict, separated by ',' (e.g. 0,2,3) (using all if None)",
    )
    message_passing_steps: int = Field(default=5, ge=0, description="number of message passing steps for MGN")
    hidden_size: int = Field(
        default=8, ge=1, description="size of the hidden layer in the multilayer perceptron (MLP) used in MGN"
    )
    dropout_prob: float = Field(default=0, ge=0, lt=1, description="probability of an element to be zeroed")
    pooling_type: PoolingType = Field(
        default=PoolingType.MEAN,
        description="Pooling type used in the MGN's model architecture",
    )
    loss_metric: LossMetric = Field(default=LossMetric.RMSE, description="loss metric")
    save_predictions_vs_actuals: bool = Field(
        default=True,
        description="save the plots showing the predictions vs the actuals for train and validation datasets",
    )
    global_condition_method: GlobalConditionMethod = Field(
        default=GlobalConditionMethod.NODE_FEATURES,
        description="The method to use for integrating global conditions",
    )

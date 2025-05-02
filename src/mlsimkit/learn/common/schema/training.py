# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional
from enum import Enum

from mlsimkit.common.schema.cli import CliExtras

from .optimizer import OptimizerSettings


class Device(str, Enum):
    AUTO = "auto"  # relies on Accelerate library behavior: https://github.com/huggingface/accelerate/tree/main/examples
    CPU = "cpu"  # force single CPU only


class LossMetric(str, Enum):
    RMSE = "rmse"
    MSE = "mse"


class MixedPrecision(str, Enum):
    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


class GlobalConditionMethod(str, Enum):
    MODEL = "model"
    NODE_FEATURES = "node_features"


class LoadCheckpointSettings(BaseModel):
    checkpoint_path: Optional[str] = Field(
        default=None, description="the path of the checkpoint to start training from"
    )
    best_checkpoint_path: Optional[str] = Field(default=None, description="the path of the best checkpoint")
    loss_path: Optional[str] = Field(default=None, description="the path of the model loss csv file")

    class Config:
        title: str = "Checkpoint Loading Settings"


class PoolingType(str, Enum):
    MEAN = "mean"
    MAX = "max"


class BaseTrainSettings(BaseModel):
    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())

    training_output_dir: Optional[str] = Field(
        None, description="path of the folder where training outputs and model checkpoints are saved"
    )
    epochs: int = Field(
        default=5,
        ge=1,
        description="Number of epochs. Default is low for quickstart experience. Higher number of epochs are required for accurate training. See user guide.",
    )

    batch_size: int = Field(
        default=4,
        ge=1,
        description="Batch size determines how to group training data. Note the batch size is per process. For multi-GPU, this means you need enough training and validation data for all processes.",
    )
    drop_last: bool = Field(
        default=True, cli=CliExtras(hidden=True), description="whether to drop last non-full batch"
    )

    seed: int = Field(default=0, description="Random seed")
    shuffle_data_each_epoch: bool = Field(default=True, description="shuffle data every epoch")
    device: Device = Field(
        default=Device.AUTO,
        description="the device on which model training is performed, by default uses one GPU if available. Use 'accelerate launch' command for multi-GPU and other platforms. See user guide.",
    )
    checkpoint_save_interval: int = Field(
        default=10,
        ge=1,
        description="the interval, in epochs, at which the model's checkpoint is saved during training",
    )
    validation_loss_save_interval: int = Field(
        default=1,
        ge=1,
        description="the interval, in epochs, at which the model's validation loss is calculated and saved during training",
    )
    deterministic: bool = Field(
        default=False,
        description="whether to use deterministic algorithms (for message passing) which produce reproducible results at the cost of longer training time",
    )
    mixed_precision: MixedPrecision = Field(
        default=MixedPrecision.NO,
        description="Whether or not to use mixed precision during training, and the type of mixed precision.",
    )
    load_checkpoint: LoadCheckpointSettings = Field(
        default=LoadCheckpointSettings(), cli=CliExtras(prefix="checkpointing")
    )
    optimizer: OptimizerSettings = Field(default=OptimizerSettings(), cli=CliExtras(prefix="opt"))
    empty_cache: bool = Field(
        default=False,
        description="release all unoccupied cached memory each iteration"
    )
    node_encoder_hidden_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="size of the hidden layer in the node encoder (override the value of 'hidden_size')",
        default_to_hidden_size=True,
    )
    edge_encoder_hidden_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="size of the hidden layer in the edge encoder (override the value of 'hidden_size')",
        default_to_hidden_size=True,
    )
    node_message_passing_mlp_hidden_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="size of the hidden layer in the node MLPs used in message passing steps (override the value of 'hidden_size')",
        default_to_hidden_size=True,
    )
    edge_message_passing_mlp_hidden_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="size of the hidden layer in the edge MLPs used in message passing steps (override the value of 'hidden_size')",
        default_to_hidden_size=True,
    )
    node_decoder_hidden_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="size of the hidden layer in the node decoder (override the value of 'hidden_size'; only relevant to surface variable prediction)",
        default_to_hidden_size=True,
    )

    @model_validator(mode='before')
    def set_defaults(cls, values):
        for field_name, field in cls.model_fields.items():
            if field.json_schema_extra and field.json_schema_extra.get('default_to_hidden_size') and values.get(field_name) is None:
                values[field_name] = values.get('hidden_size')
        return values
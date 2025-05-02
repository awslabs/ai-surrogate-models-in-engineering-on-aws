# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from mlsimkit.common.schema.cli import CliExtras


class ImageSize(BaseModel):
    width: int = 128
    height: int = 128
    depth: int = 3


class ConvAutoencoderSettings(BaseModel):
    input_channels: int = Field(
        30, ge=1, description="Number of input channels. Default of 30 assumes 10 RGB slices."
    )
    start_out_channel: int = Field(512, ge=1, description="Starting number of output channels")
    div_rate: int = Field(8, ge=1, description="Division rate for output channels")
    add_sigmoid: bool = Field(True, description="Add sigmoid activation to the output")
    dropout_prob: float = Field(0.0, ge=0, le=1, description="Dropout probability")
    image_size: ImageSize = Field(
        ImageSize(), cli=CliExtras(prefix="img"), description="Target dimension for input images"
    )

    class Config:
        title: str = "Convolutional Autoencoder Settings"

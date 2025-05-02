# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import List


class ConvertImageSettings(BaseModel):
    grayscale: bool = Field(False, description="Convert RGB images to grayscale before conversion")
    resolution: List[int] = Field(
        [128, 128], description="Output resolution of the images"
    )  # is square only supported?

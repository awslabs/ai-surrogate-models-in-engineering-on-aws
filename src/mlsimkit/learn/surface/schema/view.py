# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from enum import Enum


class ViewType(Enum):
    GroundTruth = "Ground Truth"
    Predicted = "Predicted"
    Error = "Error"


class ScreenshotSettings(BaseModel):
    outdir: str = Field(
        "screenshots",
        description="Subdirectory of the project output directory to save images. E.g,  <output_dir>/screenshots.",
    )
    variable: Optional[str] = Field(
        None,
        description="The name of the surface variable to render. Defaults to the first in the data array.",
    )
    dimension: Optional[int] = Field(None, description="The index into the surface variable's array data.")
    image_size: List[int] = Field(
        [2000, 800],
        min_items=2,
        max_items=2,
        description="Image size (width,height) in pixels for the screenshots. At least width=1500 and 2.5:1 ratio work well for the tutorial datasets with 3 split views",
    )
    prefix: Optional[str] = Field(None, description="Optional prefix for image filenames")


class ViewSettings(BaseModel):
    describe: bool = Field(
        False, description="Print a description of the manifest dataset including available surface vaiables."
    )
    gui: bool = Field(True, description="Flag to open a GUI")

    views: List[ViewType] = Field(
        ["Ground Truth", "Predicted", "Error"],
        description="The order and type of views to render. Requires at least one view.",
    )

    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from mlsimkit.learn.manifest.schema import RelativePathBase

class InterpolationMethod(str, Enum):
    POINTS = "points"
    RADIUS = "radius"

    
class PreprocessingSettings(BaseModel):
    manifest_path: Optional[str] = Field(None, description="manifest file path")

    output_dir: Optional[str] = Field(
        "preprocessed_data",
        description="path of the directory where preprocessed data are saved relative to output-dir",
    )
    downsample_remaining_perc: Optional[float] = Field(
        default=None, gt=0, le=100, description="percentage of data to be retained after downsampling"
    )
    num_processes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Preprocess in parallel with multiple CPUs. The default is one less than the total number of CPUs",
    )
    save_cell_data: bool = Field(
        default=False,
        description="save mesh cell data into preprocessed datasets for post processing purposes, e.g. results visualization",
    )
    map_data_to_stl: bool = Field(
        default=False,
        description="map surface data onto the corresponding STL mesh (potentially downsamples the mesh if the geoemtry STL mesh is coarser than the data file mesh)",
    )
    mapping_interpolation_method: InterpolationMethod = Field(
        default=InterpolationMethod.POINTS, 
        description="the interpolation method to be used for mapping")
    
    mapping_interpolation_radius: Optional[float] = Field(
        default=None,
        gt=0,
        description="the maximum distance (radius) from the mesh points within which the basis points must be located for mesh interpolation for mapping to STL",
    )
    mapping_interpolation_n_points: Optional[int] = Field(
        default=3,
        gt=0,
        description="the number of the closest points used to form the interpolation basis for mapping to STL",
    )
    save_mapped_files: bool = Field(
        default=False,
        description="save the mapped data files",
    )
    normalize_node_positions: bool = Field(
        default=True,
        description="normalize the node positions so that they fit within the [-1, 1] cube (in all 3 dimensions) while the aspect ratio is maintained",
    )

    # Allow overriding how relative paths files are found and preprocessed. By default, the current working
    # directory (CWD) is used instead. You may also specify absolute paths in the manifest
    # and ignore relative paths entirely.
    manifest_base_relative_path: RelativePathBase = Field(
        default="CWD", description="Base directory for all files that are relative within the manifest"
    )


class SurfaceVariables(BaseModel):
    """desired surface variables to predict"""

    name: Optional[str] = Field(description="surface variable name to predict")
    dimensions: List[int] = Field(
        default=[],
        description="surface variable dimensions to predict, must be empty for variables with one component",
    )

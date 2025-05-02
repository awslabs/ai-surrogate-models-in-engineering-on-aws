# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional

from mlsimkit.learn.manifest.schema import RelativePathBase


class PreprocessingSettings(BaseModel):
    manifest_path: Optional[str] = Field(None, description="manifest file path")

    output_dir: Optional[str] = Field(
        "preprocessed_data",
        description="path of the directory where preprocessed data are saved relative to output-dir",
    )
    downsample_remaining_perc: Optional[float] = Field(
        default=None,
        gt=0,
        le=100,
        description="Optional percentage of data to be retained after downsampling.",
    )
    num_processes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Preprocess in parallel with multiple CPUs. The default is one less than the total number of CPUs",
    )

    # Allow overriding how relative paths files are found and preprocessed. By default, the current working
    # directory (CWD) is used instead. You may also specify absolute paths in the manifest
    # and ignore relative paths entirely.
    manifest_base_relative_path: RelativePathBase = Field(
        default="CWD", description="Base directory for all files that are relative within the manifest"
    )

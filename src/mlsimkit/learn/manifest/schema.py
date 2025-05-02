# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional
from mlsimkit.common.cli import NamedEnum


class RelativePathBase(NamedEnum):
    CWD = "CWD"
    PackageRoot = "PackageRoot"
    ManifestRoot = "ManifestRoot"


class DataFile(BaseModel):
    """A parameter to extract values from a .dat file"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)  # non-empty
    file_glob: Optional[str] = None
    file_regex: Optional[str] = None
    columns: list = Field(..., min_length=1)
    delimiter: str = Field(",", min_length=1, max_length=1)

    @model_validator(mode="before")
    def check_file_glob_or_regex(cls, values):
        file_glob = values.get("file_glob")
        file_regex = values.get("file_regex")

        if file_glob and file_regex:
            raise ValueError("Only one of 'file_glob' or 'file_regex' should be provided.")
        elif not file_glob and not file_regex:
            raise ValueError("Either 'file_glob' or 'file_regex' must be provided.")

        return values


class FileList(BaseModel):
    """A parameter to list files"""

    model_config = ConfigDict(extra="forbid")

    name: str
    file_glob: Optional[str] = None


class SplitSettings(BaseModel):
    """
    Settings for splitting a dataset into train, validation, and test sets.
    """

    train_size: float = Field(
        0.6, ge=0, le=1, description="Percentage of data to use for training (0.0 - 1.0)"
    )
    valid_size: float = Field(
        0.2, ge=0, le=1, description="Percentage of data to use for validation (0.0 - 1.0)"
    )
    test_size: float = Field(0.2, ge=0, le=1, description="Percentage of data to use for testing (0.0 - 1.0)")

    random_seed: Optional[int] = Field(
        None,
        description="Controls the shuffling before splitting the datasets. Use an integer for reproducible splits.",
    )

    @model_validator(mode="after")
    def check_total_percentage(self) -> "SplitSettings":
        """Validate that the sum of train, validation, and test percentages is 100%.

        Args:
            values (dict): The values of the model's fields.

        Raises:
            ValueError: If the sum of train, validation, and test percentages is not 1.0.

        Returns:
            SplitSettings: The validated instance of the SplitSettings model.
        """
        total_perc = self.train_size + self.valid_size + self.test_size
        if not math.isclose(total_perc, 1.0, abs_tol=1e-6):
            raise ValueError("The sum of train, validation, and test percentages must be 100.0")
        return self

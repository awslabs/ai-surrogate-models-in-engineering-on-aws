# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class CliExtras(BaseModel):
    prefix: Optional[str] = None
    title: Optional[str] = None
    exclude: bool = False
    hidden: bool = False
    use_enum_name: bool = False


class MLFlowConfig(BaseModel):
    experiment_name: Optional[str] = Field(
        None,
        description="Enable MLFlow logging by specifying an experiment name. When specified, logs locally by default. See --tracking-uri. [default: off]",
    )
    tracking_uri: Optional[str] = Field(
        None, description="Folder or server to log MLFlow experiments [default: <output_dir>/mlruns]"
    )
    resume_run: bool = Field(
        False,
        description="Resume the most recent run  from the previous commands for the same experiment name. Fails if a run does not exist.",
    )

    # non-input fields, used to store settings at runtime
    is_local_tracking: bool = Field(
        ..., cli=CliExtras(exclude=True), description="Auto-populated based on tracking_uri"
    )
    run_id: Optional[str] = Field(
        None,
        cli=CliExtras(exclude=True),
        description="Auto-generated for every command invocation with an experiment",
    )
    experiment_id: Optional[str] = Field(
        None,
        cli=CliExtras(exclude=True),
        description="Auto-generated for every command invocation with an experiment",
    )
    experiment: Optional[Any] = Field(
        None,
        cli=CliExtras(exclude=True),
        description="Auto-populated for every command invocation with an experiment",
    )
    run: Optional[Any] = Field(
        None,
        cli=CliExtras(exclude=True),
        description="Auto-populated for every command invocation with an experiment",
    )

    @model_validator(mode="before")
    def post_update(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parsed_uri = urlparse(values["tracking_uri"])
        values["is_local_tracking"] = parsed_uri.scheme == "" or parsed_uri.scheme == "file"
        return values

    @field_validator("run_id", "experiment_id", "experiment", "run")
    @classmethod
    def check_non_input_fields(cls, value: Any, info: ValidationInfo):
        assert False, f"{info.field_name} is a non-input field and should not be set during initialization"

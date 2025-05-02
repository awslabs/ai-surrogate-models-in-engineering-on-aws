# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from mlsimkit.common.cli import CliExtras

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class InferenceSettings(BaseModel):
    inference_data_path: Optional[str] = Field(
        None, description="path of the preprocessed data used for inference"
    )
    model_path: Optional[str] = Field(None, description="path to the trained model")

    manifest_path: Optional[str] = Field(None, description="Path to the input manifest file")

    inference_results_dir: Optional[str] = Field(
        None, description="path of the directory where inference results are saved"
    )

    num_processes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Preprocess in parallel with multiple CPUs. The default is one less than the total number of CPUs",
    )
    output_kpi_indices: Optional[str] = Field(
        default=None,
        cli=CliExtras(exclude=True),
        description="Carried forward from the train command.",
    )

    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())

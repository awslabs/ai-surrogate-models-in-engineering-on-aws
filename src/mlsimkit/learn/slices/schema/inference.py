# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class InferenceSettings(BaseModel):
    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())

    manifest_path: Optional[str] = Field(None, description="Path to the input manifest file")
    model_path: Optional[str] = Field(None, description="Path to the trained AutoEncoder model file")
    inference_results_dir: Optional[str] = Field(
        None, description="path of the directory where inference results are saved"
    )


class EncoderInferenceSettings(BaseModel):
    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())

    manifest_paths: List[str] = Field(
        [], description="One or more paths to manifest files, processed inorder"
    )
    model_path: Optional[str] = Field(None, description="Path to the trained AutoEncoder model file")
    inference_results_dir: Optional[str] = Field(
        None, description="path of the directory where inference results are saved"
    )


class PredictionSettings(BaseModel):
    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())

    manifest_path: Optional[str] = Field(None, description="Path to the input manifest file")
    ae_model_path: Optional[str] = Field(None, description="Path to the trained AutoEncoder model file")
    mgn_model_path: Optional[str] = Field(None, description="Path to the trained MeshGraphNet model file")

    results_dir: Optional[str] = Field(
        None,
        description="Optional path of the directory where results are saved. If none, use top-level output-dir.",
    )

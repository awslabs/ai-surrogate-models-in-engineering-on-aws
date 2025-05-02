# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import importlib.resources  # nosemgrep: python37-compatibility-importlib2

from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Optional, Dict
from pathlib import Path

from mlsimkit.common.schema.cli import CliExtras


def get_module_path(module, resource):
    """
    Get the path into a module resources. Supports --editable and non-editable installs.
    """
    return importlib.resources.files(module) / resource


# Config directory is packaged with mlsimkt. Used for defaults out-of-the-box.
CONF_DIR = get_module_path("mlsimkit", "conf")


class Level(Enum):
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARN = logging.WARN
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    @classmethod
    def _missing_(cls, name):
        """allow construction from string names"""
        return cls[name]


# logging.getLevelNamesMapping is 3.11+
Level.name_mapping: Dict[str, "Level"] = {level.name: level.value for level in Level}


class LogConfig(BaseModel):
    # `model_config` is a special field unseen by users
    # See https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_config
    # NOTE: use_enum_values=True must be set in model_config to use strings as inputs for NamedEnum fields
    model_config = ConfigDict(use_enum_values=True)

    use_config_file: bool = Field(
        default=True, description="Flag to use YAML log config file, otherwise console logging only."
    )
    config_file: Optional[Path] = Field(
        default=CONF_DIR / "logging.yaml", description="Path to YAML log config file"
    )
    prefix_dir: Optional[Path] = Field(
        None,
        description="Prefix path prepended to filenames in YAML log config. (Default <output_dir>/logs/)",
    )

    level: Optional[Level] = Field(
        default=Level.INFO.name, description="Level for console logging", cli=CliExtras(use_enum_name=True)
    )

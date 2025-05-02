# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import mlflow

from .schema.example import Settings, MoreSettings

log = logging.getLogger(__name__)


def run_preprocess(config: Settings, more_settings: MoreSettings):
    log.info("Running preprocess function: %s", config)
    mlflow.log_param("preprocess.param1", 0.5)
    mlflow.log_metric("preprocess.metric1", 0.92)

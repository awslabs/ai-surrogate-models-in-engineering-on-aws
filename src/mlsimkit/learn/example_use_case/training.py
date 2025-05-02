# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .schema.example import Settings

import logging

log = logging.getLogger(__name__)


def run_train(config: Settings, another_option: str):
    log.info("Running training function: %s", config)

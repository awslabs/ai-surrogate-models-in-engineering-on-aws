# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import pytest

#
# Simple list of tests for all commands for now to avoid regressions with CLIs
# Deep output-level checks belong with use cases codes.
#

# Define the commands as a list of tuples (command, expected exit code)
commands = [
    # Commands without args and --help all exit without error
    ("mlsimkit-manifest", "0"),
    ("mlsimkit-manifest --help", "0"),
    ("mlsimkit-learn", "0"),
    ("mlsimkit-learn --help", "0"),
    ("mlsimkit-learn slices", "0"),
    ("mlsimkit-learn slices --help", "0"),
    ("mlsimkit-learn slices preprocess --help", "0"),
    ("mlsimkit-learn slices train --help", "0"),
    ("mlsimkit-learn slices inference --help", "0"),
    ("mlsimkit-learn kpi", "0"),
    ("mlsimkit-learn kpi --help", "0"),
    ("mlsimkit-learn kpi preprocess --help", "0"),
    ("mlsimkit-learn kpi train-image-encoder --help", "0"),
    ("mlsimkit-learn kpi inspect-image-encoder --help", "0"),
    ("mlsimkit-learn kpi process-mesh-data --help", "0"),
    ("mlsimkit-learn kpi train-prediction --help", "0"),
    ("mlsimkit-learn kpi predict --help", "0"),
]


@pytest.mark.parametrize("command, expected_output", commands)
def test_commands_with_help_or_no_args(command, expected_output):
    """Test that running the given command produces the expected output."""
    try:
        # fmt: off
        # nosemgrep: dangerous-subprocess-use-audit
        output = subprocess.check_output(
            command.split(), stderr=subprocess.STDOUT, universal_newlines=True
        )  
        # nosemgrep: insecure-subprocess-use
        output += subprocess.check_output(
            "echo $?", shell=True, universal_newlines=True
        )
        # fmt: on
    except subprocess.CalledProcessError as e:
        output = e.output

    assert expected_output in output

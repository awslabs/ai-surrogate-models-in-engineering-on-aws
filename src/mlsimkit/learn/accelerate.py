# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import subprocess
import sys

from typing import List


def parse_args(args: List[str]) -> List[str]:
    """
    Split args into two lists 1) before '--launch-args' and 2) after
    """
    accelerate_launch_args = []
    command_args = []
    encountered_flag = False

    for arg in args:
        if arg == "--launch-args":
            encountered_flag = True
            continue

        if encountered_flag:
            accelerate_launch_args.append(arg)
        else:
            command_args.append(arg)

    return accelerate_launch_args, command_args


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--dry-run", is_flag=True, default=False, help="Print the command only, don't run it")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def accelerate(ctx, dry_run, args):
    """
    Run 'mlsimkit-learn' commands with Hugging Face Accelerate for multi-GPU training.

    NOTE: Use with training commands ONLY. Non-training commands are not supported.

    MLSimKit integrates training with Hugging Face Accelerate to enable and launch multi-GPU training.
    This can significantly speed up the training process when multiple GPUs are available.

    This command invokes 'mlsimkit-learn' and passes directly through. For more information on
    the available commands and options, run 'mlsimkit-learn --help'.

    You may pass additional arguments to Accelerate using '--launch-args'. These arguments must
    be provided after the 'mlsimkit-learn' command and its arguments:

        mlsimkit-accelerate --config <config.yaml> kpi train [command_args] --launch-args <additional accelerate launch args>

    For example, the following limits to 2 GPUs:

        mlsimkit-accelerate --config train.yaml kpi train --launch-args --num_processes 2

    runs the following command:

        accelerate launch --no-python --num_processes 2 mlsimkit-learn --config train.yaml kpi train

    You may invoke 'accelerate launch' directly but always specify '--accelerate-mode' to ensure
    logging from multiple files works as expected; and do not use '--accelerate-mode'
    outside 'accelerate launch'.
    """
    accelerate_launch_args, command_args = parse_args(args)

    accelerate_cmd = [
        "accelerate",
        "launch",
        "--no-python",
    ]

    command_cmd = [
        "mlsimkit-learn",
        "--accelerate-mode",
    ]

    # Add the sub-command arguments
    command_cmd.extend(command_args)
    # Add any additional arguments for accelerate launch
    accelerate_cmd.extend(accelerate_launch_args)
    accelerate_cmd.extend(command_cmd)

    if dry_run:
        print(f"Would run command: '{' '.join(accelerate_cmd)}'")  # Uncomment to print the command
    else:
        print(f"Running command: '{' '.join(accelerate_cmd)}'")  # Uncomment to print the command
        result = subprocess.run(accelerate_cmd, stdout=None, stderr=None, text=True)
        sys.exit(result.returncode)

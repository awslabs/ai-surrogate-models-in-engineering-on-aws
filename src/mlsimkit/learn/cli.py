# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import warnings

from pathlib import Path

from .kpi import cli as kpi_cli
from .slices import cli as slices_cli
from .surface import cli as surface_cli

from .accelerate import accelerate  # noqa

import mlsimkit
import mlsimkit.learn.common.tracking as tracking
from mlsimkit._version import version as __version__
from mlsimkit.common.logging import LogConfig, configure_logging, getLogger
from mlsimkit.common.schema import MLFlowConfig
from mlsimkit.learn.common.schema.project import BaseProjectContext

from .accelerate import accelerate  # noqa

log = getLogger(__name__)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch_geometric.*", message=".*torch-(scatter|sparse)"
)


@mlsimkit.cli.program(
    name="ML for Simulation Toolkit", version=__version__, use_config_file=True, use_debug_mode=True
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, path_type=Path),
    help="Base directory for output artifacts.",
)
@mlsimkit.cli.options(MLFlowConfig, dest="mlflow_settings", help_group="Tracking")
@mlsimkit.cli.options(LogConfig, dest="log_config", prefix="log", help_group="Logging")
@click.option(
    "--accelerate-mode/--no-accelerate-mode",
    is_flag=True,
    default=False,
    help="Required ONLY with 'accelerate launch' (multi-GPU). Manages multi-process communications when training such as hiding duplicate logs.",
)
def learn(
    ctx: click.Context,
    output_dir: click.Path,
    log_config: LogConfig,
    mlflow_settings: MLFlowConfig,
    config_file: click.Path,
    debug_mode: bool,
    accelerate_mode: bool,
):
    """
    The ML for Simulation Toolkit (MLSimKit) provides engineers and designers with a starting point
    for near real-time predictions of physics-based simulations using ML models. It enables engineers
    to quickly iterate on their designs and see immediate results rather than having to wait hours
    for a full physics-based simulation to run.

    The toolkit is a collection of commands and SDKs that insert into the traditional iterative
    design-simulate workflow, where it can take days to run a single cycle. Instead, the
    "train-design-predict" workflow becomes an additional choice to train models on simulated ground
    truth and then use the newly trained models for predictions on new designs in minutes.

    For more information, please refer to the MLSimKit documentation provided.
    """
    # init the context for sub-commands because their ProjectContext expect this
    BaseProjectContext.init(ctx, output_dir)

    ctx.obj["debug-mode"] = debug_mode

    if ctx.obj["invoked_with_help"]:
        print("[MLSimKit] Learning Tools")
        print(f"Package Version: {__version__}")
        return

    #
    # Everything after here occurs when --help is NOT requested on learn().
    #   Note `--help` on a subcommand still invokes the subcommand function.
    #

    # only create output directories after --help
    if not output_dir:
        raise click.UsageError("Missing option 'output-dir'")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # commands need to check for 'accelerate launch'
    ctx.obj["invoked_with_accelerate"] = accelerate_mode

    # logs default to a subdirectory under the output directory, and enabling buffering
    # with accelerate so we can log from the main process only correctly
    log_config.prefix_dir = log_config.prefix_dir or output_dir / "logs"
    configure_logging(log_config, use_log_buffering=ctx.obj["invoked_with_accelerate"])

    # MLFlow defaults to a subdir of output dir.
    mlflow_settings.tracking_uri = mlflow_settings.tracking_uri or output_dir / "mlruns"

    # Subcommands configure MLFlow via tracking.py, we defer so multi-gpu processes share the same run ID.
    ctx.obj["mlflow-settings"] = mlflow_settings if mlflow_settings.experiment_name else None

    # Ensure all commands can access the config-file (when specified)
    ctx.obj["config_file"] = str(Path(config_file).resolve()) if config_file else None

    def after_commands():
        # NOTE: The entire logs folder is uploaded even if there are existing files
        # from previous runs.
        tracking.log_artifacts(log_config.prefix_dir.as_posix())
        log.debug("Exiting command")

    ctx.call_on_close(after_commands)

    log.info("[MLSimKit] Learning Tools")
    log.info("Package Version: %s", __version__)

    #
    # Sub-commands now invoked...
    #


# Sub-commands
learn.add_command(kpi_cli.kpi)
learn.add_command(slices_cli.slices)
learn.add_command(surface_cli.surface)
# learn.add_command(manifest_cli)

# Example use case is commented out so users don't see it.
# learn.add_command(example_cli.example_use_case)

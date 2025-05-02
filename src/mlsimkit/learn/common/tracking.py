# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import mlflow
from accelerate.state import PartialState
from mlsimkit.common.logging import getLogger
from mlsimkit.common.schema import MLFlowConfig
from mlsimkit.learn.common.schema.project import BaseProjectContext

ARTIFACT_ROOT = ""
METRIC_ROOT = ""
OUTPUT_ROOT = Path("")

# used to check if mlflow is enabled
_MLFLOW = None

log = getLogger(__name__)


def configure(ctx):
    """Configure MLFlow tracking for the given context.

    This function configures MLFlow tracking for the main process and worker processes.
    For the main process, it sets up the MLFlow experiment and run, and writes the run ID
    to the project file. For worker processes, it waits for the main process to write the
    run ID, and then configures MLFlow with the same run ID.

    Warning: This assumes accelerate.Accelerator() has been initialized. If not, accelerate
             settings will default since we get the PartialState() instance. For now, this
             works because we are careful in the train CLI functions, but it's brittle.

    Args:
        ctx (click.Context): The click context object.
    """
    global _MLFLOW
    if ctx.obj.get("mlflow-settings", None):
        if PartialState._shared_state == {} or PartialState().is_main_process:
            mlflow_settings = ctx.obj["mlflow-settings"]
            if mlflow_settings.resume_run:
                project = ctx.obj[".project"].load(ctx)  # load from file after waiting
                mlflow_settings.run_id = project.run_id  # use existing run ID to configure mlflow
            mlflow_settings = configure_mlflow(mlflow_settings)
            ctx.obj["mlflow"] = mlflow_settings
            _MLFLOW = ctx.obj.get("mlflow", None)
            project = BaseProjectContext.get(ctx)
            project.run_id = mlflow_settings.run_id
            project.save(ctx)  # write run ID to file, signals to workers that are waiting
            log.debug("Configuring tracking (main process): project='%s'", project)
        else:
            PartialState().wait_for_everyone()  # wait until main has written run ID
            project = BaseProjectContext.load(ctx)  # load from file after waiting
            log.debug("Configuring tracking (worker): project='%s'", project)
            mlflow_settings = ctx.obj["mlflow-settings"]
            mlflow_settings.run_id = project.run_id  # use existing run ID to configure mlflow
            mlflow_settings = configure_mlflow(mlflow_settings)
            ctx.obj["mlflow"] = mlflow_settings
            _MLFLOW = ctx.obj.get("mlflow", None)


def configure_mlflow(settings: MLFlowConfig):
    """Configure MLFlow with the given settings.

    This function sets up the MLFlow experiment and run based on the provided settings.
    If an experiment with the given name does not exist, it creates a new experiment.
    If a run is not already active, it starts a new run with the specified run ID (if provided).

    Args:
        settings (MLFlowConfig): The MLFlow configuration settings.

    Returns:
        MLFlowConfig: The updated MLFlow configuration settings.
    """
    if settings.experiment_name is None:
        return None

    log.debug("Connecting MLFlow tracking URI: %s...", settings.tracking_uri)
    mlflow.set_tracking_uri(settings.tracking_uri)

    active_run = mlflow.active_run()
    resuming_run = bool(settings.run_id)

    settings.experiment = mlflow.get_experiment_by_name(settings.experiment_name)
    if not settings.experiment:
        if not resuming_run:
            # create a new experiment
            settings.experiment = mlflow.get_experiment(mlflow.create_experiment(settings.experiment_name))
        else:
            raise RuntimeError(f"Cannot resume run, experiment not found: '{settings.experiment_name}'")

    if not active_run:
        mlflow.start_run(experiment_id=settings.experiment.experiment_id, run_id=settings.run_id)
        active_run = mlflow.active_run()
    elif resuming_run and active_run.info.run_id != settings.run_id:
        # we could instead end the run and start a new run, but we don't know of a use case
        raise RuntimeError(
            f"MLFLow run '{active_run.info.run_id}' is already active, different to settings '{settings.run_id}'"
        )
    elif settings.experiment_name != mlflow.get_experiment(active_run.info.experiment_id).name:
        raise RuntimeError(
            f"MLFLow experiment '{mlflow.get_experiment(active_run.info.experiment_id).name}' (id={active_run.info.experiment_id}) is already active, different to settings '{settings.experiment_name}'"
        )
    # else continue using existing run, support resume-run and chaining commands

    settings.run = active_run
    settings.run_id = active_run.info.run_id
    settings.experiment_id = active_run.info.experiment_id

    mf = settings  # convenience to reduce text
    log.info("Experiment: %s (id=%s)", mf.experiment.name, mf.experiment_id)
    log.info(
        "%s run: %s (id=%s)",
        "Resuming" if resuming_run else "Starting",
        mf.run.info.run_name,
        mf.run.info.run_uuid,
    )
    log.info("Output URI: %s", mf.run.info.artifact_uri)

    if not mf.is_local_tracking:  # nosemgrep: is-function-without-parentheses
        # Link user to the run the URL for convenience
        # TODO: Instructions for local UI, it needs an extra step to setup.
        #       See https://mlflow.org/docs/latest/tracking.html
        url = f"{mf.tracking_uri}/#/experiments/{mf.experiment.experiment_id}/runs/{mf.run.info.run_id}"
        log.info("Tracking console: %s", url)

    return settings


@contextmanager
def context(ctx, artifact_root, metric_root):
    """Context manager for configuring MLFlow tracking.

    This context manager sets the global ARTIFACT_ROOT and METRIC_ROOT variables,
    configures MLFlow tracking, and restores the variables to their original values
    when the context is exited.

    Example usage::

        # MLFlow is configured with the tracking context
        with tracking.context(ctx, artifact_root='kpi/train', metric_root='kpi.train'):
            run_train(...)

    Args:
        ctx (click.Context): The click context object.
        artifact_root (str): The root path for artifact logging.
        metric_root (str): The root path for metric logging.

    Yields:
        None
    """
    global OUTPUT_ROOT, ARTIFACT_ROOT, METRIC_ROOT
    try:
        OUTPUT_ROOT = Path(ctx.obj[".project"].outdir)
        ARTIFACT_ROOT = artifact_root
        METRIC_ROOT = metric_root
        configure(ctx)
        yield
    finally:
        OUTPUT_ROOT = Path("")
        ARTIFACT_ROOT = ""
        METRIC_ROOT = ""


def should_log():
    """Determine if logging should be performed.

    Logging should be performed in the following cases:
    1. If MLFlow has been configured on start
    2. If the Accelerate library is not initialized (i.e., not during training).
    3. If the Accelerate library is initialized and it's the main process (to avoid duplicate logging during training).

    The Accelerate library is used for distributed training, and during training, only the main process should log
    to MLflow to avoid duplicate logging from worker processes.

    Returns:
        bool: True if logging should be performed, False otherwise.
    """
    global _MLFLOW
    return _MLFLOW and (PartialState._shared_state == {} or PartialState().is_main_process)


def log_artifact_wrapper(func):
    """Wrap the mlflow.log_artifact or mlflow.log_artifacts function.

    This decorator wraps the provided function to conditionally log artifacts
    based on the should_log() function and generates a default artifact path.

    For instance, when artifact path is NOT specified, the parents of the local path root are retained
    relative to the project output root. By keeping the parents, the artifact path is unique just
    like local output files. This depends on the Project output directory (OUTPUT_ROOT) and the
    tracking context's ARTIFACT_ROOT.

    For example::

        OUTPUT_ROOT = "outputs/training"
        ARTIFACT_ROOT = "kpi/train"

        # Log a single file with a custom artifact path
        log_artifact("/path/to/local/file.txt", "custom/artifact/path/file.txt")

        # Log a directory with an automatically generated artifact path
        log_artifact("outputs/training/path/to/local/dir")  # Artifact path will be "kpi/train/path/to/local/dir"

        # Log a file with an automatically generated artifact path
        log_artifact("outputs/training/path/to/local/file.txt")  # Artifact path will be "kpi/train/path/to/local"

    Args:
        func (callable): The function to wrap (mlflow.log_artifact or mlflow.log_artifacts).

    Returns:
        callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapped(local_path: str, artifact_path: Optional[str] = None, **kwargs):
        global OUTPUT_ROOT, ARTIFACT_ROOT

        if not should_log():
            return None

        if isinstance(local_path, Path):
            local_path = str(local_path.resolve())

        if artifact_path:
            artifact_path = os.path.basename(artifact_path)
        elif Path(local_path).is_dir():
            base = OUTPUT_ROOT
            artifact_path = os.path.join(ARTIFACT_ROOT, Path(local_path).relative_to(base))
        else:
            base = OUTPUT_ROOT
            artifact_path = os.path.join(ARTIFACT_ROOT, Path(local_path).relative_to(base).parent)

        return func(local_path, artifact_path, **kwargs)

    return wrapped


def log_metric_wrapper(func):
    """Wrap the mlflow.log_metric and mlflow.log_param functions.

    This decorator wraps the mlflow.log_metric/mlflow.log_param function to conditionally log metrics/params
    based on the should_log() function and the global METRIC_ROOT setting.

    Args:
        func (callable): The mlflow.log_metric or mlflow.log_param functions.

    Returns:
        callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapped(key, value, **kwargs):
        global METRIC_ROOT
        return func(METRIC_ROOT + "." + key, value, **kwargs) if should_log() else None

    return wrapped


def log_metrics_wrapper(func):
    """Wrap the mlflow.log_metrics and mlflow.log_params functions.

    This decorator wraps the mlflow.log_metric function to conditionally log metrics
    based on the should_log() function and the global METRIC_ROOT setting.

    Args:
        func (callable): The mlflow.log_metrics or mlflow.log_params functions.

    Returns:
        callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapped(inputs, **kwargs):
        global METRIC_ROOT
        return (
            func({METRIC_ROOT + "." + key: value for key, value in inputs.items()}, **kwargs)
            if should_log()
            else None
        )

    return wrapped


log_artifact = log_artifact_wrapper(mlflow.log_artifact)
log_artifacts = log_artifact_wrapper(mlflow.log_artifacts)
log_metric = log_metric_wrapper(mlflow.log_metric)
log_param = log_metric_wrapper(mlflow.log_param)
log_metrics = log_metrics_wrapper(mlflow.log_metrics)
log_params = log_metrics_wrapper(mlflow.log_params)

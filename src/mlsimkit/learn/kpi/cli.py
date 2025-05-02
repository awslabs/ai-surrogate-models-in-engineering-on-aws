# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

import click
import mlsimkit
import mlsimkit.learn.common.tracking as tracking
from mlsimkit.common.logging import getLogger
from mlsimkit.learn.manifest import split_manifest as run_split_manifest
from mlsimkit.learn.manifest.schema import SplitSettings
from mlsimkit.learn.common.schema.project import BaseProjectContext
from pydantic import ConfigDict

from .schema.inference import InferenceSettings
from .schema.preprocessing import PreprocessingSettings
from .schema.training import TrainingSettings

log = getLogger(__name__)


class ProjectContext(BaseProjectContext):
    """
    Persist outputs for chaining commands.
    """

    # original input manifest
    manifest_path: Optional[str] = None

    # working manifests
    train_manifest_path: Optional[str] = None
    validation_manifest_path: Optional[str] = None
    test_manifest_path: Optional[str] = None

    model_path: Optional[str] = None

    run_id: Optional[str] = None

    output_kpi_indices: Optional[str] = None

    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())


@mlsimkit.cli.group(chain=True)
@click.option(
    "--manifest-uri",
    type=mlsimkit.cli.ResourcePath(search_paths=["mlsimkit.datasets"]),
    help="Manifest file to drive all subcommands. Searches mlsimkit/datasets as a fallback",
)
def kpi(ctx: click.Context, manifest_uri: click.Path):
    """
    Use Case: KPI prediction via a variant of MeshGraphNets (MGN)
    """
    log.info("Use Case: KPI prediction via a variant of MeshGraphNets")

    if ctx.obj["invoked_with_help"]:
        return

    # load the .project file from disk and into the context
    project = ProjectContext.load(ctx)
    if manifest_uri:
        project.manifest_path = str(Path(manifest_uri).resolve())
        project.save(ctx)


@kpi.command()
@mlsimkit.cli.options(PreprocessingSettings, dest="settings")
@mlsimkit.cli.options(SplitSettings, dest="split_settings", help_group="Split Manifest")
@click.option("--split-manifest/--no-split-manifest", is_flag=True, default=True)
def preprocess(
    ctx: click.Context, settings: PreprocessingSettings, split_manifest: bool, split_settings: SplitSettings
):
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.load(ctx)
    settings.output_dir = os.path.join(project.outdir, settings.output_dir)
    settings.manifest_path = settings.manifest_path or project.manifest_path

    # manifest is required, so check here to allow for config loading, and because
    #  click.option required=True are not supported with config loading (TODO)
    if not settings.manifest_path:
        raise click.UsageError("Missing option --manifest-path")
    if not Path(settings.manifest_path).exists():
        raise click.UsageError(f"--manifest-path '{settings.manifest_path}' not found")

    from .preprocessing import run_preprocess

    with tracking.context(ctx, artifact_root="kpi/preprocess", metric_root="kpi.preprocess"):
        working_manifest_path = run_preprocess(settings, project.outdir)

    if split_manifest:
        log.info(
            "Splitting manifest into train-size=%s valid-size=%s test-size=%s",
            split_settings.train_size,
            split_settings.valid_size,
            split_settings.test_size,
        )
        manifests = run_split_manifest(working_manifest_path, split_settings, Path(project.outdir))

        # set project context for other commands to get these manifests
        project.train_manifest_path = manifests.get("train", None)
        project.validation_manifest_path = manifests.get("validation", None)
        project.test_manifest_path = manifests.get("test", None)
    else:
        # Prediction defaults to the test manifest path, so when not splitting for
        # predict-only workflows, pass the input manifest to prediction.
        # FIXME: need an explicit intent when sharing manifest with prediction, implicitly
        #        relying on the 'test' is opaque
        project.test_manifest_path = str(working_manifest_path.resolve())

    project.save(ctx)


@kpi.command()
@mlsimkit.cli.options(TrainingSettings, dest="settings")
def train(ctx: click.Context, settings: TrainingSettings):
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.load(ctx)
    project.output_kpi_indices = settings.output_kpi_indices
    settings.train_manifest_path = settings.train_manifest_path or project.train_manifest_path
    settings.validation_manifest_path = settings.validation_manifest_path or project.validation_manifest_path

    settings.training_output_dir = settings.training_output_dir or os.path.join(
        project.outdir, "training_output"
    )

    from mlsimkit.learn.common.training import make_accelerator, validate_training_settings

    from .training import run_train

    validate_training_settings(settings, ctx)
    accelerator = ctx.obj.get("accelerator", make_accelerator(settings))

    with tracking.context(ctx, artifact_root="kpi/train", metric_root="kpi.train"):
        run_train(settings, accelerator)

    project.model_path = f"{settings.training_output_dir}/best_model.pt"
    project.save(ctx)


@kpi.command()
@mlsimkit.cli.options(InferenceSettings, dest="settings")
@click.option(
    "--compare-groundtruth/--no-compare-groundtruth",
    is_flag=True,
    default=False,
    help="Compare predictions against groundtruth, outputs result metrics and error images. Expects the manifest to contain KPI values.",
)
@click.argument(
    "mesh-files", type=click.Path(exists=True, dir_okay=False, resolve_path=True), nargs=-1, required=False
)
def predict(ctx: click.Context, settings: InferenceSettings, compare_groundtruth, mesh_files):
    """
    Perform inference on the provided mesh files or preprocessed files, or use a manifest if no files are provided.

    Usage examples:

    \b
    # Use mesh files for prediction
    mlsimkit-learn --output-dir predictions/ kpi predict --model-path path/to/best_model.pt path/to/datasets/ahmed/run_10*/*.stl

    \b
    # Use preprocessed files for prediction
    mlsimkit-learn --output-dir predictions/ kpi predict --model-path path/to/best_model.pt path/to/preprocessed_data/preprocessed_run_000*[234].pt

    \b
    # Use a mix of preprocessed files and mesh files for prediction
    mlsimkit-learn --output-dir predictions/ kpi predict --model-path path/to/best_model.pt path/to/preprocessed_data/preprocessed_run_000*[234].pt path/to/datasets/ahmed/run_10*/*.stl

    \b
    # Use a manifest for prediction and compare to groundtruth KPI values
    mlsimkit-learn --output-dir predictions/ kpi predict --model-path path/to/best_model.pt --compare-groundtruth --manifest-path path/to/manifest.json
    """
    log.info(f"Running command '{ctx.info_name}'")
    project = ProjectContext.load(ctx)

    settings.model_path = settings.model_path or os.path.join(project.outdir, "training_output/best_model.pt")
    settings.manifest_path = settings.manifest_path or project.test_manifest_path
    settings.inference_results_dir = settings.inference_results_dir or os.path.join(
        project.outdir, "predictions"
    )
    settings.output_kpi_indices = settings.output_kpi_indices or project.output_kpi_indices
    log.info("Settings: %s", settings)
    if len(mesh_files) > 0 and Path(settings.manifest_path).exists():
        log.info("Mesh files specified on the command-line, ignoring any manifest")

    from .inference import run_predict

    with tracking.context(ctx, artifact_root="kpi/predict", metric_root="kpi.predict"):
        run_predict(settings, compare_groundtruth, mesh_files)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import os
from typing import Optional
from pydantic import ConfigDict
from pathlib import Path

import mlsimkit
import mlsimkit.learn.common.tracking as tracking

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.common.schema.project import BaseProjectContext
from mlsimkit.learn.manifest import split_manifest as run_split_manifest
from mlsimkit.learn.manifest.schema import SplitSettings

from .schema.preprocessing import PreprocessingSettings, SurfaceVariables
from .schema.training import TrainingSettings
from .schema.inference import InferenceSettings
from .schema.view import ViewSettings, ScreenshotSettings

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

    # suppress "model_" namespace warnings, see
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())


@mlsimkit.cli.group(chain=True)
@click.option(
    "--manifest-uri",
    type=mlsimkit.cli.ResourcePath(search_paths=["mlsimkit.datasets"]),
    help="Manifest file to drive all subcommands. Searches mlsimkit/datasets as a fallback for quickstart",
)
def surface(ctx: click.Context, manifest_uri: click.Path):
    """
    Use Case: Surface variable prediction via MeshGraphNets (MGN)
    """
    log.info("Use Case: Surface variable prediction via MeshGraphNets (MGN)")

    # note: we don't require manifest here, instead subcommands may check themselves

    # load the .project file from disk and into the context
    project = ProjectContext.load(ctx)
    if manifest_uri:
        project.manifest_path = str(Path(manifest_uri).resolve())
        project.save(ctx)


def validate_filepath(option, filepath, hint=""):
    # check manifests are required outside options() decorator so sub-commands
    # can choose require=True/False.
    if not filepath:
        raise click.UsageError(f"Missing option {option}. {hint}")
    if not Path(filepath).exists():
        raise click.UsageError(f"{option} '{filepath}' not found. {hint}")


@surface.command()
@mlsimkit.cli.options(PreprocessingSettings, dest="settings")
@mlsimkit.cli.options_from_schema_shorthand(
    "-v",
    "--surface-variables",
    model=SurfaceVariables,
    multiple=True,
    help='Selects desired surface variables to predict. Format is "name=<str>,dimensions=<list[int]>". '
    "If not specified, no variable will be selected (suitable for inference). For training, at least one variable must be specified.",
)
@mlsimkit.cli.options(SplitSettings, dest="split_settings", help_group="Split Manifest")
@click.option("--split-manifest/--no-split-manifest", is_flag=True, default=True)
def preprocess(
    ctx: click.Context,
    settings: PreprocessingSettings,
    surface_variables,
    split_manifest: bool,
    split_settings: SplitSettings,
):
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.load(ctx)
    settings.output_dir = os.path.join(project.outdir, settings.output_dir)
    settings.manifest_path = settings.manifest_path or project.manifest_path

    validate_filepath("--manifest-path", settings.manifest_path)

    from .preprocessing import run_preprocess

    with tracking.context(ctx, artifact_root="surface/preprocess", metric_root="surface.preprocess"):
        working_manifest_path = run_preprocess(settings, project.outdir, surface_variables)

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


@surface.command()
@mlsimkit.cli.options(TrainingSettings, dest="settings")
@click.option(
    "--start-xvfb/--no-start-xvfb",
    is_flag=True,
    help="Only available on Linux. Flag for rendering screenshots in headless environments by starting an X virtual framebuffer (Xvfb) using PyVista. "
    "Requires 'libgl1-mesa-glx' and 'xvfb' packages",
)
def train(ctx: click.Context, settings: TrainingSettings, start_xvfb):
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.load(ctx)
    settings.train_manifest_path = settings.train_manifest_path or project.train_manifest_path
    settings.validation_manifest_path = settings.validation_manifest_path or project.validation_manifest_path
    settings.training_output_dir = settings.training_output_dir or os.path.join(
        project.outdir, "training_output"
    )

    validate_filepath(
        "--train-manifest-path", settings.train_manifest_path, hint="Have you run preprocessing?"
    )
    validate_filepath(
        "--validation-manifest-path", settings.validation_manifest_path, hint="Have you run preprocessing?"
    )

    from .training import run_train
    from mlsimkit.learn.common.training import make_accelerator, validate_training_settings

    validate_training_settings(settings, ctx)
    accelerator = ctx.obj.get("accelerator", make_accelerator(settings))

    if start_xvfb:
        import pyvista as pv

        pv.start_xvfb()

    with tracking.context(ctx, artifact_root="surface/train", metric_root="surface.train"):
        run_train(settings, accelerator)

    project.model_path = f"{settings.training_output_dir}/best_model.pt"
    project.save(ctx)


@surface.command()
@mlsimkit.cli.options(InferenceSettings, dest="settings")
def predict(ctx: click.Context, settings: InferenceSettings):
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.load(ctx)
    settings.model_path = settings.model_path or os.path.join(project.outdir, "training_output/best_model.pt")
    settings.manifest_path = settings.manifest_path or project.test_manifest_path
    settings.inference_results_dir = settings.inference_results_dir or os.path.join(
        project.outdir, "predictions"
    )

    validate_filepath("--manifest-path", settings.manifest_path)

    from .inference import run_predict

    with tracking.context(ctx, artifact_root="surface/predict", metric_root="surface.predict"):
        run_predict(settings)


@surface.command()
@mlsimkit.cli.options(ViewSettings, dest="settings")
@mlsimkit.cli.options(ScreenshotSettings, dest="screenshot_settings", help_group="Screenshot (--no-gui)")
@click.option(
    "--manifest",
    type=click.File("r"),
    required=False,
    help="Optional manifest file or use '-' to read JSON Lines from stdin. If unspecified, defaults to the project training manifest.",
)
@click.option(
    "--start-xvfb/--no-start-xvfb",
    is_flag=True,
    help="Only available on Linux. Flag for rendering screenshots in headless environments by starting an X virtual framebuffer (Xvfb) using PyVista. "
    "Requires 'libgl1-mesa-glx' and 'xvfb' packages",
)
def view(
    ctx: click.Context, manifest, settings: ViewSettings, screenshot_settings: ScreenshotSettings, start_xvfb
):
    from .data import SurfaceDataset
    from .visualize import Viewer, take_screenshots

    project = ProjectContext.load(ctx)

    if manifest is None:
        validate_filepath(
            "'surface view --manifest' or 'surface --manifest-uri'", project.train_manifest_path
        )

    manifest = manifest.read() if manifest else project.train_manifest_path
    dataset = SurfaceDataset(manifest, device="cpu")

    if settings.describe:
        print("Number of runs: ", len(dataset))
        print("Surface variables: ", dataset.surface_variables())
        print("Predictions available? ", bool("predicted_file" in dataset.manifest))
        return

    if start_xvfb and settings.gui:
        raise click.UsageError(
            "Must render screenshots (via --no-gui) when using --start-xvbf on remote machines, cannot open GUI display"
        )

    if start_xvfb:
        import pyvista as pv

        pv.start_xvfb()

    viewer = Viewer(dataset, interactive=settings.gui, views=settings.views)
    viewer.start()
    if not viewer.interactive:
        screenshot_settings.outdir = Path(project.outdir) / screenshot_settings.outdir
        take_screenshots(viewer, screenshot_settings)

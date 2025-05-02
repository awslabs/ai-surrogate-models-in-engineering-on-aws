# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

import click
import mlsimkit
import mlsimkit.learn.common.tracking as tracking
from mlsimkit.common.logging import getLogger
from mlsimkit.learn.common.schema.project import BaseProjectContext
from mlsimkit.learn.manifest import get_base_path, split_manifest
from mlsimkit.learn.manifest.schema import RelativePathBase, SplitSettings

from .schema.inference import EncoderInferenceSettings, InferenceSettings, PredictionSettings
from .schema.preprocessing import ConvertImageSettings
from .schema.training import TrainAESettings, TrainMGNSettings

log = getLogger(__name__)


class ProjectContext(BaseProjectContext):
    """
    Persists output locations for chaining commands.
    """

    train_manifest_path: Optional[str] = None
    validation_manifest_path: Optional[str] = None
    test_manifest_path: Optional[str] = None
    ae_model_path: Optional[str] = None
    mgn_model_path: Optional[str] = None


@mlsimkit.cli.group(chain=True)
def slices(ctx: click.Context):
    """
    Use Case: Slice Prediction

    Slice Prediction is used to predict parameters from slices for 3D geometry meshes. A slice is a 2D cross-sectional plane cut through
    the 3D geometry and volume that captures parametrics stored as an image file. In computational fluid dynamics (CFD), common parametrics
    include velocity and pressure.

    Training Slice Prediction requires slice images to be already prepared. The sample dataset and public datasets provide slices for you.See the
    Slices user guide for details how to create custom slices for your own geometry data and corresponding simulated ground truth (for training).

    There are four steps as sub-commands to fully train the slice prediction model. Once you have completed training, you can predict the
    parameters used during training for new unseen geometry using the predict command.

    Training and prediction requires a user manifest file to reference a dataset. See the Manifest user guide for details. An example manifest
    file is provide in datasets/ahmed-sample/slices.manifest.

    The output directory is used as a prefix for all sub-commands. You may use YAML as a configuration file instead of using command-line options.
    A example configuration file is provided in conf/slices/sample.yaml.
    """
    # load the .project file from disk and into the context
    ProjectContext.load(ctx)


@slices.command()
@click.option(
    "--manifest-uri",
    type=mlsimkit.cli.ResourcePath(search_paths=["mlsimkit.datasets"]),
    help="Manifest file for slices. Searches mlsimkit/datasets as a fallback",
)
@mlsimkit.cli.options(ConvertImageSettings, dest="settings")
@mlsimkit.cli.options(SplitSettings, dest="split_settings", help_group="Split Manifest")
@click.option("--split-manifest/--no-split-manifest", "should_split_manifest", default=True)
@click.option(
    "--manifest-base-relative-path",
    default="CWD",
    type=click.Choice([e.name for e in RelativePathBase]),
    help="Base directory for all files that are relative paths within the manifest",
)
def preprocess(
    ctx: click.Context,
    manifest_uri: click.Path,
    settings: ConvertImageSettings,
    should_split_manifest: bool,
    split_settings: SplitSettings,
    manifest_base_relative_path,
):
    """
    Step (1): Process input data and prepare manifests

    Slice prediction model predicts slices directly from a 3D geometry mesh. The first step is to preprocess the slice image files,
    converting them into data objects that can be easily consumed. The input manifest is copied and split into three datasets for the subsequent
    training, validation, and testing stages.
    """
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.get(ctx)

    # manifest is required, so check here to allow for config loading, and because
    #  click.option required=True are not supported with config loading (TODO)
    if not manifest_uri:
        raise click.UsageError("Missing option --manifest-uri")
    if not Path(manifest_uri).exists():
        raise click.UsageError(f"--manifest-uri '{manifest_uri}' not found")

    input_manifest_path = Path(manifest_uri)

    # lazy load module
    from .preprocessing import run_preprocessing

    base_relative_path = get_base_path(manifest_base_relative_path, manifest_uri)
    with tracking.context(ctx, artifact_root="slices/preprocess", metric_root="slices.preprocess"):
        working_manifest_path = run_preprocessing(
            input_manifest_path, Path(project.outdir), base_relative_path, settings
        )

        if should_split_manifest:
            log.info(
                "Splitting manifest into train-size=%s valid-size=%s test-size=%s",
                split_settings.train_size,
                split_settings.valid_size,
                split_settings.test_size,
            )
            manifests = split_manifest(working_manifest_path, split_settings, Path(project.outdir))

            # set project context for other commands to get these manifests
            project.train_manifest_path = manifests.get("train", None)
            project.validation_manifest_path = manifests.get("validation", None)
            project.test_manifest_path = manifests.get("test", None)
        else:
            # Prediction defaults to the test manifest path, so when not splitting for
            # predict-only workflows, pass the input manifest to prediction.
            # FIXME: need an explicit intent when sharing manifest with prediction, implicitly
            #        relying on the 'test' is opaque
            project.test_manifest_path = str(input_manifest_path)

    project.save(ctx)


# Step 2: Train AutoEncoder
@slices.command()
@mlsimkit.cli.options(TrainAESettings, dest="settings")
def train_image_encoder(ctx: click.Context, settings: TrainAESettings):
    """
    Step (2): Train image encoding model

    Training the image encoder step is where the machine learning model learns to compress the image slices and reconstruct that compression.
    It takes the preprocessed data as input and outputs models for subsequent steps including training the final slice prediction model. There
    are a number of hyper-parameters associated with model training of the AE, and all of them have default values.

    The training step outputs to <output_directory>/ae/training_output/. Files include the models saved as checkpoints, the best performing model,
    and loss plots as images.
    """
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.get(ctx)
    settings.train_manifest_path = settings.train_manifest_path or project.train_manifest_path
    settings.validation_manifest_path = settings.validation_manifest_path or project.validation_manifest_path
    settings.training_output_dir = settings.training_output_dir or os.path.join(
        project.outdir, "ae", "training_output"
    )

    assert settings.train_manifest_path and Path(settings.train_manifest_path).exists()
    assert settings.validation_manifest_path and Path(settings.validation_manifest_path).exists()

    from mlsimkit.learn.common.training import make_accelerator, validate_training_settings

    from .training import run_train_ae

    validate_training_settings(settings, ctx)
    ctx.obj["accelerator"] = ctx.obj.get("accelerator", make_accelerator(settings))

    with tracking.context(ctx, artifact_root="slices/training", metric_root="slices.ae.training_output"):
        run_train_ae(settings, ctx.obj["accelerator"])

    project.ae_model_path = f"{settings.training_output_dir}/best_model.pt"
    project.save(ctx)


# (Optional) AutoEncoder inference
@slices.command()
@mlsimkit.cli.options(InferenceSettings, dest="settings")
def inspect_image_encoder(ctx: click.Context, settings: InferenceSettings):
    """
     (Debug): Evaluate image encoder performance

    Once image encoder model training is complete (step 2), you can optionally run inference to verify that the model can adequately
    encode and decode (or reconstruct) the image slices. This is the same model used to reconstruct images from prediction data.
    This step reconstructs the original image slice data and saves the output to <output_dir>/ae/inference_output/. Result metrics are
    also output.
    """
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.get(ctx)
    settings.manifest_path = settings.manifest_path or project.train_manifest_path
    settings.model_path = settings.model_path or project.ae_model_path
    settings.inference_results_dir = settings.inference_results_dir or os.path.join(
        project.outdir, "ae", "inference_output"
    )

    assert settings.manifest_path
    assert settings.model_path

    from .inference import run_inference_ae

    with tracking.context(ctx, artifact_root="slices/training", metric_root="slices.ae.inference_output"):
        run_inference_ae(settings)


# Step 3: Encoder inference
@slices.command()
@mlsimkit.cli.options(EncoderInferenceSettings, dest="settings")
def process_mesh_data(ctx: click.Context, settings: EncoderInferenceSettings):
    """
    Step (3): Link the geometry and image training data

    Once the training for the image encoder is adequate, process the mesh data together with the image data. This step combines the geometry
    files from the manifest and the preprocessed outputs from step (1) into an 'encoding' using inference on the image encoder model from step (2).

    After this step is complete, you are ready to train the prediction model (step 4).
    """
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.get(ctx)
    settings.manifest_paths = settings.manifest_paths or [
        project.train_manifest_path,
        project.validation_manifest_path,
    ]
    settings.model_path = settings.model_path or project.ae_model_path
    settings.inference_results_dir = settings.inference_results_dir or os.path.join(
        project.outdir, "ae", "inference_output"
    )

    assert settings.manifest_paths
    assert settings.model_path

    from .inference import run_inference_encoder

    with tracking.context(ctx, artifact_root="slices/training", metric_root="slices.ae.inference_output"):
        run_inference_encoder(settings)


# Step 4: train MGN
@slices.command()
@mlsimkit.cli.options(TrainMGNSettings, dest="settings")
def train_prediction(ctx: click.Context, settings: TrainMGNSettings):
    """
    Step (4): Train prediction using encoder outputs

    Once step (3) is complete, and the image encoder model and mesh data is processed, you can train the final prediction model. Training
    the full slice prediction step is where the machine learning model learns to predict image slices from mesh geometry.

    There are a number of hyperparameters associated with training of the full slice prediciton model, and all of them have default values.

    The training step produces a number of output files in the folder <output_directory>/mgn/training_output/. Among them, there are model
    checkpoints including the "best" model (best_model.pt) which has the lowest validation error and is recommended to use for new predictions.
    """
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.get(ctx)
    settings.train_manifest_path = settings.train_manifest_path or project.train_manifest_path
    settings.validation_manifest_path = settings.validation_manifest_path or project.validation_manifest_path
    settings.training_output_dir = settings.training_output_dir or os.path.join(
        project.outdir, "mgn", "training_output"
    )

    assert settings.train_manifest_path and Path(settings.train_manifest_path).exists()
    assert settings.validation_manifest_path and Path(settings.validation_manifest_path).exists()

    from mlsimkit.learn.common.training import make_accelerator, validate_training_settings

    from .training import run_train_mgn

    validate_training_settings(settings, ctx)
    ctx.obj["accelerator"] = ctx.obj.get("accelerator", make_accelerator(settings))

    with tracking.context(ctx, artifact_root="slices/training", metric_root="slices.mgn.training_output"):
        run_train_mgn(settings, ctx.obj["accelerator"])

    project.mgn_model_path = f"{settings.training_output_dir}/best_model.pt"
    project.save(ctx)


# Step 5: Predict using MGN+AE models
@slices.command()
@mlsimkit.cli.options(PredictionSettings, dest="settings")
@click.option(
    "--compare-groundtruth/--no-compare-groundtruth",
    default=True,
    help="Compare predictions against groundtruth, outputs result metrics and error images. Expects the manifest to reference the original slice data output during preprocessing.",
)
def predict(ctx: click.Context, settings: PredictionSettings, compare_groundtruth: bool):
    """
    Step (5): Predict results and evaluate performance

    After completing training, the model can be used to predict slices on new geometry. A manifest is required
    to reference the geometry to predict as well as the two models output during the previous two
    training steps (train-image-encoder and train-prediction).

    By default, prediction will use the test manifest from preprocessing. Prediction will use the entire
    manifest if the preprocessing step does not split the manifest. You may override this behavior by
    explicitly specifying a manifest-path for the predict command.

    You may optionally compare predictions to the original data (ground truth). By doing so, results
    and and error images are output so you can visually inspect performance. To predict new
    geometry without ground truth, turn this off by setting --no-compare-groundtruth.
    """
    log.info(f"Running command '{ctx.info_name}'")

    project = ProjectContext.get(ctx)
    settings.manifest_path = settings.manifest_path or project.test_manifest_path
    settings.ae_model_path = settings.ae_model_path or project.ae_model_path
    settings.mgn_model_path = settings.mgn_model_path or project.mgn_model_path
    settings.results_dir = settings.results_dir or os.path.join(project.outdir, "prediction")

    assert settings.manifest_path and Path(settings.manifest_path).exists()
    assert settings.ae_model_path and Path(settings.ae_model_path).exists()
    assert settings.mgn_model_path and Path(settings.mgn_model_path).exists()

    from .inference import run_prediction

    with tracking.context(ctx, artifact_root="slices", metric_root="slices.predict"):
        run_prediction(settings, Path(settings.results_dir), compare_groundtruth)


# Override command order in --help
# TODO: this is brittle to name changes, we should set the order within the command itself e.g, slices.command(help_order=1)
def list_commands_for_help(self):
    return [
        "preprocess",
        "train-image-encoder",
        "process-mesh-data",
        "train-prediction",
        "predict",
        "inspect-image-encoder",
    ]


slices.list_commands = list_commands_for_help

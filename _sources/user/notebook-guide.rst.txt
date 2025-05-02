==================================================================
Using the MLSimKit SDK Interactively (Notebooks, IPython)
==================================================================

You may use MLSimKit programmatically, without the CLI as a Pythom module.

To access your MLSimKit module from ``ipython``, run:

.. code-block:: bash
    
    % source .venv/bin/activate
    % pip install ipython
    % python -m IPython


Running the Surface Training Pipeline
----------------------------------------

The example below runs the Surface model end-to-end on the sample dataset. 

.. code-block:: python

    #
    # Setup preprocessing using the sample dataset manifest 
    #
    from mlsimkit.learn.surface.schema import *
    from mlsimkit.learn.surface.data import SurfaceDataset
    from mlsimkit.learn.surface.preprocessing import run_preprocess

    settings = PreprocessingSettings(
        manifest_path="tutorials/surface/sample/training.manifest",
        output_dir="notebook_example",
        manifest_base_relative_path="PackageRoot")

    #
    # Run preprocessing convert mesh inputs to training data
    #
    working_manifest = run_preprocess(
        settings, 
        surface_variables=[SurfaceVariables(name='pMean')])

    #
    # Split the preprocessed manifest into train, validation and test sets
    #
    from mlsimkit.learn.manifest.schema import SplitSettings
    from mlsimkit.learn.manifest import split_manifest

    split_settings = SplitSettings(train_size=0.6, valid_size=0.2, test_size=0.2)
    manifests = split_manifest(working_manifest, split_settings, settings.output_dir)

    #
    # Run training using the same settings as tutorials/surface/sample/training.yaml 
    #
    from mlsimkit.learn.surface.schema.training import TrainingSettings
    from mlsimkit.learn.surface.training import run_train
    from mlsimkit.learn.common.training import make_accelerator

    settings = TrainingSettings(
        train_manifest_path=manifests['train'],
        validation_manifest_path=manifests['validation'],
        training_output_dir="notebook_example/training_output",
        device="cpu",
        epochs=5,
        batch_size=1)

    run_train(settings, make_accelerator(settings))

    #
    # Make predictions on the meshes in the test manifest 
    #
    from mlsimkit.learn.surface.schema.inference import InferenceSettings
    from mlsimkit.learn.surface.inference import run_predict

    inference_settings = InferenceSettings(
        model_path="notebook_example/training_output/best_model.pt",
        manifest_path=manifests['test'],
        inference_results_dir="notebook_example/predictions",
        device="cpu")

    run_predict(inference_settings)


You will now have training and predicted outputs. For example, 

.. code-block:: bash

    % find notebook_example/predictions
    notebook_example/predictions
    notebook_example/predictions/pMean_errors_by_geometry.csv
    notebook_example/predictions/results
    notebook_example/predictions/results/predicted_boundary_7_mapped.vtp
    notebook_example/predictions/results/predicted_boundary_5_mapped.vtp
    notebook_example/predictions/results/predicted_boundary_3_mapped.vtp
    notebook_example/predictions/error_metrics.csv


To get the original vs. predict files, load the training or validation manifest into a dataset:

.. code-block:: python

    dataset = SurfaceDataset("notebook_example/validate.manifest", device='cpu')

    # Original mesh file
    print(dataset.data_files(0))
    # prints ['.../datasets/ahmed-sample/mapped_vtps/boundary_6_mapped.vtp']

    # Prediction mesh results
    print(dataset.predicted_file(0))
    # prints '.../notebook_example/training_output/best_model_predictions/validation/results/predicted_boundary_6_mapped.vtp'

To learn more, see :ref:`dev-guide-dataset-interaction`.

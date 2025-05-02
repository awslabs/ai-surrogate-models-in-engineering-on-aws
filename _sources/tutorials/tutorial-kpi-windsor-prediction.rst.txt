.. _tutorial-kpi-windsor-prediction:

Using KPI Prediction on New Geometries
===================================================================

This tutorial demonstrates how to use a trained model from the KPI training pipeline to predict KPI values for new geometries in the :ref:`datasets-windsor`.

If you haven't yet completed the :ref:`training tutorial <tutorial-kpi-windsor-training>`, do so first to output a model ready for predictions.

If you haven't yet completed the :ref:`training tutorial <tutorial-kpi-windsor-training>`, do so first to output a model ready for predictions.

Creating the Manifest
---------------------

The ``run-create-manifest-prediction`` script generates the required manifest for the WindsorML dataset. To create the manifest, run the script pointing to your dataset location:

.. code-block:: shell

   ./run-create-manifest-prediction /path/to/windsor/dataset

This will generate a manifest

- ``prediction.manifest``: Lists of geometry files for predicting the KPIs

You can customize the manifest to use different geometries by editing ``run-create-manifest-prediction.sh`` script:

.. code-block:: shell

   #!/bin/bash

   ...

   # Get a list of run folders for prediction (5 random runs)
   predict_run_folders=($(ls -d "$dataset_prefix/run_"* | shuf -n 5))

   # Create predict.manifest with geometries only (no ground truth KPI data)
   mlsimkit-manifest create -m "prediction.manifest" -f "name=geometry_files,file_glob=*.stl" "${predict_run_folders[@]}"

A manifest is a JSON Lines (``.manifest``) file that lists the paths to the geometry files. Each line in the manifest represents a single data file entry, containing the following keys:

- ``"geometry_files"``: A list of relative or absolute paths to the geometry files (e.g., ``.stl``)

Here's an example manifest entry:

.. code-block:: json

   {
     "geometry_files": ["file:///data/windsor/dataset/run_90/windsor_90.stl"]
   }

This entry lists the path to a single geometry file (``windsor_90.stl``).  Note that ``"kpi"`` does not exist in this manifest, as we intend to demonstrate the case where we do not have ground truth.


Understanding the configuration File
------------------------------------

The KPI prediction pipeline is configured using **prediction.yaml**.

This file configures the prediction step, using the trained models from the training pipeline. Key settings include:

- ``output-dir``: Directory for storing prediction outputs (e.g., predicted values)
- ``kpi.manifest_uri``: Path to the manifest of unseen geometries
- ``kpi.preprocess``: Hyperparameters related to data preprocessing of unseen geometries
- ``kpi.predict``: Hyperparameters related to KPI model inference

To get an introduction to the available configuration options, use the ``mlsimkit-learn kpi --help`` command and the ``--help`` option for each sub-command. This will provide an overview of the options and their purposes, which can be helpful when configuring the prediction pipeline.


Running the Pipeline
--------------------

With the manifest created and the configuration file in place, you can run the KPI prediction pipeline using the provided script:

Run prediction:

.. code-block:: shell

   ./run-prediction

This script executes the necessary commands using the ``prediction.yaml`` configuration file.


Reviewing Results
------------------

During Prediction
~~~~~~~~~~~~~~~~~

The prediction step generates KPI values predicted by the trained model for unseen geometries. The prediction output is located in the ``outputs/prediction/predictions`` directory. Since the prediction manifest contains only geometry files (no ground truth KPI data), the prediction outputs do not include error metrics or actual KPIs for comparison. Here is the prediction output.

- ``prediction_results.csv``: Predicted KPI values for unseen geometries


Next Steps
----------

Dive into the :ref:`KPI user guide <user-guide-kpi>` for detailed information on more configuration options and how they impact model training and performance.

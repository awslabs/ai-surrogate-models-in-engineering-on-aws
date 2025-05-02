.. _tutorial-slices-windsor-prediction:

Using Slice Prediction on new Geometry 
===================================================================

This tutorial demonstrates how to use a trained model from the Slice prediction pipeline on the :ref:`datasets-windsor` on new geometry.

If you haven't yet completed the :ref:`training tutorial <tutorial-slices-windsor-training>`, do so first to output a model ready for predictions.

Creating the Manifest
----------------------

The ``run-create-manifest-prediction`` script generates the required manifest for the WindsorML dataset. To create the manifest, run the script pointing to your dataset location:

.. code-block:: shell

   ./run-create-manifest-prediction /path/to/windsor/dataset

This will generate one manifest

- ``predict.manifest``: Lists geometry files for predicting the simulation variable

You can customize the manifest to use different geometries by editing ``run-create-manifest-prediction`` script:

.. code-block:: shell

   #!/bin/bash

   # Get a list of run folders for prediction (5 random runs) only
   predict_run_folders=($(ls -d "$dataset_prefix/run_"* | shuf -n 5))

   # Create predict.manifest with geometry-only, no image data 
   mlsimkit-manifest create -m "prediction.manifest" -f "name=geometry_files,file_glob=*.stl" "${predict_run_folders[@]}"

A manifest is a JSON Lines (``.manifest``) file that lists the paths to the data files and their associated slice image files if applicable. Each line in the manifest represents a single data file entry, containing the following key:

- ``"geometry_files"``: A relative or absolute path to the geometry file (e.g., ``.stl``)

Here's an example manifest entry:

.. code-block:: json

   {
     "geometry_files": [
       "file:///mnt/caemldatasets/windsor/dataset/run_99/windsor_99.stl"
     ]
   }

This entry lists the path to a single geometry file (``windsor_99.stl``).  Note that ``"slices_uri"`` will not exist in this manifest which is intended as we are demostrating the case where we do not have ground truth.


Running the Pipeline
--------------------

With the manifests created and configuration files in place, you can run the full Slice prediction pipeline using the provided scripts:

Run prediction:

.. code-block:: shell

   ./run-prediction

This script executes the necessary command using the ``prediction.yaml`` configuration file.


Reviewing Results
------------------

During Prediction
~~~~~~~~~~~~~~~~~

The prediction step generates images showing the slice predictions made by the trained model for new, unseen geometries. These images are located in the ``outputs/prediction/images/`` directory.

- ``*-prediction-*.png``: The predicted slice images from the trained model

Since the prediction manifest contains only geometry files (no ground truth slice data), the prediction outputs do not include original or error images for comparison. However, you can visually inspect the predicted slices to ensure they match your expectations for the given geometry.

If you have access to the ground truth simulation data for the prediction geometries, you can modify the ``predict.yaml`` configuration to include a comparison against the ground truth. Set ``slices.predict.compare-groundtruth: True`` and provide the appropriate manifest with both geometry and slice data.


Configuration Files
-------------------

The Slice prediction pipeline is configured using separate YAML files for training and prediction.  Below we show the prediction YAML file :

**prediction.yaml**

This file configures the prediction step, using the trained models from the training pipeline. Key settings include:

- ``output-dir``: Directory for storing prediction outputs (images, metrics)
- ``slices.preprocess.manifest-uri``: Path to the prediction data manifest
- ``slices.predict.ae-model-path``: Path to the trained image autoencoder model
- ``slices.predict.mgn-model-path``: Path to the trained prediction model

To get an introduction to the available configuration options, use the ``mlsimkit-learn slices --help`` command and the ``--help`` option for each sub-command such as ``mlsimkit-learn slices predict --help``. This will provide an overview of the options and their purposes, which can be helpful when configuring the training and prediction pipelines. 

Next Steps
----------

Dive into the :ref:`user-guide-slices` for detailed information on all configuration options and how they impact model training and performance. 

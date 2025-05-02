.. _user-guide-slices:

Model User Guide -- Slice Prediction
====================================

Introduction
------------

MLSimKit uses a deep learning architecture inspired by MeshGraphNets coupled with an AutoEncoder to predict slices of unseen geometries. This user guide walks through how to build a model to predict slices of the :ref:`datasets-windsor` geometries using MLSimKit.

Key concepts:
 
 - Slice: a 2D cross-sectional plane cut through the 3D geometry and volume that captures parametrics stored as an image file.  In computational fluid dynamics (CFD), common parametrics include velocity and preasure.  
 - Manifest: a JSON Lines file that links geometries to their slices, used in the preprocessing step.
 - Config: a YAML file that can be used to specify preprocessing, training, and inference settings.


How to Build a Slice Model
--------------------------

Getting the Data
~~~~~~~~~~~~~~~~

This user guide utilizes the WindsorML Body dataset. For detailed instructions on accessing and downloading the example dataset refer to :ref:`datasets-windsor`.  For this user guide we assume the data is downloaded at the relative path of ``data/windsor/``.  

Creating a Manifest File and a Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _slice-manifest:

Create a Manifest File
^^^^^^^^^^^^^^^^^^^^^^

To preprocess your data for training and inference steps, you'll need to create a manifest file. This manifest file should list the paths to the data files and the associated slice images files. The manifest file should be a JSON Lines (.jsonl or .manifest) file, with each line representing a single data file entry. Each entry contains the following keys:

    - ``"geometry_files"``: A relative or absolute path to the file(s) associated with a geometry (.stl or .vtp)
    - ``"slices_uri"``: A list of relative or absolute path(s) to slice(s) stored as image(s) associated with the geometry (optional for inference manifest)

Here is an example of two rows from a manifest file:

.. code-block:: json

    {"geometry_files": ["data/windsor/dataset/run_0/windsor_0.stl"], "slices_uri": ["data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0004.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0009.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0001.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0005.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0002.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0003.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0006.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0008.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0007.png", "data/windsor/dataset/run_0/images/velocityxavg/view1_constz_scan_0000.png"]}
    {"geometry_files": ["data/windsor/dataset/run_1/windsor_1.stl"], "slices_uri": ["data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0004.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0009.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0001.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0005.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0002.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0003.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0006.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0008.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0007.png", "data/windsor/dataset/run_1/images/velocityxavg/view1_constz_scan_0000.png"]}

We provide a utility for easier creation of manifest files using the datasets referenced in our guide.  

.. code-block:: shell

    mlsimkit-manifest create ...

Please refer to the :ref:`Training Slice Prediction on the WindsorML Dataset - Manifest Creation<tutorial-slices-windsor-training-manifest-creation>` for more details and an example shell script that demostrates how to use this command.

.. _slice-config:

Create a Config File
^^^^^^^^^^^^^^^^^^^^

To run preprocessing, training, and inference, you can provide the configurations either in an input config file or via the command-line interface (CLI). Here is an example config file:

.. code-block:: shell

    log:
        prefix-dir: <log_output_directory>             # all logs go here

    output_dir: <output_directory>
    
    slices:
        # step 1  
        preprocess:
            manifest-uri: <manifest_file_path>

        # step 2  
        train-image-encoder:
            epochs: 500
            batch-size: 4
            train-manifest-path: <output_directory>/train.manifest          # automatically detected within pipeline
            validate.manifest-path: <output_directory>/validate.manifest   # automatically detected within pipeline
      
        # (Optional) debug/inspect step
        inspect-image-encoder:
            manifest-path: <output_directory>/<manifest_filename_without_extension>-valid.manifest              # automatically detected within pipeline
            model-path: <output_directory>/ae/training_output/best_model.pt                                     # automatically detected within pipeline

        # step 3
        process-mesh-data:
            model-path: <output_directory>/ae/training_output/best_model.pt                                     # automatically detected within pipeline
            manifest-paths:
                - <output_directory>/train.manifest                       # automatically detected within pipeline
                - <output_directory>/validate.manifest                     # automatically detected within pipeline

        # step 4
        train-prediction:
            train-manifest-path: <output_directory>/train.manifest        # automatically detected within pipeline
            validate.manifest-path: <output_directory>/validate.manifest # automatically detected within pipeline
            hidden-size: 128
            message-passing-steps: 10
            batch-size: 1
            epochs: 200 

        # step 5
        predict:
            manifest-path: <output_directory>/<manifest_filename_without_extension>-test.manifest               # automatically detected within pipeline
            ae-model-path: <output_directory>/ae/training_output/best_model.pt                                  # automatically detected within pipeline
            mgn-model-path: <output_directory>/mgn/training_output/best_model.pt                                # automatically detected within pipeline
            compare-groundtruth: true


Replace ``<output_directory>`` with the directory where you want to save the outputs, and  ``<manifest_uri>`` with the path to your manifest file in the format described in the :ref:`previous step <slice-manifest>`. You can modify other configuration parameters discussed in the following sections by adding or updating their values in the configuration file. This allows you to override the default settings as per your requirements.  
Additionally our training pipeline will create a `.project` file in the ``<output_directory>``.  This file tracks configurations between pipeline steps, allowing the user to avoid needing to specify certain field.  In the above config example, we added the following comment to designate a number of these fields ``# automatically detected within pipeline``.  This allows the user flexibilty if they need or want to override configurations in the ``.project`` file.

.. _slice-preprocess:

Preprocessing
~~~~~~~~~~~~~

Slice prediction model predicts slices directly from a 3D geometry mesh. The first step is to preprocess the slice image files, converting them into data objects that can be easily consumed by the PyTorch deep learning framework. MLSimKit supports preprocessing of most common image file formats such as `.jpeg` and `.png`.

Run the following command to preprocess the data, replacing ``<config_file_path>`` with the path to the config file created in the :ref:`previous step <slice-config>`:

.. code-block:: shell

    mlsimkit-learn --config <config_file_path> slices preprocess

The preprocessing takes around 2 minutes on an AWS g5.2xlarge instance on 3290 RGB image slices. During preprocessing, logs are printed in the console.

The preprocessing step makes a "working" copy of the input manifest file to ``<output_directory>/<manifest_filename>-copy`` containing three additional fields: ```id```, ``slices_data_uri``, and ``slices_data_frame_count``.  The ```id``` field is a generated field that uniquely identifies the row of data.  The ``slices_data_uri`` is a path pointing to the processed data generated from preocessing step.  The ``slices_data_frame_count`` is the number of image slices contained in the group of data.  This should match the number of files that are listed in ``slices_uri``.  The output resolution of the preprocessed data is determined by the ``resolution`` field in the config or CLI arguments, which is `[128, 128]` ([horizontal resolution, vertical resolution]).  The resolution is restricted to whole number aspect ratios (larger resolution / smaller resolution).  The resolution directly impacts the model size, which may cause out of memory issues if increased too much.  You can modify this value in the config file or via command line arguments. To modify it in the config file, simply add a ``preprocess`` section along with the ``resolution`` and its desired value:

.. code-block:: shell

      slices:
        preprocess:
            resolution: [192, 96]
            train-size: 0.7
            valid-size: 0.1
            test-size: 0.2            


The preprocessing step writes three manifest files to ``<output_directory>/``: ``train.manifest``, ``validate.manifest``, and ``test.manifest``.  These three files define the train, validation, and test datasets.  Each contains a portion of the rows from the working manifest.  The portion of the data for each file can be adjusted by defining ``train-size``, ``valid-size``, and ``test-size`` in the config file under ``preprocess``.  These fields should be defined between 0 and 1 and when summed equal 1.  They also have default values of 0.6 for ``train-size``, 0.2 for ``valid-size`` and 0.2 for ``test-size`` and therefore only need to be defined if you want a different split in the data.   

After preprocessing the data, you can proceed to training the image encoder portion of the model.


.. _slice-train-image-encoder:

Training The Image Encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training the image encoder step is where the machine learning model learns to compress the image slices and reconstruct that compression. It takes the preprocessed data as input and produces PyTorch model files as output. The model files can then be used as part of the full slice prediciton model to make predictions in the :ref:`final step - Full Slice Prediction<slice-slice_predict>`.

There are a number of hyperparameters associated with model training of the AE, and all of them have default values. If you choose to use values other than the default ones, you can specify them in the :ref:`config file <slice-config>` or via command line arguments. To do it in the config file, simply add a ``train-image-encoder`` section along with the hyper-parameter names and values. Here is an example.

.. code-block:: shell

    slices:

        train-image-encoder:
            train-manifest-path: <output_directory>/train.manifest
            validate.manifest-path: <output_directory>/validate.manifest
            batch-size: 4
            epochs: 1000
            opt:
                learning-rate: 0.0003                


In this example, the values of 2 hyperparameters are adjusted. ``batch-size`` is set to ``4`` which controls the number of data samples ran through the model before model weights are updated.  A ``batch-size`` can impact model performance and the speed of training.  Setting this value too low or high can degrade model performance and is dependent on the specifics of the use case and dataset.  We recommend a value of at least 4.
``epochs`` determines the number of times the dataset is passed through the neural network during training. The larger the number of epochs, the longer the model training time. A value that is too small though may lead to models that have not fully learned.
Lastly, ``learning-rate`` stands for learning rate, which controls how fast the model learns. With a larger learning rate,  the number of epochs can typically be smaller, as the neural network makes bigger updates with every data point. A learning rate that is too large, however, can lead to poor performing models.
Note that ``learning-rate`` is one of the optimizer settings. Thus, it should be added under ``opt``.
To see the full list of training hyperparameters and their definitions, run the command ``mlsimkit-learn slices train-image-encoder --help``.

Once the config file is ready, run the following command to start training.

.. code-block:: shell

    mlsimkit-learn --config <config_file_path> slices train-image-encoder

The training takes around 40 minutes on an AWS g5.2xlarge instance with a dataset of 3290 RGB image slices (10 slices per geometry) at a resolution of 128 x 128 for 1000 epochs. During training, the training loss and validation loss of each epoch are printed in the console.

The training step produces a number of output files (see an example list at the end of this section) in the folder ``<output_directory>/ae/training_output/``. Among them, there are model checkpoints including ``best_model.pt`` which by default will be used in the inference step to make predictions on unseen data. It is the model that has the lowest validation error.

The model training loss plots (original scale: ``model_loss.png``; log scale: ``model_loss_log.png``) are typically useful to look at. Training losses and validation losses should be gradually decreasing until no longer decreasing. The gap between training losses and validation losses shouldn’t be too big. If it’s not the case, the model is likely not going to perform well, and hyperparameter values and/or training data may need to be adjusted.    The following is an example log loss plot and note that we could have kept on training by increasing the number of epochs as the loss was still declining.

.. image:: images/example-slice-ae-loss-log.png
   :width: 400
   :height: 400
   :alt: Figure 1. An example log loss plot

The list of training output files:

.. code-block:: shell

   <output_directory>/
    ├── ae
        ├── ...
        └── training_output
            ├── best_model.pt
            ├── checkpoint_models
            │   ├── model_epoch0.pt
            │   ├── model_epoch10.pt
            │   ...
            ├── last_model.pt
            ├── model_loss.csv
            ├── model_loss.png
            └── model_loss_log.png


.. _slice-reconstruct-images:

Reconstruct Images
~~~~~~~~~~~~~~~~~~

Once AE model training is complete, you can optionally run inference on the AE model to verify that the model can adequately encode and decode (or reconstruct) the image slices. The inference step takes the preprocessed data produced by the :ref:`preprocessing step <slice-preprocess>` as one of the inputs.
The toolkit will use the manifest specified in ``manifest-path`` field and the model specified in the ``model-path`` field.  The ``model-path`` is typically either ``<output_dir>/ae/training_output/best_model.pt`` or  ``<output_dir>/ae/training_output/last_model.pt``.  This step performs the reconstruction on the image slice data and saves the inference output in the folder ``<output_dir>/ae/inference_output/``.
You can specify the fields in a section named ``reconstruct-images`` to the :ref:`config file <slice-config>` or via command line arguments. 

Here is an example of how to specify these fields in the config file.

.. code-block:: shell

    slices:

        process-mesh-data:
            model-path: <output_directory>/ae/training_output/best_model.pt
            manifest-paths:
                - <output_directory>/train.manifest
                - <output_directory>/validate.manifest

Run reconstruct-images via the following command.

.. code-block:: shell

    mlsimkit-learn --config <config_file_path> slices reconstruct-images

The following is an example of the output files that can be expected:

.. code-block:: shell
    
    <output_directory>/
    ├── ae
       ├── inference_output
       │  └──images
       │     ├── slice-group-0-combined-0.png
       │     ├── slice-group-0-combined-1.png
       │     ├── ...
       │     ├── slice-group-0-error-0.png
       │     ├── slice-group-0-error-1.png
       │     ├── ...
       │     ├── slice-group-0-original-0.png
       │     ├── slice-group-0-original-1.png
       │     ├── ...
       │     ├── slice-group-0-reconstructed-0.png
       │     ├── slice-group-0-reconstructed-1.png
       │     ├── ...
       │  ├── results.jsonl
       │  ├── slice-group-0-reconstructed.pt
       │  ├── slice-group-1-reconstructed.pt
       │  ├── ...      


``slice-group-<id>-original-<frame #>.png`` is the ground truth image after preprocessing.  The image should match the source images the user pointed to in the :ref:`preprocessing step <slice-preprocess>` in the manifest file.

``slice-group-<id>-reconstructed-<frame #>.png`` is the reconstructed image after the above original file is passed through the AE that was trained in :ref:`Training AE step <slice-train-image-encoder>`.  

``slice-group-<id>-error-<frame #>.png`` is the root mean squared error (RMSE) for each pixel value between the above 2 files.

All of these files are enumerated with both an ``id`` and ``frame #``, where the ``id`` value corresponds to the ``id`` value generated during :ref:`preprocessing <slice-preprocess>` that uniquely identifies the individual rows in ``<output_directory>/<manifest_filename>`` and the ``frame #`` corresponds to the individual images in the same manifest.  The ``frame #`` corresponds to the order of the files listed in `slices_data_uri` starting from 0.

``results.jsonl`` contains the various metrics that quantify the comparision in the ``id`` group between the original and the reconstructed images.  The following is an example row in json pretty-print format.

.. code-block:: json

    {
        "metrics": {
            "mae": 0.007588425055046,
            "mape": 0.052369099693203,
            "mse": 0.00004475500306656,
            "msle": 0.000563740206406696,
            "psnr": 27.6023534626506
        },
        "slice_data_uri": "file:///<output_directory>/slices/slice-group-<id>.npy"
    }

Description of each of the metrics under the `metrics` field can be reviewed at `torchmetrics <https://lightning.ai/docs/torchmetrics/stable/all-metrics.html>`_, where the abreviations correspond to the following metric names:

mse - Mean Squared Error
mae - Mean Absolute Error
mape - Mean Absolute Percentage Error
msle - Mean Squared Log Error
psnr - Peak Signal to Noise Ratio

``slice-group-<id>-reconstructed.pt`` contains the raw predictions from the reconstructions for each ``id`` group.  These are stored as Pytorch Tensors.

.. _slice-encode-mesh-data:

Encode Mesh Data
~~~~~~~~~~~~~~~~

Once the training for the image encoder is adequate, the encoding of the mesh data and image data together can be done. This step processes the geometry files defined in the manifest in the fields ``geometry_files`` and uses the image encoding model to take the preprocessed data from :ref:`preprocessing step <slice-preprocess>` to produce training data for the next step :ref:`Training The Full Slice Prediction Model <slice-train-prediction>`.
The toolkit will use the manifest files specified under ``manifest-paths`` field in the :ref:`config file <slice-config>` under `train-prediction`.  The model specified in the ``model-path`` field will be used to encode the slices.  The output will be saved in the folder ``<output_dir>/ae/inference_output/``.
These fields can also be defined via command line arguments.

Here is how to provide a encoding data path in the config file.

.. code-block:: shell

    slices:

        encode-mesh-data:
            model-path: <output_directory>/ae/training_output/best_model.pt
            manifest-paths:
                - <output_directory>/train.manifest
                - <output_directory>/validate.manifest

Run inference encoder via the following command.

.. code-block:: shell

    mlsimkit-learn --config <config_file_path> slices encode-mesh-data

The inference encoder step produces the following output files:

.. code-block:: shell
    
    <output_directory>/
    ├── ae
       ├── inference_output
       │  ├── geometry-group-0.pt
       │  ├── geometry-group-1.pt
       │  ├── ...
       │  ├── geometry-group-353.pt
       │  ├── geometry-group-354.pt      

``geometry-group-<id>.pt`` is the processed output data that contains the image slices encoding combined with the geometry mesh file in a format that can be used to train the final prediction model in :ref:`the next step training prediction model <slice-train-prediction>`. 

Also note that the manifests listed under the ``manifest-paths`` field in the :ref:`config file <slice-config>` are updated to include ``encoding_uri`` for each row that includes the absolute path to the files discussed above. 

.. _slice-train-prediction:

Training The Full Slice Prediction Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the :ref:`Encode Mesh Data <slice-encode-mesh-data>` step is complete, the training of the full slice prediction model can be done. Training the full slice prediction step is where the machine learning model learns to predict image slices from mesh geometry. It takes the processed data from :ref:`Encode Mesh Data <slice-encode-mesh-data>` as input and produces PyTorch model files as output. The model files can then be used as part of the full slice prediciton model to make predictions in the :ref:`final step - Full Slice Prediction<slice-slice_predict>`.
There are a number of hyperparameters associated with training of the full slice prediciton model, and all of them have default values. If you choose to use values other than the default ones, you can specify them in the :ref:`config file <slice-config>` or via command line arguments. To do it in the :ref:`config file <slice-config>`, simply add a ``train-prediction`` section along with the hyperparameter names and values. Here is an example.


.. code-block:: shell

    slices:

        train-prediction:
            train_manifest_path: <output_directory>/train.manifest
            validate.manifest_path: <output_directory>/validate.manifest          

To see the full list of training hyperparameters and their definitions, run the command ``mlsimkit-learn slices train-prediction --help``.

Once the config file is ready, run the following command to start training.

.. code-block:: shell

    mlsimkit-learn --config <config_file_path> slices train-prediction

The training takes around 220 minutes on an AWS g5.2xlarge instance with a dataset of 329 ~5000 node meshes. During training, the training loss and validation loss of each epoch are printed in the console.

The training step produces a number of output files (see an example list at the end of this section) in the folder ``<output_directory>/mgn/training_output/``. Among them, there are model checkpoints including ``best_model.pt`` which is recommended to use for new predictions. The ``best_model.pt`` is the model that has the lowest validation error.

The model training loss plots (original scale: ``model_loss.png``; log scale: ``model_loss_log.png``) are typically useful to look at. Training losses and validation losses should be gradually decreasing until no longer decreasing. The gap between training losses and validation losses shouldn’t be too big. If it’s not the case, the model is likely not going to perform well, and hyperparameter values and/or training data may need to be adjusted.  The following is an example log loss plot.

.. image:: images/example-slice-mgn-loss-log.png
   :width: 400
   :height: 400
   :alt: Figure 2. An example log loss plot

The list of training output files:

.. code-block:: shell

    <output_directory>/
    ├── ae
       └── training_output
           ├── best_model.pt
           ├── checkpoint_models
           │   └── model_epoch0.pt
           │   ├── model_epoch10.pt
           │   ├── ...
           ├── last_model.pt
           ├── model_loss.csv
           ├── model_loss.png
           └── model_loss_log.png

.. _slice-slice_predict:

Full Slice Prediction 
~~~~~~~~~~~~~~~~~~~~~

After :ref:`training the full slice prediction model <slice-train-prediction>`, the model can be used to predict slices on new geometry.  In the :ref:`config file <slice-config>` or via command line arguments, a manifest and the models from :ref:`the previous step <slice-train-prediction>` and :ref:`training the image encoder step <slice-train-image-encoder>` need to be provided.  The following is example of how to define these in the :ref:`config file <slice-config>`.  

.. code-block:: shell

    slices:

        predict:
            manifest-path: <output_directory>/<manifest_filename_without_extension>-test.manifest
            ae-model-path: <output_directory>/ae/training_output/best_model.pt
            mgn-model-path: <output_directory>/mgn/training_output/best_model.pt


Once the config file is ready, run the following command to genrate slice predictions.

.. code-block:: shell

    mlsimkit-learn --config <config_file_path> slices predict


If ground truth is provided in the manifest file, the full slice prediction produces a similiar output as the :ref:`Reconstruct Images step <slice-reconstruct-images>` but under ``<output_dir>/prediction/`` folder using the manifest defined under ``manifest_path`` field.  Note that the images files are named with ``<geometry_files_without_extension>`` instead of ``slice-group-<id>``.  Additionally there are ``<geometry_files_without_extension>.npy`` files that are the predictions saved as numpy arrays.  The following are examples of what to expect:

.. code-block:: shell

    <output_directory>/
    ├── prediction/
        ├── geometry-group-102-prediction.npy
        ├── ...
        ├── geometry-group-91-prediction.npy
        ├── images
        │   ├── geometry-group-102-error-0.png
        │   ...
        │   ├── geometry-group-102-error-9.png
        │   ├── geometry-group-102-original-0.png
        │   ...
        │   ├── geometry-group-102-original-0.png
        │   ├── geometry-group-102-original-9.png
        │   ...
        │   ├── geometry-group-102-prediction-9.png
        │   ...
        │   ├── geometry-group-91-error-0.png
        │   ...
        │   ├── geometry-group-91-error-9.png
        │   ├── geometry-group-91-original-0.png
        │   ...
        │   ├── geometry-group-91-original-9.png
        │   ...
        │   ├── geometry-group-91-prediction-9.png
        ├── results.jsonl 

If the ground truth is not provided ``results.jsonl``, and files with suffixes ``error-<frame #>.png`` and ``original-<frame #>.png`` will not be present.

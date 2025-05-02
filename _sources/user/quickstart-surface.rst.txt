.. _quickstart-surface:

Quickstart with Surface Variable Prediction
===========================================

To get started, first make sure that:

* MLSimKit is :ref:`installed <install>`

We'll use a sample dataset to quickly show you an end-to-end workflow. 

Surface variable prediction is accessed via the ``mlsimkit-learn surface`` command:

.. code-block:: shell

    Usage: mlsimkit-learn surface [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                                  [ARGS]...]...

      Use Case: Surface variable prediction via MeshGraphNets (MGN)

    Options:
      --manifest-uri PATH  Manifest file to drive all subcommands. Searches
                           mlsimkit/datasets as a fallback
      --help               Show this message and exit.

    Commands:
      predict
      preprocess
      train

To run surface variable prediction commands, you may provide a config only or override the config with command-line arguments by running:

.. code-block:: shell

    mlsimkit-learn --config <config> surface <command-name>

You can use the ``help`` command to see the definitions of all the hyperparameters associated with any command. For example, the following command will display all hyperparameters for training:

.. code-block:: shell

    mlsimkit-learn surface train --help


Sample Dataset 
------------------------
There is a sample config and a very small sample dataset called "ahmed-sample" so you can run end-to-end quickly:

.. code-block:: shell

    src/mlsimkit/conf
    └── surface
        └── sample.yaml

    src/mlsimkit/datasets
    ├── ...
    └── ahmed-sample
        ├── mapped_vtps
        └── surface.manifest

External Datasets
------------------------
In addition to the sample dataset, there are tutorials to get started with publicly available datasets::

    tutorials/
    └── surface
        ├── ahmed/
        ├── drivaer/
        ├── sample/
        └── windsor/

Run the Sample 
------------------------

First, make a folder for all the outputs. Replace ``--output-dir quickstart/surface`` in all the commands below with your own folder location.

Second, run the entire train-predict pipeline to make predictions on the sample data:

.. code-block:: shell

    mlsimkit-learn --output-dir quickstart/surface \
        --config src/mlsimkit/conf/surface/sample.yaml \
        surface preprocess train --device cpu predict

Also, note that commands can be chained together.  For example, the above runs `preprocess`, `train`, and then `predict`.

Running on GPU
~~~~~~~~~~~~~~

MLSimKit automatically uses a GPU by default. To use your GPU, remove ``--device cpu`` from the previous command and run again:

.. code-block:: shell

    mlsimkit-learn --output-dir quickstart/surface \
        --config src/mlsimkit/conf/surface/sample.yaml \
        surface preprocess train predict

All artifacts are written into the output directory ``--output-dir``. You may also set the output directory in the config file. Commands automatically share paths to the output artifacts such as the train model path. The sample configuration below sets some input options but most options use defaults. There are many options, which we go into detail after the quickstart.

The sample configuration ``conf/surface/sample.yaml`` looks like this:

.. code-block:: yaml

   surface:
      manifest_uri: ahmed-sample/surface.manifest

      preprocess:
        surface_variables:
        - name: pMean
        manifest-base-relative-path: ManifestRoot

      train:
        epochs: 10
        batch_size: 1

.. note::
   A manifest (``manifest_uri``) describes the paths to a dataset and is used to share data between tasks. For now, know that ``surface.manifest``
   references a small dataset packaged with MLSimKit.

You will see console logs for all three commands, something like below. File artifacts are written to the ``--output-dir``. 

.. code-block:: shell

    [INFO] [MLSimKit] Learning Tools
    [INFO] Package Version: 0.2.1.dev44+gff018da
    [INFO] Use Case: Surface variable prediction via MeshGraphNets (MGN)
    [INFO] Running command 'preprocess'
    [INFO] Preprocessing configuration: manifest_path='/home/ubuntu/mlsimkit/src/mlsimkit/datasets/ahmed-sample/surface.manifest' output_dir='/home/ubuntu/mlsimkit/quickstart/surface/preprocessed_data' downsample_remaining_perc=None num_processes=None save_cell_data=True map_data_to_stl=False mapping_interpolation_method=<InterpolationMethod.POINTS: 'points'> mapping_interpolation_radius=None mapping_interpolation_n_points=3 save_mapped_files=False normalize_node_positions=True manifest_base_relative_path=<RelativePathBase.ManifestRoot: 'ManifestRoot'>
    [INFO] Using 'data_files' for preprocessing
    [INFO] Preprocessing mesh files (num_processes=95)
    [INFO] Selected surface variables: [SurfaceVariables(name='pMean', dimensions=[])]
    ...
    [INFO] Saved output files in /home/ubuntu/mlsimkit/quickstart/surface/preprocessed_data
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/surface/surface-copy.manifest' written (7 records)
    [INFO] Total preprocessing time: 2.452 seconds
    [INFO] Splitting manifest into train-size=0.6 valid-size=0.2 test-size=0.2
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/surface/train.manifest' written (4 records)
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/surface/validate.manifest' written (1 records)
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/surface/test.manifest' written (2 records)
    [INFO] Running command 'train'
    ...
    [INFO] Training started for 'model'
    [INFO] Train dataset size: 4
    [INFO] Validation dataset size: 1
    [INFO] Training:   0%|                                                         | 0/10 [00:00<?, ?epochs/s ]
    [INFO] Epoch 0: train loss = 1.58301 (1.58301, 0.00000, 0.00000, 0.00000); validation loss = 0.93436 (0.93436, 0.00000, 0.00000, 0.00000); best validation loss = inf
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/surface/training_output/best_model.pt'
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/surface/training_output/checkpoint_models/model_epoch0.pt'
    [INFO] Training:  10%|████▉                                            | 1/10 [00:00<00:08,  1.02epochs/s ]
    [INFO] Epoch 1: train loss = 0.92012 (0.92012, 0.00000, 0.00000, 0.00000); validation loss = 0.83899 (0.83899, 0.00000, 0.00000, 0.00000); best validation loss = 0.93436
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/surface/training_output/best_model.pt'
    [INFO] Training:  20%|█████████▊                                       | 2/10 [00:01<00:04,  1.76epochs/s ]
    [INFO] Epoch 2: train loss = 0.87975 (0.87975, 0.00000, 0.00000, 0.00000); validation loss = 0.80079 (0.80079, 0.00000, 0.00000, 0.00000); best validation loss = 0.83899
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/surface/training_output/best_model.pt'
    [INFO] Training:  30%|██████████████▋                                  | 3/10 [00:01<00:03,  2.31epochs/s ]
    [INFO] Epoch 3: train loss = 0.73869 (0.73869, 0.00000, 0.00000, 0.00000); validation loss = 0.62471 (0.62471, 0.00000, 0.00000, 0.00000); best validation loss = 0.80079
    ...
    [INFO] Training:  90%|████████████████████████████████████████████     | 9/10 [00:02<00:00,  3.98epochs/s ]
    [INFO] Epoch 9: train loss = 0.17296 (0.17296, 0.00000, 0.00000, 0.00000); validation loss = 0.18069 (0.18069, 0.00000, 0.00000, 0.00000); best validation loss = 0.15008
    [INFO] Epoch 8 had the minimum validation loss.
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/surface/training_output/last_model.pt'
    [INFO] Training: 100%|████████████████████████████████████████████████| 10/10 [00:02<00:00,  3.74epochs/s ]
    [INFO] Training time for model: 2.689 seconds / 0.045 minutes
    [INFO] Minimum validation loss: 0.15008
    [INFO] Minimum train loss: 0.17296
    [INFO] Get predictions on the training set
    [INFO] Run inference on geometry 1
    [INFO] Run inference on geometry 2
    [INFO] Run inference on geometry 3
    [INFO] Run inference on geometry 4
    [INFO] Prediction error for surface variable 'pMean': RMSE (root mean squared error) = 0.0742051303, MAE (mean absolute error) = 0.04554, WMAPE (weighted mean absolute percentage error) = 0.297, MAE normalized by 1%-99% ground truth range = 0.039, Average largest 1% absolute deviation normalized by ground truth range = 0.249, 
    [INFO] Get predictions on the validation set
    [INFO] Run inference on geometry 1
    [INFO] Prediction error for surface variable 'pMean': RMSE (root mean squared error) = 0.0691222399, MAE (mean absolute error) = 0.04100, WMAPE (weighted mean absolute percentage error) = 0.250, MAE normalized by 1%-99% ground truth range = 0.034, Average largest 1% absolute deviation normalized by ground truth range = 0.231, 
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/surface/train.manifest' written (4 records)
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/surface/validate.manifest' written (1 records)
    [INFO] Training Completed
    [INFO] Total training time: 3.376 seconds / 0.056 minutes
    [INFO] Running command 'predict'
    [INFO] Inference configuration: model_path='/home/ubuntu/mlsimkit/quickstart/surface/training_output/best_model.pt' manifest_path='/home/ubuntu/mlsimkit/quickstart/surface/test.manifest' inference_results_dir='/home/ubuntu/mlsimkit/quickstart/surface/predictions' device=<Device.CUDA: 'cuda'> save_vtp_output=<VtpOutput.BOTH: 'prediction_and_difference'> save_prediction_screenshots=False screenshot_size=[2000, 800]
    [INFO] Inference dataset size: 2
    [INFO] Time to load dataset and model: 0.061 seconds
    [INFO] Run inference on geometry 1
    [INFO] Run inference on geometry 2
    [INFO] Prediction error for surface variable 'pMean': RMSE (root mean squared error) = 0.0844738632, MAE (mean absolute error) = 0.05038, WMAPE (weighted mean absolute percentage error) = 0.320, MAE normalized by 1%-99% ground truth range = 0.048, Average largest 1% absolute deviation normalized by ground truth range = 0.309, 
    [INFO] Inference time for each data point: 0.037 seconds
    [INFO] Total inference time: 0.135 seconds


When complete, the output directory ``quickstart/surface/`` should now contain the data from preprocessing and training and the results for prediction. Something like this:

.. code-block:: shell

   quickstart/surface/
    ├── predictions
    │   ├── error_metrics.csv
    │   ├── pMean_errors_by_geometry.csv
    │   └── results
    │       ├── predicted_boundary_1_mapped.vtp
    │       └── predicted_boundary_4_mapped.vtp
    ├── preprocessed_data
    │   ├── preprocessed_run_00000.pt
    │   ├── preprocessed_run_00001.pt
    │   ├── preprocessed_run_00002.pt
    │   ├── preprocessed_run_00003.pt
    │   ├── preprocessed_run_00004.pt
    │   ├── preprocessed_run_00005.pt
    │   └── preprocessed_run_00006.pt
    ├── surface-copy.manifest
    ├── test.manifest
    ├── train.manifest
    ├── training_output
    │   ├── best_model.pt
    │   ├── best_model_predictions
    │   │   ├── training
    │   │   │   ├── error_metrics.csv
    │   │   │   ├── pMean_errors_by_geometry.csv
    │   │   │   └── results
    │   │   │       ├── predicted_boundary_2_mapped.vtp
    │   │   │       ├── predicted_boundary_3_mapped.vtp
    │   │   │       ├── predicted_boundary_6_mapped.vtp
    │   │   │       └── predicted_boundary_7_mapped.vtp
    │   │   └── validation
    │   │       ├── error_metrics.csv
    │   │       ├── pMean_errors_by_geometry.csv
    │   │       └── results
    │   │           └── predicted_boundary_5_mapped.vtp
    │   ├── checkpoint_models
    │   │   └── model_epoch0.pt
    │   ├── last_model.pt
    │   ├── model_loss.csv
    │   ├── model_loss.png
    │   └── model_loss_log.png
    └── validate.manifest


Results
------------------------
You have trained a surface variable prediction model from scratch on sample data.  You can compare the training and validation predictions using the output VTP files in ``training_output/best_model_predictions/`` against the original VTP files. The manifest files link inputs to outputs. For example, the ``train.manifest`` will look something like this: 

.. code-block:: shell

   $ cat quickstart/surface/train.manifest | jq .
    {
      "data_files": [
        "/home/ubuntu/mlsimkit/src/mlsimkit/datasets/ahmed-sample/mapped_vtps/boundary_6_mapped.vtp"
      ],
      "id": 5,
      "preprocessed_files": "file:///home/ubuntu/mlsimkit/quickstart/surface/preprocessed_data/preprocessed_run_00005.pt",
      "predicted_file": "/home/ubuntu/mlsimkit/quickstart/surface/training_output/best_model_predictions/training/predictions/predicted_boundary_6_mapped.vtp"
    }

(``jq`` is a command-line tool to ease JSON formatting and parsing.)

You can compare the ``predicted_file`` to the original ``data_files`` by loading the linked files into your preferred software like ParaView. We also provide a convenient viewer tool to automate the file loading for datasets.

Similar to training, results for prediction only via the ``mlsimkit-learn surface predict`` command are in ``predictions/``. 

.. note::

    The prediction results are poor because we are using a very small dataset and much reduced training time (few epochs) to show you the end-to-end workflow. The :ref:`tutorial-surface-ahmed` demonstrates more accurate training on a real-world sized dataset. 



Result Visualization
---------------------

The simple 3D interactive viewer GUI and automated screenshot tooling provides a quick method to review surface prediction results. The viewer compares the original (ground truth) input mesh vs. the predicted mesh across an entire dataset. The viewer uses the same config and output directories, including manifest files, to locate the input and output files.  

The viewer is not intended as a substitute feature-rich applications like ParaView. 

Start the viewer on a machine with a display by running:

.. code-block:: shell

    mlsimkit-learn --output-dir quickstart/surface \
        --config src/mlsimkit/conf/surface/sample.yaml \
        surface view

.. only:: html

    .. image:: ../images/quickstart-surface-viewer.gif
       :align: center

.. only:: latex or pdf

    .. image:: ../images/quickstart-surface-viewer.png
       :align: center

.. warning:: 

   **The viewer needs a display**. On remote Linux/Ubuntu machines, follow the :ref:`troubleshooting guide<troubleshooting_xvfb>` to render screenshots without a GUI. You will see the following error otherwise::
 
        This system does not appear to be running an xserver.
        PyVista will likely segfault when rendering.

        Try starting a virtual frame buffer with xvfb, or using
          ``pyvista.start_xvfb()``

          warnings.warn(
        Segmentation fault (core dumped)

    

By default, the prediction results for the training dataset are displayed. Use the <right> and <left> arrow keys to scroll through the meshes and <tab> to switch between surface variables, when available. The quickstart sample data has just one surface variable ("pMean") in the dataset. 

**Screenshots:** To automatically output images without a GUI, pass ``--no-gui``. Screenshot images still require access to a display: 

.. code-block:: shell

    mlsimkit-learn --output-dir quickstart/surface \
        --config src/mlsimkit/conf/surface/sample.yaml \
        surface view --no-gui --variable "pMean"

By default, screenshots are written to ``<output-dir>/screenshots``. For example, for this quickstart, you will see console logs like this:

.. code-block:: shell

    [INFO] [MLSimKit] Learning Tools
    [INFO] Package Version: 0.1.1.dev74+gf3672bc.d20240607
    [INFO] Use Case: Surface variable prediction via MeshGraphNets (MGN)
    [INFO] Screenshot written: /home/ubuntu/mlsimkit/quickstart/surface/screenshots/run0000_boundary_1_mapped_pMean.png
    [INFO] Screenshot written: /home/ubuntu/mlsimkit/quickstart/surface/screenshots/run0006_boundary_7_mapped_pMean.png
    [INFO] Screenshot written: /home/ubuntu/mlsimkit/quickstart/surface/screenshots/run0001_boundary_2_mapped_pMean.png
    [INFO] Screenshot written: /home/ubuntu/mlsimkit/quickstart/surface/screenshots/run0004_boundary_5_mapped_pMean.png

**Screenshots on remote Linux servers:** See the :ref:`surface prediction user guide <user_guide_surface_visualizations>` for how to take screenshots on remote (headless) Linux servers using a virtual framebuffer. You may also configure what meshes and variables are used for the visualizations.

Next Steps
------------------------

Follow the :ref:`tutorial-surface-ahmed` to train on a real large-scale dataset and make accurate predictions.


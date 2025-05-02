.. _quickstart-kpi:

Quickstart with KPI Prediction
===================================

To get started, first make sure that:

* MLSimKit is :ref:`installed <install>`

We'll use a sample dataset to quickly show you an end-to-end workflow. 

KPI Prediction (KPI) is accessed via the ``mlsimkit-learn kpi`` command:

.. code-block:: shell

    Usage: mlsimkit-learn kpi [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                              [ARGS]...]...

      Use Case: KPI prediction using a variant of Mesh Graph Networks (MGN)

    Options:
      --manifest-uri PATH  Manifest file to drive all subcommands. Searches
                           mlsimkit/datasets as a fallback
      --help               Show this message and exit.

    Commands:
      predict 
      preprocess
      train


To run KPI commands, you may provide a config only or override the config with command-line arguments by running:

.. code-block:: shell

    mlsimkit-learn --config <config> kpi <command-name>

You can use the ``help`` command to see the definitions of all the hyperparameters associated with any command. For example, the following command will display all hyperparameters for training:

.. code-block:: shell

    mlsimkit-learn kpi train --help


Sample Dataset 
------------------------
There is a sample config and a very small sample dataset called "drivaer-sample" so you can run end-to-end quickly:

.. code-block:: shell

    src/mlsimkit/conf
    └── kpi
        └── sample.yaml

    src/mlsimkit/datasets
    ├── ...
    └── drivaer-sample
        ├── downsampled_stls
        └── kpi.manifest

External Datasets
------------------------
In addition to the sample dataset, there are tutorials to get started with publicly available datasets::

    tutorials/
    └── kpi
        ├── ahmed/
        ├── drivaer/ (coming soon)
        ├── sample/
        └── windsor/

Run the Sample
--------------

First, make a folder for all the outputs. Replace ``--output-dir quickstart/kpi`` in the command below with your own folder location.

Second, run the entire train-predict pipeline to make predictions on the sample data using CPU-only:

.. code-block:: shell

    mlsimkit-learn --output-dir quickstart/kpi \
        --config src/mlsimkit/conf/kpi/sample.yaml \
        kpi preprocess train --device cpu predict

Also, note that commands can be chained together.  For example, the above runs `preprocess`, `train`, and then `predict`.

Running on GPU
~~~~~~~~~~~~~~

MLSimKit automatically uses a GPU by default. To use your GPU, remove ``--device cpu`` from the previous command and run again:

.. code-block:: shell

    mlsimkit-learn --output-dir quickstart/kpi \
        --config src/mlsimkit/conf/kpi/sample.yaml \
        kpi preprocess train predict


.. note::
   
   On older MacOS hardware, you may see the error ``Cannot convert a MPS Tensor to float64 dtype``. If so, force CPU by specifying ``--device cpu`` for train commands. 
   
   In general, please see the :ref:`Troubleshooting <troubleshooting>` guide for possible errors if commands do not work.
   


All artifacts are written into the output directory ``--output-dir``. You may also set the output directory in the config file. Commands automatically share paths to the output artifacts such as the train model path. The sample configuration below sets some input options but most options use defaults. There are many options, which we go into detail after the quickstart.

The sample configuration ``conf/kpi/sample.yaml`` looks like this:

.. code-block:: yaml

   kpi:
      manifest_uri: drivaer-sample/kpi.manifest

      train:
        epochs: 10
        batch_size: 1

.. note::
   A manifest (``manifest_uri``) describes the paths to a dataset and is used to share data between tasks. For now, know that ``kpi.manifest``
   references a small dataset packaged with MLSimKit.

You will see console logs for all three commands, something like below. File artifacts are written to the ``--output-dir``. 

.. code-block:: shell

    [INFO] [MLSimKit] Learning Tools
    [INFO] Package Version: 0.2.3.dev3+gaf49957.d20240808
    [INFO] Use Case: KPI prediction via a variant of MeshGraphNets
    [INFO] Running command 'preprocess'
    [INFO] Preprocessing configuration: manifest_path='/home/ubuntu/mlsimkit/src/mlsimkit/datasets/drivaer-sample/kpi.manifest' output_dir='/home/ubuntu/mlsimkit/quickstart/kpi/preprocessed_data' downsample_remaining_perc=None num_processes=None manifest_base_relative_path=<RelativePathBase.ManifestRoot: 'ManifestRoot'>
    [INFO] Preprocessing mesh files (num_processes=95)
    [INFO] Pre-processing file 1 out of 7 files
    [INFO] Pre-processing file 2 out of 7 files
    [INFO] Pre-processing file 3 out of 7 files
    [INFO] Pre-processing file 4 out of 7 files
    [INFO] Pre-processing file 5 out of 7 files
    [INFO] Pre-processing file 6 out of 7 files
    [INFO] Pre-processing file 7 out of 7 files
    [INFO] Saved output files in /home/ubuntu/mlsimkit/quickstart/kpi/preprocessed_data
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/kpi/kpi-copy.manifest' written (7 records)
    [INFO] Total preprocessing time: 2.036 seconds
    [INFO] Splitting manifest into train-size=0.6 valid-size=0.2 test-size=0.2
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/kpi/train.manifest' written (4 records)
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/kpi/validate.manifest' written (1 records)
    [INFO] Manifest '/home/ubuntu/mlsimkit/quickstart/kpi/test.manifest' written (2 records)
    [INFO] Running command 'train'
    ...
    [INFO] Training state configuration: {"Distributed": "no", "Num processes": 1, "Process index": 0, "Local process index": 0, "Device": "cuda", "Mixed precision": "no"}
    [INFO] Training started for 'model'
    [INFO] Train dataset size: 4
    [INFO] Validation dataset size: 1
    [INFO] Training:   0%|                                                         | 0/10 [00:00<?, ?epochs/s ]
    [INFO] Epoch 0: train loss = 0.98888; validation loss = 1.22718; best validation loss = inf
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/kpi/training_output/best_model.pt'
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/kpi/training_output/checkpoint_models/model_epoch0.pt'
    [INFO] Training:  10%|████▉                                            | 1/10 [00:00<00:07,  1.27epochs/s ]
    [INFO] Epoch 1: train loss = 0.90357; validation loss = 1.04253; best validation loss = 1.22718
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/kpi/training_output/best_model.pt'
    [INFO] Training:  20%|█████████▊                                       | 2/10 [00:00<00:03,  2.32epochs/s ]
    [INFO] Epoch 2: train loss = 0.88518; validation loss = 1.04486; best validation loss = 1.04253
    [INFO] Epoch 1 had the minimum validation loss.
    [INFO] Training:  30%|██████████████▋                                  | 3/10 [00:00<00:02,  3.27epochs/s ]
    [INFO] Epoch 3: train loss = 0.87817; validation loss = 1.12413; best validation loss = 1.04253
    [INFO] Epoch 1 had the minimum validation loss.

    ...
    [INFO] Training:  90%|████████████████████████████████████████████     | 9/10 [00:01<00:00,  7.32epochs/s ]
    [INFO] Epoch 9: train loss = 0.85995; validation loss = 1.23066; best validation loss = 1.04253
    [INFO] Epoch 1 had the minimum validation loss.
    [INFO] Model saved to '/home/ubuntu/mlsimkit/quickstart/kpi/training_output/last_model.pt'
    [INFO] Training: 100%|████████████████████████████████████████████████| 10/10 [00:01<00:00,  6.27epochs/s ]
    [INFO] Training time for model: 1.606 seconds / 0.027 minutes
    [INFO] Minimum validation loss: 1.04253
    [INFO] Minimum train loss: 0.85995
    [INFO] best_model: prediction error for train data, kpi 0: MAPE = 0.07986, MAE = 0.02254, MSE = 0.00057
    [INFO] best_model: prediction error for validation data, kpi 0: MAPE = 0.10089, MAE = 0.02740, MSE = 0.00075
    [INFO] Saved error metrics: /home/ubuntu/mlsimkit/quickstart/kpi/training_output/best_model_predictions/dataset_prediction_error_metrics.csv
    [INFO] Saved results: /home/ubuntu/mlsimkit/quickstart/kpi/training_output/best_model_predictions/prediction_results.csv
    [INFO] last_model: prediction error for train data, kpi 0: MAPE = 0.08033, MAE = 0.02231, MSE = 0.00065
    [INFO] last_model: prediction error for validation data, kpi 0: MAPE = 0.11909, MAE = 0.03234, MSE = 0.00105
    [INFO] Saved error metrics: /home/ubuntu/mlsimkit/quickstart/kpi/training_output/last_model_predictions/dataset_prediction_error_metrics.csv
    [INFO] Saved results: /home/ubuntu/mlsimkit/quickstart/kpi/training_output/last_model_predictions/prediction_results.csv
    [INFO] Training Completed
    [INFO] Total training time: 2.281 seconds / 0.038 minutes
    [INFO] Running command 'predict'
    [INFO] Settings: inference_data_path=None model_path='/home/ubuntu/mlsimkit/quickstart/kpi/training_output/best_model.pt' manifest_path='/home/ubuntu/mlsimkit/quickstart/kpi/test.manifest' inference_results_dir='/home/ubuntu/mlsimkit/quickstart/kpi/predictions' num_processes=None output_kpi_indices=None
    [INFO] Inference configuration: inference_data_path=None model_path='/home/ubuntu/mlsimkit/quickstart/kpi/training_output/best_model.pt' manifest_path='/home/ubuntu/mlsimkit/quickstart/kpi/test.manifest' inference_results_dir='/home/ubuntu/mlsimkit/quickstart/kpi/predictions' num_processes=None output_kpi_indices=None
    [INFO] Predicting using input manifest '/home/ubuntu/mlsimkit/quickstart/kpi/test.manifest'
    [INFO] Inference dataset size: 2
    [INFO] Time to load dataset and model: 0.034 seconds
    [INFO] Inference time for each data point: 0.009 seconds
    [INFO] prediction error for inference data, kpi 0: MAPE = 0.07365, MAE = 0.02141, MSE = 0.00047
    [INFO] Saved error metrics: /home/ubuntu/mlsimkit/quickstart/kpi/predictions/dataset_prediction_error_metrics.csv
    [INFO] Saved results: /home/ubuntu/mlsimkit/quickstart/kpi/predictions/prediction_results.csv
    [INFO] Total inference time: 0.136 seconds

When complete, the output directory ``quickstart/kpi/`` should now contain the data from preprocessing and training and the results for prediction

.. code-block:: shell

   quickstart/
    └── kpi
        ├── kpi-copy.manifest
        ├── train.manifest
        ├── validate.manifest
        ├── test.manifest
        ├── logs
        │   └── kpi
        │       ├── inference.log
        │       ├── preprocessing.log
        │       └── training.log
        ├── predictions
        │   ├── dataset_prediction_error_metrics.csv
        │   ├── predicted_vs_actual_kpi0.png
        │   └── prediction_results.csv
        ├── preprocessed_data
        │   ├── preprocessed_run_00000.pt
        │   ├── preprocessed_run_00001.pt
        │   ├── preprocessed_run_00002.pt
        │   ├── preprocessed_run_00003.pt
        │   ├── preprocessed_run_00004.pt
        │   ├── preprocessed_run_00005.pt
        │   └── preprocessed_run_00006.pt
        └── training_output
            ├── best_model.pt
            ├── best_model_predictions
            │   ├── dataset_prediction_error_metrics.csv
            │   ├── predicted_vs_actual_kpi0.png
            │   └── prediction_results.csv
            ├── checkpoint_models
            │   └── model_epoch0.pt
            ├── last_model.pt
            ├── last_model_predictions
            │   ├── dataset_prediction_error_metrics.csv
            │   ├── predicted_vs_actual_kpi0.png
            │   └── prediction_results.csv
            ├── model_loss.csv
            ├── model_loss.png
            └── model_loss_log.png

Results
------------------------
You have trained a KPI model from scratch on sample data. The best model is saved to ``training_output/best_model.pt``. You can compare the train and validation predictions against the ground truth in ``training_output/best_model_predictions/prediction_results.csv`` and view the plot ``.png`` files in the same folder. Results for the ``mlsimkit-learn kpi predict`` command are in ``predictions/``.

Next Steps
------------------------

Follow the :ref:`tutorial-kpi-windsor` to train on a real large-scale dataset and make accurate predictions.


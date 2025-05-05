.. _tutorial-slices-windsor-data-access:

Downloading WindsorML data for Slice Prediction (48G)
=============================================================

The :ref:`datasets-windsor` is a publicly available dataset hosted in Hugging Face datasets.

Download Commands
-----------------

The tutorial includes a convenient command to download the training data (~48G):

.. code-block:: shell
    
   ./download-dataset /path/to/dataset 

Replace ``/path/to/dataset`` to your own directory and follow the prompts.

You may download the entire dataset or select a subset of the data. For example, below we choose one variable (``velocityxzvg``) and one view (``view1_constz_scan``) for runs 1, 1x, 1xx, which are also the specific variable and view used in the remainder of the slices tutorial. Additionally, this reduces the download size to approximately 0.83G.

Please note that you need at least 18 data samples to run the tutorials with the default configurations. You may adjust the train/validation/test sizes during the preprocessing step or modify the batch size during the training step to enable running the tutorials with fewer data samples.

.. code-block:: shell

    $ ./download-dataset ~/datasets/windsor/
    Download entire dataset (48G)? (y/n) n
    Select a variable:
    1) pressureavg
    2) rstress_xx
    3) rstress_yy
    4) rstress_zz
    5) velocityxavg
    #? 5
    Select a view:
    1) view1_constz_scan
    2) view2_constx_scan
    3) view3_consty_scan
    #? 1
    Choose runs by prefix e.g., 1*, 20* (default: *)
    1*

The download command wraps the Hugging Face ``snapshot_download`` function, which you may want to customize for specific downloads. For example, the following code downloads the training data for all runs:

.. code-block:: python

   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="neashton/windsorml",
       repo_type="dataset",
       local_dir="/path/to/dataset",
       allow_patterns=[
           "run_*/windsor_*.stl",
           "run_*/images/*"
       ]
   )

.. note::

   If you encounter a "Too Many Requests" error like this::

       huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests

   This is due to rate limiting. Wait and try again. Hugging Face uses a local cache, so already downloaded files won't need to be re-downloaded on retry.
   If the error continues, set the function argument ``max_workers=1`` to download one file at a time.


Next Step: Training
--------------------

Next, follow :ref:`tutorial-slices-windsor-training` to learn how to train a Slice prediction model on the WindsorML dataset.

For more detailed information on the Slice Prediction use case, including how to build a model, configuration options, and advanced topics, refer to the :ref:`user-guide-slices` documentation.

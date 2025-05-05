.. _tutorial-kpi-windsor-data-access:

Downloading WindsorML data for KPI Prediction (1.7G)
===========================================================

The :ref:`datasets-windsor` is a publicly available dataset hosted in Hugging Face datasets.

Download Commands
-----------------

The tutorial includes a convenient command to download the complete training data (~1.7G):

.. code-block:: shell
    
   ./download-dataset /path/to/dataset 

Replace ``/path/to/dataset`` to your own directory and follow the prompt.

You may download a subset of runs by specifying a prefix at the prompt. For example, ``1*`` gets runs 1, 1x, 1xx and reduces the download size to approximately 530 MB. Please note that you need at least 18 data samples to run the tutorials with the default configurations. You may adjust the train/validation/test sizes during the preprocessing step or modify the batch size during the training step to enable running the tutorials with fewer data samples.

The download command wraps the Hugging Face ``snapshot_download`` function, which you may want to customize for specific downloads. For example, the following code downloads the training data for all runs:

.. code-block:: python

   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="neashton/windsorml",
       repo_type="dataset",
       local_dir="/path/to/dataset",
       allow_patterns=[
           "run_*/windsor_*.stl",
           "run_*/force_mom_*.csv"
       ]
   )

.. note::

   If you encounter a "Too Many Requests" error like this::

       huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests

   This is due to rate limiting. Wait and try again. Hugging Face uses a local cache, so already downloaded files won't need to be re-downloaded on retry.
   If the error continues, set the function argument ``max_workers=1`` to download one file at a time.



Next Step: Training
--------------------

Follow :ref:`tutorial-kpi-windsor-training` to learn how to train a KPI prediction model on the WindsorML dataset.

For more detailed information on the KPI prediction use case, including how to build a model, configuration options, and advanced topics, refer to the :ref:`KPI user guide <user-guide-kpi>` documentation.

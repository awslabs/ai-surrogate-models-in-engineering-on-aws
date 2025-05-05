.. _tutorial-surface-ahmed-data-access:

Downloading AhmedML data for Surface Variable Prediction (43G)
==============================================================

The :ref:`datasets-ahmed` is a publicly available dataset hosted in Hugging Face datasets.

Download Commands
-----------------

The tutorial includes a convenient command to download the complete training data (~43G):

.. code-block:: shell
    
   ./download-dataset /path/to/dataset 

Replace ``/path/to/dataset`` to your own directory and follow the prompt.

You may download a subset of runs by specifying a prefix at the prompt. For example, ``1*`` gets runs 1, 1x, 1xx and reduces the download size to approximately 9.2G. Please note that you need at least 18 data samples to run the tutorials with the default configurations. You may adjust the train/validation/test sizes during the preprocessing step or modify the batch size during the training step to enable running the tutorials with fewer data samples.

The download command wraps the Hugging Face ``snapshot_download`` function, which you may want to customize for specific downloads. For example, the following code downloads the training data for all runs:

.. code-block:: python

   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="neashton/ahmedml",
       repo_type="dataset",
       local_dir="/path/to/dataset",
       allow_patterns=[
           "run_*/boundary_*.vtp",
           "run_*/ahmed_*.stl"
       ]
   )

.. note::

   If you encounter a "Too Many Requests" error like this::

       huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests

   This is due to rate limiting. Wait and try again. Hugging Face uses a local cache, so already downloaded files won't need to be re-downloaded on retry.
   If the error continues, set the function argument ``max_workers=1`` to download one file at a time.


Next Step: Training
--------------------

Follow :ref:`tutorial-surface-ahmed-training` to learn how to train a surface variable prediction model on the AhmedML dataset.

For more detailed information on the surface variable prediction use case, including how to build a model, configuration options, and advanced topics, refer to the :ref:`Surface variable prediction user guide <user-guide-surface>` documentation.

.. _tutorial-kpi-windsor-data-access:

Downloading WindsorML data for KPI Prediction (1.7G)
===========================================================

The :ref:`datasets-windsor` is a publicly available dataset hosted in an S3 bucket. 

.. note::

    We use the `AWS CLI <https://docs.aws.amazon.com/cli/>`_. If you don't already have this installed, please follow the `install guide <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_ and `configure credentials <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`_.

Download Commands
-----------------

The tutorial includes a convenient command to download the complete training data (~1.7G):

.. code-block:: shell
    
   ./download-dataset /path/to/dataset 

Replace ``/path/to/dataset`` to your own directory and follow the prompt.

You may download a subset of runs by specifying a prefix at the prompt. For example, ``1*`` gets runs 1, 1x, 1xx and reduces the download size to approximately 530 MB. Please note that you need at least 18 data samples to run the tutorials with the default configurations. You may adjust the train/validation/test sizes during the preprocessing step or modify the batch size during the training step to enable running the tutorials with fewer data samples.

The download command wraps the AWS CLI ``s3 sync`` command, which you may want to customize for specific downloads. For example, the following command downloads the training data for all runs:

.. code-block:: shell

    aws s3 sync s3://caemldatasets/windsor/dataset /path/to/dataset \
        --exclude "*" \
        --include "run_*/windsor_*.stl" \
        --include "run_*/force_mom_*.csv"

.. note:: 

   Follow the `AWS CLI documentation <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`_ to configure credentials if you get the following error::

       fatal error: An error occurred (InvalidToken) when calling the ListObjectsV2 operation: The provided token is malformed or otherwise invalid.



Next Step: Training
--------------------

Follow :ref:`tutorial-kpi-windsor-training` to learn how to train a KPI prediction model on the WindsorML dataset.

For more detailed information on the KPI prediction use case, including how to build a model, configuration options, and advanced topics, refer to the :ref:`KPI user guide <user-guide-kpi>` documentation.

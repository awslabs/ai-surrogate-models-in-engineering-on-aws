.. _tutorial-slices-windsor-data-access:

Downloading WindsorML data for Slice Prediction (48G)
=============================================================

The :ref:`datasets-windsor` is a publicly available dataset hosted in an S3 bucket. 

.. note::

    We use the `AWS CLI <https://docs.aws.amazon.com/cli/>`_. If you don't already have this installed, please follow the `install guide <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_ and `configure credentials <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`_.

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

The download command wraps the AWS CLI ``s3 sync`` command, which you may want to customize for specific downloads. For example, the following command downloads the training data for all runs:

.. code-block:: shell

   aws s3 sync s3://caemldatasets/windsor/dataset /path/to/dataset \
      --exclude "*" \
      --include "run_*/windsor_*.stl" \
      --include "run_*/images/*"

.. note:: 

   Follow the `AWS CLI documentation <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`_ to configure credentials if you get the following error::

       fatal error: An error occurred (InvalidToken) when calling the ListObjectsV2 operation: The provided token is malformed or otherwise invalid.


Next Step: Training
--------------------

Next, follow :ref:`tutorial-slices-windsor-training` to learn how to train a Slice prediction model on the WindsorML dataset.

For more detailed information on the Slice Prediction use case, including how to build a model, configuration options, and advanced topics, refer to the :ref:`user-guide-slices` documentation.

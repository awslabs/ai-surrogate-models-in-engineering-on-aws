.. _install:

Install
=========================

ML for Simulation Toolkit requires Python >= 3.9, < 3.13 and Pip. It is tested on Ubuntu 22.04 with Cuda 12.1. Running on MacOS and Windows is not tested but may work because there are no OS-specific codes or dependencies.

A GPU is recommended for following the tutorials but you can get started with the packaged sample data using CPU-only.

To get started on AWS, we recommend using the `AWS Deep Learning Base GPU AMI (Ubuntu 22.04) <https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-22-04/>`_ with Cuda and NVIDIA pre-installed on Amazon Elastic Compute Cloud (AWS EC2) instance. The `G5 instance types <https://aws.amazon.com/ec2/instance-types/g5/>`_ with one GPU such as ``g5.xlarge`` or ``g5.2xlarge`` are suitable to complete the MLSimKit tutorials. Follow `these steps to launch an AWS EC2 Linux instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html>`_. We recommend 500G+ storage for the tutorials.

MLSimKit supports multi-GPU training to accelerate performance. Once you are familiar with the toolchain and workflows, we recommend training on larger real-world sized datasets such as :ref:`DrivAerML <datasets-drivaer>` using a ``g5.12xlarge`` or ``g5.48xlarge`` instance that have four and eight GPUs respectively. This will signficantly speed up training time by utilizing all available GPUs.

.. _install-from-source:

Install from Source
-------------------------------

Extract the source distributable (e.g, ``.tar.gz`` or ``.zip``) to a directory e.g, ``mlsimkit``.

First, check your Python version via: ``python3 --version`` and upgrade if possible (up to Python 3.12).

We use `pip <https://pip.pypa.io/en/stable/>`_ to install Python dependencies. Check if pip is already installed: ``python3 -m pip --version``. If not, we recommend installing pip via their `get-pip steps <https://pip.pypa.io/en/stable/installation/#get-pip-py>`_:

#. Download the script, from https://bootstrap.pypa.io/get-pip.py
#. Run ``python3 get-pip.py``

We recommend using a `virtual environment <https://virtualenv.pypa.io/en/latest/index.html>`_ to contain the dependencies. We use the standard ``venv`` module packaged with Python on MacOS but installed separately on Linux/Ubuntu::

    sudo apt install python3-venv        # Linux/Ubuntu only (not MacOS)

Next, create and activate the virtual environment within the ``mlsimkit`` directory: 

.. code-block:: shell

    cd mlsimkit
    python3 -m pip install --upgrade pip # ensure latest pip
    python3 -m venv .venv                # create a .venv directory 
    source .venv/bin/activate            # activate the virtual environment

Install ``mlsimkit`` and dependencies via pip (we recommend ``--edit`` if you want to edit source files):

.. code-block:: shell

    pip3 install --edit .

You now have ``mlsimkit-learn`` installed (versions may be different):

.. code-block:: shell

   % mlsimkit-learn --version
   ML for Simulation Toolkit, version 0.1.0b1.dev35+gc33bc75.d20240508

See the :ref:`Troubleshooting <troubleshooting>` guide if the install did not work. 

Follow :ref:`install-next-steps` to start training.


Install from Source to a Remote Machine (via .whl)
--------------------------------------------------

If you want to install the ``mlsimkit`` python package outside of a virtual environment, you can use a pre-built wheel file (``.whl``). Below, we use the source install to 
build the wheel file, copy the wheel file to the remote machine and then install via ``pip``.

.. note::

   Currently the tutorials are not included in the ``.whl`` Python package. Please copy the tutorials separately. Hint: run ``make sdist`` from the ``mlsimkit`` directory to output a source ``.tar.gz`` file including the tutorials to the ``dist/`` folder.

Follow these steps to build the wheel file and install the ``mlsimkit`` python package:

1. Ensure you have the :ref:`source installation <install-from-source>` working. 

2. From your ``mlsimkit`` source install directory, run the following command to build the wheel file from current source:

   .. code-block:: shell

      cd <mlsimkit>
      make wheel

   This command will create a ``.whl`` file in the ``dist/`` directory. The name of the ``.whl`` file created by this command is printed to the terminal.

3. Copy the newly created ``.whl`` file in ``dist/`` to the remote machine. 
   
4. On the remote machine, install using ``pip``:

   .. code-block:: shell

      pip install mlsimkit-<your version>.whl --prefix=/opt/mlsimkit
      export PATH=$PATH:/opt/mlsimkit/bin
      export PYTHONPATH=$PYTHONPATH:/opt/mlsimkit/lib/python3.11/site-packages

Replace ``mlsimkit-<your version>.whl`` with the filename from step (2). e.g, ``mlsimkit-0.1.0b0-py3-none-any.whl``.

Replace ``/opt/mlsimkit`` with your desired installation directory, and update the ``PYTHONPATH`` with the appropriate Python version e.g., ``python3.11``.

After following these steps, the package will be installed on the remote machine, and you can use it without the need for a virtual environment.

.. _install-next-steps:

Next Steps
----------

After installing the ML for Simulation Toolkit, proceed to running training pipelines and make predictions:

1. **Quickstart:**  Follow :doc:`KPI <quickstart-kpi>`, :doc:`Slice Prediction <quickstart-slices>` or :doc:`Surface Prediction <quickstart-surface>` quickstart guides and train a model and make predictions on sample data in 15 minutes. You will familiarize yourself with the CLI and configuration tools.

2. **Tutorials:**  Reproduce results on a real dataset for one of the use cases by following the :ref:`tutorial-kpi-windsor` or :ref:`tutorial-slices-windsor` that use the :ref:`datasets-windsor`; or follow :ref:`tutorial-surface-ahmed` that uses the :ref:`datasets-ahmed`. See codes in ``tutorials/`` in the source code for walkthroughs on other datasets.

3. **Customize a use case**: Once you have reproduced results following the tutorials, explore in detail how to use your own datasets by diving into the :doc:`KPI prediction <user-guide-kpi>`, :doc:`Slice prediction <user-guide-slice>`, and :doc:`Surface variable prediction <user-guide-surface>` users guides. You will be ready to experiment with your data and customize model codes for your own use cases.

4. **Running on AWS ParallelCluster (coming soom)**: Train at scale on AWS ParallelCluster.

5. **Running inside a SageMaker Notebook (coming soom)**: A guide to setting up MLSimKit with a your SageMaker Notebook for interactive development.

6. **Use the MLSimKit SDK in your Python code**: For example, you may want to integrate model codes from other libraries and utilize the MLSimKit CLI/Configuration framework. Please refer to :ref:`api-index` to learn more.

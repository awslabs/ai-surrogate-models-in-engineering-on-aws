.. _troubleshooting:

Troubleshooting
================

Before troubleshooting, it's helpful to gather information about your system's hardware and software configurations. Common issues are related to CUDA version mismatches as PyTorch includes its own CUDA binaries. You want to ensure that PyTorch is using its own CUDA binaries and not conflicting with other CUDA installations on your system.

Check Torch, CUDA, and MLSimKit Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the following Python code to print the installed versions of Torch, CUDA used by Torch, and MLSimKit:

.. code-block:: python

    import torch, mlsimkit
    print(torch.__version__)
    print(torch.version.cuda)
    print(mlsimkit.__version__)

Or using the command-line:

.. code-block:: bash

    python -c "import torch; import mlsimkit; print(f'Torch: {torch.__version__}\nCUDA: {torch.version.cuda}\nMLSimKit: {mlsimkit.__version__}')"

You will see output like:

.. code-block:: text

    Torch: 2.2.2+cu121
    CUDA: 12.1
    MLSimKit: 0.1.0b0

Check NVIDIA Driver Version
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the ``nvidia-smi`` command to check the installed NVIDIA driver version:

.. code-block:: shell

    $ nvidia-smi

Look for the "Driver Version" field in the output.


Common Issues
^^^^^^^^^^^^^^

GPU out-of-memory (OOM) issues
---------------------------------
You may run into an out-of-memory (OOM) issue while training a model, but there are steps that can be taken to address it.  The following are some suggestions of things to try:

1. Change training hyperparameters to reduce the size of the model or in-memory data such as ``batch-size``, ``message-passing-steps``, ``hidden-size``, ``ae.start-out-channel`` and ``ae.div-rate``.
2. Try uing mixed precision (``--mixed-precision fp16`` or ``--mixed-precision bf16``).
3. Make sure you are not using the deterministic flag, as this will increase the memory requirements of the model training (``--deterministic``).
4. Reduce the size of your training data in preprocessing.  For instance in the slices use case, the resolution can be reduced via ``--resolution`` argument, and the number of input channels can be reduced via the ``--grayscale`` argument.  For input meshes in the KPI and Surface use cases, ``--downsample-remaining-perc`` can be used to decrease the size of the meshes.
5. Reduce the size of your training data externally.  Alternatively, outside methods can be used to reduce the size of the data.  For instance the number of slices can be reduced or external software can be used to decimate meshes.
6. Try training on the cpu (``--device cpu``) instead.  Note that this can lead to much longer training times which may be undesirable.
7. Make sure other processes aren't consuming your GPU's memory.  This can be done by using the ``nvidia-smi`` command.
8. Train on a bigger GPU with more memory.


``"Could not load library libcudnn_cnn_train.so"`` on Deep Learning AMI
--------------------------------------------------------------------------

``mlsimkit`` depends on Pytorch, which installs its own Cuda binaries. You may see this error if your environment has a version mismatch:

.. code-block:: shell

    Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8

One solution is to remove Cuda from the system entirely and/or install the identical version. For example,
see the `Deep Learning tutorial <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html>`_. 

Alternatively, you may remove the Cuda binaries from your library path when running ``torch`` applications. For example, in the Deep Learning AMI for Ubuntu 22.04, the default ``LD_LIBRARY_PATH`` includes Cuda 12.1:

.. code-block:: shell

    $ echo $LD_LIBRARY_PATH
    /opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda-12.1/lib:/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1:/usr/local/cuda-12.1/targets/x86_64-linux/lib/:/usr/local/cuda-12.1/extras/CUPTI/lib64:/usr/local/lib:/usr/lib


Remove the ``cuda-12.1`` directories and then ``Pytorch`` will use its own cuda binaries:

.. code-block:: shell

   export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/lib:/usr/lib

``mlsimkit-accelerate`` train command hangs or timeouts on multi-GPU instances
------------------------------------------------------------------------------

The train processes (across use cases) can hang and/or timeout when running on a NVIDIA multiple GPU compute when executing via ``mlsimkit-accelerate`` or ``accelerate launch``.  This can be caused by NVIDIA drivers such as ``NVIDIA-SMI 555.42.06``.  To fix the issue you can remove and install a different NVIDIA driver that is compatible:

.. code-block:: shell

    $ sudo apt-get --purge remove nvidia-kernel-source-555
    $ sudo apt-get install --verbose-versions cuda-drivers-535


``"Cannot convert a MPS Tensor to float64 dtype..."`` or ``"fp16 mixed precision requires a GPU (not 'mps')"`` on MacOS
------------------------------------------------------------------------------------------------------------------------

On older MacOS hardware, you may need to force CPU-only for training if you see one of the following errors:

.. code-block:: text

    TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.

.. code-block:: text

    Error: fp16 mixed precision requires a GPU (not 'mps')

Use ``--device cpu`` for all training commands. For example::

    mlsimkit-learn kpi train --device cpu 
    mlsimkit-learn slices train-image-encoder --device cpu 
    mlsimkit-learn slices train-prediction --device cpu


``"qt.qpa.plugin: Could not load the Qt platform plugin..."``
-------------------------------------------------------------------------------------------

The Slice prediction code utilizes the ``opencv-python`` package, which relies on a specific version of the Qt library. In some cases, this version may conflict with other Qt installations already present on your system, resulting in the following error::

    QObject::moveToThread: Current thread (0x55fff86fe4f0) is not the object's thread (0x55fff91b1b70).
    Cannot move to target thread (0x55fff86fe4f0)

    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/ubuntu/miniconda3/lib/python3.12/site-packages/cv2/qt/plugins" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: xcb, eglfs, minimal, minimalegl, offscreen, vnc, webgl.

    Aborted (core dumped)

To prevent such conflicts, we recommend setting up and using a virtual environment for running the Slice prediction code. This approach isolates the required dependencies, including the necessary Qt version, from your system's global environment.
Please refer to the :ref:`install` section for instructions on creating and activating a virtual environment.


.. _troubleshooting_file_descriptors:

Multi-processes warning "may exceed file descriptors limits" (``"RuntimeError: received 0 items of ancdata"``)
-------------------------------------------------------------------------------------------------------------------

During the ``kpi preprocess`` or ``surface preprocess`` steps, MLSimKit uses Python's ``multiprocessing`` module to parallelize the processing of mesh files across multiple CPU cores. However, this parallelization can sometimes exceed the system's file descriptor limit, leading to the following error:

.. code-block:: text

    RuntimeError: received 0 items of ancdata

This error occurs when the number of open file descriptors (used for inter-process communication) exceeds the system's limit. The 
likelihood of encountering this issue increases when processing a higher number of simulation runs,

You will encounter a warning in the logs when running the `mlsimkit-learn kpi preprocess` or `mlsimkit-learn surface preprocess` commands with multiple processes:

.. code-block:: text

    [WARNING] Using multi-processes (2) for preprocessing data. May exceed file descriptor limits, use 'ulimit -n'. See Troubleshooting in the user guide.

**Workaround: Increase your File Descriptor Limit**

If you prefer to continue preprocessing with multiple CPUs, increase the file descriptor limit on your system by running ``ulimit -n <higher_value>`` before running ``mlsimkit-learn``. However, this may require administrative privileges and may not be a viable option in some environments.

**Workaround: Use a Single Process**

To avoid this issue enitrely, use a single process for preprocessing by setting ``--num-processes 1``:

KPI command

.. code-block:: text

    mlsimkit-learn --config training.yaml --log.prefix-dir logs/preprocess kpi preprocess --num-processes 1


Surface command

.. code-block:: text

    mlsimkit-learn --config training.yaml --log.prefix-dir logs/preprocess surface preprocess --num-processes 1
    

This workaround eliminates the need for inter-process communication and shared memory segments, preventing the file descriptor limit from being exceeded.



.. _troubleshooting_xvfb:
   
Surface view screenshots error "PyVista will likely segfault when rendering" (``"bad X server connection"``)
------------------------------------------------------------------------------------------------------------

Outputting screenshots of the surface prediction data requires 3D rendering. On systems without a display, you will see an error like this:

.. code-block:: shell

    $ mlsimkit-learn --config training.yaml surface view --no-gui
    /usr/local/lib/python3.10/dist-packages/pyvista/plotting/plotter.py:151: UserWarning:
    This system does not appear to be running an xserver.
    PyVista will likely segfault when rendering.

    Try starting a virtual frame buffer with xvfb, or using
      ``pyvista.start_xvfb()``

      warnings.warn(
    2024-06-03 00:15:35.535 (   1.463s) [    7FE7D7B70480]vtkXOpenGLRenderWindow.:456    ERR| vtkXOpenGLRenderWindow (0x561067609970): bad X server connection. DISPLAY=
    [ERROR] bad X server connection. DISPLAY=
    Aborted (core dumped)

**Linux/Ubuntu:** 

We support the package Xvfb (X virtual framebuffer) on Linux/Ubuntu. Install this package::

    sudo apt install xvfb

Now start Xvfb when starting the viewer to enable rendering on remote machines::

    mlsimkit-learn surface view --start-xvfb ...
    
See the `PyVista documentation 'Running on Remote Servers' <https://docs.pyvista.org/version/stable/getting-started/installation.html#running-on-remote-servers>`_ for more details. 

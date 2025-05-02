.. _datasets-windsor:

WindsorML Dataset
======================

The WindsorML dataset is a publicly available dataset `licensed as CC BY-SA 4.0 <https://caemldatasets.s3.amazonaws.com/windsor/dataset/LICENSE.txt>`_ and distributed separately to MLSimKit. It is a collection of high-fidelity CFD simulations showing different geometric variants of a body for automotive aerodynamics modeling. Please see the `README <https://caemldatasets.s3.amazonaws.com/windsor/dataset/README.txt>`_ for additional details.

Downloading the training data (170G)
-------------------------------------

The Windsor dataset is hosted in an S3 bucket.

Use the following `AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_ command to download the data required for the training tutorials:

.. code-block:: shell

   aws s3 sync s3://caemldatasets/windsor/dataset /path/to/dataset \
      --exclude "*" \
      --include "run_*/windsor_*.stl" \
      --include "run_*/images/*" \
      --include "run_*/boundary*.vtu" \
      --include "run_*/force_mom_*.csv"

Replace ``/path/to/dataset`` to your own directory. 

The :ref:`Slices<tutorial-slices-windsor-data-access>` and :ref:`KPI<tutorial-kpi-windsor-data-access>` tutorials have steps to download only the necessary data for each. 
The entire dataset is ~8TB and includes high-resolution meshes not required for training. 

.. _dataset-contents-windsor:

Understanding the dataset
-------------------------

The dataset contains 355 cases, each representing a unique geometry variant.  The cases are organized into separate folders named ``run_0`` to ``run_354``, one for each simulation case.
The variability in geometry across the 355 cases leads to a diverse set of flow physics, making this dataset well-suited for machine learning model development.

Within each run folder is a standard set of files:

.. code-block:: shell

    run_0/
    ├── boundary_0.vtu
    ├── force_mom_0.csv
    ├── force_mom_varref_0.csv
    ├── geo_parameters_0.csv
    ├── images
    │   ├── pressureavg
    │   │   ├── *.png
    │   ├── rstress_xx
    │   │   ├── *.png
    │   ├── rstress_yy
    │   │   ├── *.png
    │   ├── rstress_zz
    │   │   ├── *.png
    │   ├── velocityxavg
    │   │   ├── *.png
    │   └── windsor_0.png
    ├── volume_0.vtu
    ├── windsor_0.stl
    └── windsor_0.stp


- ``windsor_<run #>.stl`` - The surface geometry definition in STL format
- ``boundary_<run #>.vtu`` - Simulation results on the surface 
- ``volume_<run #>.vtu`` - Volumetric simulation outputs
- ``force_mom_<run #>.csv`` - Time-averaged force and moment coefficients 
- ``force_mom_varref_<run #>.csv`` - Time-averaged force and moment coefficients using unique reference area per geometry
- ``images/`` - Folder containing slice images through the volume
- ``windsor_<run #>.png`` - Image of the windsor body (see below)

.. image:: ../images/windsor_0.png
   :width: 1000
   :height: 500
   :alt: Figure 1. An example Windsor Body

The slice images show simulation variables like pressure and velocity captured on 2D planes along the X, Y and Z axes. Multiple views are available.

Slice Images Views
------------------

The ``images`` folder within each case contains subfolders organizing the slice images by simulation output variable.

Within each variable folder (e.g. ``pressureavg``, ``velocityxavg``), there are multiple image sets showing different sliced views through the volume.

The main image sets are:

**Z-Axis Slices**

Filename pattern: ``view1_constz_*.png``

Slices along the Z-axis, showing the XY-plane at different Z positions.  Useful for visualizing the flow as it passes over the geometry from front to back.

.. image:: ../images/view1_constz_scan_0005.png
   :width: 400
   :height: 225
   :alt: Figure 2. An example Z-Axis slice

**X-Axis Slices** 

Filename pattern: ``view2_constx_*.png`` 

These are slices along the X-axis, showing the YZ-plane at different X positions.

.. image:: ../images/view2_constx_scan_0005.png
   :width: 400
   :height: 225
   :alt: Figure 3. An example X-Axis slice

**Y-Axis Slices** 

Filename pattern: ``view3_consty_*.png``

Slices along the Y-axis, showing the XZ-plane. Gives a top-down view at different heights.

.. image:: ../images/view3_consty_scan_0010.png
   :width: 400
   :height: 225
   :alt: Figure 4. An example Y-Axis slice

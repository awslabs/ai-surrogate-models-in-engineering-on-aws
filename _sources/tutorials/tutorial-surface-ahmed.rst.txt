.. _tutorial-surface-ahmed:

Surface Variable Prediction Tutorial
====================================


This tutorial demonstrates how to run the surface variable prediction pipeline on the publicly available :ref:`datasets-ahmed`, which contains computational fluid dynamics (CFD) simulations for a vehicle geometry.

Surface variable prediction is used to predict surface variable values of 3D geometry meshes. In computational fluid dynamics (CFD), examples of common surface variables are pressure (or pressure coefficient) and wall shear stress (or skin friction coefficient).

The tutorial is divided into three parts.

1. First, we'll :ref:`download the data <tutorial-surface-ahmed-data-access>` necessary for surface variable prediction.
2. Second, we :ref:`train a surface variable prediction model <tutorial-surface-ahmed-training>` on the :ref:`datasets-ahmed`, including data preparation, model training, and evaluating the trained model's performance.
3. Third, we use the trained model to :ref:`predict surface variable values for new, unseen geometries <tutorial-surface-ahmed-prediction>` without ground truth simulation data.

By following these tutorials, you'll learn how to:

- Access the subset of data required for this use case
- Create data manifests for the AhmedML dataset
- Configure and run the surface variable prediction pipeline for training and prediction
- Interpret the training and prediction results, including metrics
- Apply the trained model to predict surface variable values for new geometries

For more detailed information on the surface variable prediction use case, including how to build a model, configuration options, and advanced topics, refer to the :ref:`surface variable prediction user guide <user-guide-surface>` documentation.

Let's get started.

.. toctree::
   :maxdepth: 1
   :numbered:

   tutorial-surface-ahmed-data-access
   tutorial-surface-ahmed-training
   tutorial-surface-ahmed-prediction

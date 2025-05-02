.. _tutorial-kpi-windsor:

KPI Prediction Tutorial
=======================


This tutorial demonstrates how to run the KPI prediction pipeline on the publicly available :ref:`datasets-windsor`, which contains computational fluid dynamics (CFD) simulations for a vehicle geometry.

KPI prediction is used to predict key performance indicators (KPIs) of 3D geometry meshes. In computational fluid dynamics (CFD), examples of common KPIs are drag coefficient (Cd) and lift coefficient (Cl).

The tutorial is divided into three parts.

1. First, we'll :ref:`download the data <tutorial-kpi-windsor-data-access>` necessary for KPI prediction.
2. Second, we :ref:`train a KPI prediction model <tutorial-kpi-windsor-training>` on the :ref:`datasets-windsor`, including data preparation, model training, and evaluating the trained model's performance.
3. Third, we use the trained model to :ref:`predict KPIs for new, unseen geometries <tutorial-kpi-windsor-prediction>` without ground truth simulation data.

By following these tutorials, you'll learn how to:

- Access the subset of data required for this use case
- Create data manifests for the WindsorML dataset
- Configure and run the KPI prediction pipeline for training and prediction
- Interpret the training and prediction results, including plots and metrics
- Apply the trained model to predict KPIs for new geometries

For more detailed information on the KPI prediction use case, including how to build a model, configuration options, and advanced topics, refer to the :ref:`KPI user guide <user-guide-kpi>` documentation.

Let's get started.

.. toctree::
   :maxdepth: 1
   :numbered:

   tutorial-kpi-windsor-data-access
   tutorial-kpi-windsor-training
   tutorial-kpi-windsor-prediction

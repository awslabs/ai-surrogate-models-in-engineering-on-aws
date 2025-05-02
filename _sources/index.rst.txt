.. _welcome:

Welcome to AI Surrogate Models in Engineering on AWS (MLSimKit) |release|
===================================


AI Surrogate Models in Engineering on AWS (project name: ML for Simulation Toolkit or MLSimKit) provides engineers and designers a starting point for near real-time predictions of physics-based simulations using ML models. It enables engineers to quickly iterate on their designs and see immediate results rather than having to wait hours for a full physics-based simulation to run.

The toolkit is a collection of commands and SDKs that insert into the traditional iterative design-simulate workflow (see diagram below), where it can take days to run a single cycle. Instead, the “train-design-predict” workflow becomes an additional choice to train models on simulated ground truth and then use the newly trained models for predictions on new designs in minutes.

.. image:: images/mlsimkit-workflow-diagram.png
   :width: 800
   :alt: Figure 1. Traditional vs. MLSimKit-based workflow diagram

Get Started
------------------------

.. toctree::
    :maxdepth: 1

    user/install
    user/quickstart-kpi
    user/quickstart-surface
    user/quickstart-slices
    user/troubleshooting

Tutorials 
------------------------
Explore tutorials that dive deeper into specific use cases for training on larger datasets. 

.. toctree::
    :maxdepth: 1

    tutorials/tutorial-kpi-windsor
    tutorials/tutorial-surface-ahmed
    tutorials/tutorial-slices-windsor

External Datasets
------------------------

Learn about external publicly available datasets that are supported by the MLSimKit tutorials.

.. toctree::
    :maxdepth: 1

    datasets/ahmed
    datasets/windsor
    datasets/drivaer

User Guides
------------------------

Discover comprehensive user guides to configure and tune model parameters for training on your custom datasets. Learn how to use the MLSimKit SDK to build your own applications and setup MLFlow for tracking results with a UI dashboard.

.. toctree::
    :maxdepth: 1

    user/user-guide-kpi
    user/user-guide-surface
    user/user-guide-slice
    user/notebook-guide
    user/mlflow-guide

Developer Guides
---------------------------

Customize MLSimKit by learning how the code is structured and the various modules to modify for your own custom use cases.

.. toctree::
    :maxdepth: 2

    dev/guide.rst
    dev/learn.rst
    dev/cli-toolkit

API Reference
---------------------------

.. toctree::
    :maxdepth: 2

    dev/api.rst

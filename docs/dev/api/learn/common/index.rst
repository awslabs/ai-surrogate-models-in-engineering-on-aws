.. _api-learn-common:

:mod:`mlsimkit.learn.common` -- MLSimKit Learn Common
==========================================================

This part of the documentation covers the common interfaces shared across use cases like KPI and Surface prediction models.

Training
----------------

The Training module contains the main training loop function ``train()`` and related helpers.

.. automodule:: mlsimkit.learn.common.training
   :members:
   :undoc-members:
   

Tracking 
-----------------------

The Tracking module wraps MLFlow to make it easier to track and report experiments.

.. automodule:: mlsimkit.learn.common.tracking
   :members:
   :undoc-members:


Mesh Utilities
----------------

The Mesh module contains helpers for third-parties to process meshes such as down-sampling and format conversions.

.. automodule:: mlsimkit.learn.common.mesh
   :members:
   :undoc-members:


Miscellaneous Utilities
-----------------------

The Utils module collects mostly unrelated functions that should eventually be moved into specifically-named modules.

.. automodule:: mlsimkit.learn.common.utils
   :members:
   :undoc-members:


Schemas
-----------------------

The learning schemas implement common data classes used across the codebase.

.. automodule:: mlsimkit.learn.common.schema.optimizer
   :members:
   :undoc-members:
   :no-index:

.. automodule:: mlsimkit.learn.common.schema.project
   :members:
   :undoc-members:
   :no-index:

.. automodule:: mlsimkit.learn.common.schema.training
   :members:
   :undoc-members:
   :no-index:

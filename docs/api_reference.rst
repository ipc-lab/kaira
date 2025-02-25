Kaira API Reference
===================

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------
This documentation provides a comprehensive API reference for Kaira. Each section below is generated from the source code.

Core
----
.. automodule:: kaira.core
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. currentmodule:: kaira.core

.. autosummary::
   :toctree: generated
   BaseChannel
   BaseConstraint
   BaseMetric
   BaseModel
   BasePipeline

Channels
--------
.. automodule:: kaira.channels
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. currentmodule:: kaira.channels

.. autosummary::
   :toctree: generated
   PerfectChannel

Constraints
-----------
.. automodule:: kaira.constraints
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. currentmodule:: kaira.constraints

.. autosummary::
   :toctree: generated
   AveragePowerConstraint
   ComplexAveragePowerConstraint
   ComplexTotalPowerConstraint
   TotalPowerConstraint

Pipelines
---------
.. automodule:: kaira.pipelines
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. currentmodule:: kaira.pipelines

.. autosummary::
   :toctree: generated
   DeepJSCCPipeline

Models
------
.. automodule:: kaira.models
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. currentmodule:: kaira.models

.. autosummary::
   :toctree: generated
   components.∏

Utilities
---------
.. automodule:: kaira.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. currentmodule:: kaira.utils

.. autosummary::
   :toctree: generated
   :no-index:
   snr_db_to_linear
   snr_linear_to_db
   to_tensor

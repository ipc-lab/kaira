Kaira API Reference
===================

This documentation provides a comprehensive reference of Kaira's components organized by functional category.
Each component is documented with its parameters, methods, and usage examples.

.. contents:: Table of Contents
   :depth: 2
   :local:

Core Components
---------------

Core components provide the fundamental abstractions for building communication systems in Kaira.
These base classes define interfaces that all derived implementations must adhere to.

.. currentmodule:: kaira.core

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseChannel
   BaseConstraint
   BaseMetric
   BaseModel
   BasePipeline

Channels
--------

Channel implementations model the transmission medium between sender and receiver.
They simulate various channel conditions and noise models.

.. currentmodule:: kaira.channels

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PerfectChannel
   AWGNChannel
   ComplexAWGNChannel

.. seealso::
   See :class:`~kaira.core.BaseChannel` for the interface all channels must implement.

Constraints
-----------

Constraints enforce signal power limitations and other physical constraints on transmitted signals.

.. currentmodule:: kaira.constraints

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AveragePowerConstraint
   ComplexAveragePowerConstraint
   ComplexTotalPowerConstraint
   TotalPowerConstraint

.. seealso::
   See :class:`~kaira.core.BaseConstraint` for the interface all constraints must implement.

Pipelines
---------

Pipelines integrate encoders, decoders, and channels into end-to-end communication systems.

.. currentmodule:: kaira.pipelines

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DeepJSCCPipeline

.. seealso::
   See :class:`~kaira.core.BasePipeline` for the interface all pipelines must implement.

Models
------

Models implement the neural network architectures for encoders, decoders, and other learnable components.

Components
^^^^^^^^^^

Common components that can be reused across different model architectures.

.. currentmodule:: kaira.models.components

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AFModule

Image Models
^^^^^^^^^^^^^^^^^^^

Specialized models for image transmission tasks.

.. currentmodule:: kaira.models.image

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DeepJSCCQ2Encoder
   DeepJSCCQ2Decoder

.. seealso::
   These models can be used with :class:`~kaira.pipelines.DeepJSCCPipeline` for end-to-end image transmission.

Metrics
-------

Metrics evaluate the quality of transmitted and reconstructed signals.

.. currentmodule:: kaira.metrics

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PeakSignalNoiseRatio
   PSNR
   StructuralSimilarityIndexMeasure
   SSIM
   MultiScaleSSIM
   LearnedPerceptualImagePatchSimilarity

.. seealso::
   See :class:`~kaira.core.BaseMetric` for the interface all metrics must implement.

Utilities
---------

Utility functions to assist with common tasks across the library.

.. currentmodule:: kaira.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   snr_db_to_linear
   snr_linear_to_db
   to_tensor

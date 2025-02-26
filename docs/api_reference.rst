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
   BaseModulator
   BaseDemodulator

Channels
--------

Channel implementations model the transmission medium between sender and receiver.
They simulate various channel conditions and noise models.

.. currentmodule:: kaira.channels

Basic Channels
^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PerfectChannel
   AWGNChannel
   ComplexAWGNChannel

Fading Channels
^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   RayleighChannel
   RicianChannel
   FrequencySelectiveChannel

Hardware Impairments
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PhaseNoiseChannel
   IQImbalanceChannel
   NonlinearChannel
   RappModel

Composition
^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ChannelPipeline
   ParallelChannels

Utilities
^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   snr_to_noise_power
   noise_power_to_snr
   calculate_snr
   evaluate_ber
   plot_channel_response
   plot_constellation
   plot_impulse_response
   measure_snr_vs_param
   plot_snr_vs_param
   evaluate_channel_ber
   plot_ber_vs_snr

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

Modulations
-----------

The modulations package provides digital modulation schemes for communications systems,
including modulators, demodulators, and visualization tools for constellation diagrams.

Modulation Schemes
^^^^^^^^^^^^^^^^^^

Concrete implementations of various digital modulation schemes:

.. currentmodule:: kaira.modulations

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PSKModulator
   PSKDemodulator
   BPSKModulator
   BPSKDemodulator
   QPSKModulator
   QPSKDemodulator
   QAMModulator
   QAMDemodulator
   PAMModulator
   PAMDemodulator
   OQPSKModulator
   OQPSKDemodulator
   Pi4QPSKModulator
   Pi4QPSKDemodulator
   DPSKModulator
   DPSKDemodulator
   DBPSKModulator
   DBPSKDemodulator
   DQPSKModulator
   DQPSKDemodulator

Constellation Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tools for visualizing and analyzing modulation constellations.

.. currentmodule:: kaira.modulations

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ConstellationVisualizer

.. currentmodule:: kaira.modulations.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   plot_constellation
   binary_to_gray
   gray_to_binary
   calculate_theoretical_ber

Benchmarking Tools
^^^^^^^^^^^^^^^^^^

Tools for testing and comparing modulation schemes.

.. currentmodule:: kaira.modulations.benchmark

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   awgn_channel
   measure_ber
   plot_ber_curve
   compare_modulation_schemes
   measure_throughput
   benchmark_modulation_schemes

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

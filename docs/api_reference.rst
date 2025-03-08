Kaira API Reference
=====================================

.. note::
   Kaira version |version| documentation. For older versions, please refer to the version selector above.

This documentation provides a comprehensive reference of Kaira's components organized by functional category.
Each component is documented with its parameters, methods, and usage examples.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
--------------

Kaira is a modular toolkit for wireless communication systems built on PyTorch. The library is organized into
several key modules that handle different aspects of communication systems:

- **Channels**: Model transmission mediums with various noise and distortion characteristics
- **Constraints**: Enforce practical limitations on transmitted signals
- **Metrics**: Evaluate quality and performance of communication systems
- **Models**: Implement neural network architectures for encoding and decoding
- **Modulations**: Implement digital modulation schemes for wireless transmission
- **Pipelines**: Integrate components into end-to-end communication systems
- **Losses**: Provide objective functions for training neural networks
- **Utilities**: Helper functions and tools for common operations

Base Components
--------------------------

Base classes define the fundamental interfaces that all implementations must adhere to.
These abstract classes establish the contract that derived classes must fulfill.

.. currentmodule:: kaira

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   channels.BaseChannel
   constraints.BaseConstraint
   metrics.BaseMetric
   models.BaseModel
   pipelines.BasePipeline
   modulations.BaseModulator
   modulations.BaseDemodulator
   losses.BaseLoss

Channels
--------------

Channel models simulate the transmission medium between sender and receiver.
They apply noise, distortion, fading, and other effects that impact signal quality in real-world scenarios.

.. currentmodule:: kaira.channels

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseChannel
   LambdaChannel

   # Perfect/Identity channels
   PerfectChannel
   IdentityChannel
   IdealChannel

   # Analog channels
   AWGNChannel
   GaussianChannel
   LaplacianChannel
   PoissonChannel
   PhaseNoiseChannel
   FlatFadingChannel
   NonlinearChannel

   # Digital channels
   BinarySymmetricChannel
   BinaryErasureChannel
   BinaryZChannel

.. seealso::
   :class:`~kaira.pipelines.FadingChannelPipeline` for working with fading channels in a complete pipeline.
   :class:`~kaira.modulations.BaseModulator` for modulation schemes that prepare signals for channel transmission.

Constraints
--------------------

Constraints enforce signal limitations that must be satisfied in physical systems.
These include power limitations, hardware restrictions, and regulatory requirements.

.. currentmodule:: kaira.constraints

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseConstraint
   CompositeConstraint

   # Power constraints
   TotalPowerConstraint
   AveragePowerConstraint
   PAPRConstraint

   # Antenna constraints
   PerAntennaPowerConstraint

   # Signal constraints
   PeakAmplitudeConstraint
   SpectralMaskConstraint

Constraint Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Helper functions to create and combine constraints for common scenarios.

.. currentmodule:: kaira.constraints.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   combine_constraints
   create_ofdm_constraints
   create_mimo_constraints

.. seealso::
   :class:`~kaira.pipelines.OFDMPipeline` and :class:`~kaira.pipelines.MIMOPipeline` for using these constraints in complete systems.

Metrics
-------------

Metrics evaluate the quality of transmitted and reconstructed signals.
They quantify performance in terms of accuracy, fidelity, and perceptual quality.

.. currentmodule:: kaira.metrics

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseMetric
   CompositeMetric

   # Image quality metrics
   PeakSignalNoiseRatio
   PSNR
   StructuralSimilarityIndexMeasure
   SSIM
   MultiScaleSSIM
   LearnedPerceptualImagePatchSimilarity
   LPIPS

   # Signal metrics
   SignalToNoiseRatio
   SNR
   BitErrorRate
   BER
   BlockErrorRate
   BLER

Metric Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Helper functions for working with metrics and formatting results.

.. currentmodule:: kaira.metrics.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   compute_multiple_metrics
   format_metric_results
   visualize_metrics_comparison
   benchmark_metrics
   batch_metrics_to_table
   print_metric_table
   summarize_metrics_over_batches

Metric Factories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Factory functions to simplify creation of commonly used metric combinations.

.. currentmodule:: kaira.metrics

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   create_image_quality_metrics
   create_composite_metric

Metric Registry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Registry system for dynamic metric creation and discovery.

.. currentmodule:: kaira.metrics

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   register_metric
   create_metric
   list_metrics

Models
------------

Models implement the neural network architectures for encoders, decoders, and other learnable components.
These form the core of learning-based communication systems.

.. currentmodule:: kaira.models

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseModel
   ModelRegistry

Model Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reusable neural network building blocks for constructing communication models.

.. currentmodule:: kaira.models.components

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   # Common neural network components for building models
   AFModule

Image Models
^^^^^^^^^^^^^^^^^^^^

Models specialized for image transmission and reconstruction.

.. currentmodule:: kaira.models.image

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   # Image-specific neural network models
   DeepJSCCQEncoder
   DeepJSCCQDecoder
   DeepJSCCQ2Encoder
   DeepJSCCQ2Decoder

Losses
------------

Loss functions used for training neural network models with different optimization objectives.

.. currentmodule:: kaira.losses

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseLoss
   CompositeLoss
   adversarial
   audio
   text
   image
   multimodal


Modulations
--------------------

Digital modulation schemes for mapping bits to symbols for wireless transmission.
These transform digital data into waveforms suitable for transmission over physical channels.

.. currentmodule:: kaira.modulations

Basic Modulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base classes and simple modulation schemes.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseModulator
   BaseDemodulator

   # Identity schemes
   IdentityModulator
   IdentityDemodulator

PSK Modulation
^^^^^^^^^^^^^^^^^^^^^^^^^

Phase Shift Keying modulation schemes, which encode data by varying the phase of a carrier wave.

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

   # Special PSK variants
   Pi4QPSKModulator
   Pi4QPSKDemodulator
   OQPSKModulator
   OQPSKDemodulator

QAM & PAM Modulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quadrature Amplitude Modulation and Pulse Amplitude Modulation schemes.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   QAMModulator
   QAMDemodulator
   PAMModulator
   PAMDemodulator

Differential Modulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Differential encoding schemes that encode information in the change between symbols.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DPSKModulator
   DPSKDemodulator
   DBPSKModulator
   DBPSKDemodulator
   DQPSKModulator
   DQPSKDemodulator

Modulation Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Helper functions for working with modulation schemes and analyzing their properties.

.. currentmodule:: kaira.modulations.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   binary_to_gray
   gray_to_binary
   binary_array_to_gray
   gray_array_to_binary
   plot_constellation
   calculate_theoretical_ber
   calculate_spectral_efficiency

Pipelines
------------------

Pipelines integrate encoders, decoders, channels, and other components into end-to-end communication systems.
They provide a high-level interface for running simulations and training models.

.. currentmodule:: kaira.pipelines

Generic Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

General-purpose pipelines that can be used with any types of components.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BasePipeline
   SequentialPipeline
   ParallelPipeline
   BranchingPipeline

Communication System Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specialized pipelines for specific communication scenarios and techniques.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DeepJSCCPipeline
   WynerZivPipeline
   WynerZivCorrelationModel
   OFDMPipeline
   MIMOPipeline
   FadingChannelPipeline
   FadingType
   FeedbackChannelPipeline

Utilities
------------------

General utility functions for data manipulation, visualization, and configuration.

.. currentmodule:: kaira.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   seed_everything
    to_tensor
    calculate_num_filters_image
    snr_db_to_linear
    snr_linear_to_db
    snr_to_noise_power
    noise_power_to_snr
    calculate_snr
    add_noise_for_snr
    estimate_signal_power

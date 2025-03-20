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
- **Models**: Implement neural network architectures for encoding/decoding and end-to-end communication systems
- **Modulations**: Implement digital modulation schemes for wireless transmission
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
   
   # Registry
   ChannelRegistry

.. seealso::
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
   
   # Basic functional constraints
   IdentityConstraint
   LambdaConstraint
   
   # Registry
   ConstraintRegistry

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
   FrameErrorRate
   FER
   SymbolErrorRate
   SER
   
   # Registry
   MetricRegistry

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
   ConfigurableModel
   DeepJSCCModel
   FeedbackChannelModel
   WynerZivModel
   ModelRegistry

Generic Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic model implementations that serve as building blocks for more complex architectures.

.. currentmodule:: kaira.models.generic

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   IdentityModel
   SequentialModel
   ParallelModel
   BranchingModel
   LambdaModel

Model Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reusable neural network building blocks for constructing communication models.

.. currentmodule:: kaira.models.components

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AFModule

Binary Models
^^^^^^^^^^^^^^^^^^^^

Models specialized for binary data transmission.

.. currentmodule:: kaira.models.binary

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Kurmukova2025TransCoder

Image Models
^^^^^^^^^^^^^^^^^^^^

Models specialized for image transmission and reconstruction.

.. currentmodule:: kaira.models.image

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Tung2022DeepJSCCQEncoder
   Tung2022DeepJSCCQDecoder
   Tung2022DeepJSCCQ2Encoder
   Tung2022DeepJSCCQ2Decoder
   Yang2024DeepJSCCSwinEncoder
   Yang2024DeepJSCCSwinDecoder
   Yilmaz2023DeepJSCCNOMAModel

Image Compressors
'''''''''''''''''''''''''''''

Specialized image compression components.

.. currentmodule:: kaira.models.image.compressors

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BPGCompressor
   NeuralCompressor

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
   LossRegistry


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
   ModulationRegistry

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

Data
--------------

Datasets and data generation utilities for training and evaluation.

.. currentmodule:: kaira.data

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   WynerZivCorrelationModel
   WynerZivCorrelationDataset
   BinaryTensorDataset
   UniformTensorDataset

Data Generation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions for creating synthetic datasets.

.. currentmodule:: kaira.data

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   create_binary_tensor
   create_uniform_tensor

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

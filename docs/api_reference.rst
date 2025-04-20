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

Kaira is a modular toolkit for communication systems built on PyTorch. The library is organized into
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
   :recursive:
   :nosignatures:

   channels.BaseChannel
   constraints.BaseConstraint
   metrics.BaseMetric
   models.BaseModel
   modulations.BaseModulator
   modulations.BaseDemodulator
   losses.BaseLoss

Channels
--------

Channel models for communication systems.

.. currentmodule:: kaira.channels

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AWGNChannel
   BaseChannel
   BinaryErasureChannel
   BinarySymmetricChannel
   BinaryZChannel
   ChannelRegistry
   FlatFadingChannel
   GaussianChannel
   IdealChannel
   IdentityChannel
   LambdaChannel
   LaplacianChannel
   LogNormalFadingChannel
   NonlinearChannel
   PerfectChannel
   PhaseNoiseChannel
   PoissonChannel
   RayleighFadingChannel
   RicianFadingChannel


Constraints
-----------

Constraints module for Kaira.

.. currentmodule:: kaira.constraints

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AveragePowerConstraint
   BaseConstraint
   CompositeConstraint
   ConstraintRegistry
   IdentityConstraint
   LambdaConstraint
   PAPRConstraint
   PeakAmplitudeConstraint
   PerAntennaPowerConstraint
   SpectralMaskConstraint
   TotalPowerConstraint


Utils
^^^^^

Utility functions for constraints.

.. currentmodule:: kaira.constraints.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   apply_constraint_chain
   combine_constraints
   create_mimo_constraints
   create_ofdm_constraints
   measure_signal_properties
   verify_constraint


Metrics
-------

Metrics module for Kaira.

.. currentmodule:: kaira.metrics

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseMetric
   CompositeMetric
   MetricRegistry


Image
^^^^^

Image metrics module.

.. currentmodule:: kaira.metrics.image

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   LPIPS
   LearnedPerceptualImagePatchSimilarity
   MultiScaleSSIM
   PSNR
   PeakSignalNoiseRatio
   SSIM
   StructuralSimilarityIndexMeasure


Signal
^^^^^^

Signal metrics module.

.. currentmodule:: kaira.metrics.signal

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BER
   BLER
   BitErrorRate
   BlockErrorRate
   FER
   FrameErrorRate
   SER
   SNR
   SignalToNoiseRatio
   SymbolErrorRate


Models
------

Models module for Kaira.

.. currentmodule:: kaira.models

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseModel
   ChannelCodeModel
   ConfigurableModel
   DeepJSCCModel
   FeedbackChannelModel
   ModelRegistry
   MultipleAccessChannelModel
   WynerZivModel


Binary
^^^^^^

Binary data communication model implementations for Kaira.

.. currentmodule:: kaira.models.binary

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Kurmukova2025TransCoder


Soft Bit Thresholding
^^^^^^^^^^^^^^^^^^^^^

Soft bit thresholding module for binary data processing.

.. currentmodule:: kaira.models.binary.soft_bit_thresholding

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AdaptiveThresholder
   DynamicThresholder
   FixedThresholder
   HysteresisThresholder
   InputType
   LLRThresholder
   MinDistanceThresholder
   OutputType
   RepetitionSoftBitDecoder
   SoftBitEnsembleThresholder
   SoftBitThresholder
   WeightedThresholder


Components
^^^^^^^^^^

Components module for Kaira models.

.. currentmodule:: kaira.models.components

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AFModule
   ConvDecoder
   ConvEncoder
   MLPDecoder
   MLPEncoder


Decoders
^^^^^^^^

Forward Error Correction (FEC) decoders for Kaira.

.. currentmodule:: kaira.models.fec.decoders

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseBlockDecoder
   BerlekampMasseyDecoder
   BruteForceMLDecoder
   ReedMullerDecoder
   SyndromeLookupDecoder
   WagnerSoftDecisionDecoder


Encoders
^^^^^^^^

Forward Error Correction encoders for Kaira.

.. currentmodule:: kaira.models.fec.encoders

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BCHCodeEncoder
   BaseBlockCodeEncoder
   CyclicCodeEncoder
   GolayCodeEncoder
   HammingCodeEncoder
   LinearBlockCodeEncoder
   ReedSolomonCodeEncoder
   RepetitionCodeEncoder
   SingleParityCheckCodeEncoder
   SystematicLinearBlockCodeEncoder


Generic
^^^^^^^

Generic model implementations for Kaira.

.. currentmodule:: kaira.models.generic

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BranchingModel
   IdentityModel
   LambdaModel
   ParallelModel
   SequentialModel


Image
^^^^^

Image model implementations for Kaira.

.. currentmodule:: kaira.models.image

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Bourtsoulatze2019DeepJSCCDecoder
   Bourtsoulatze2019DeepJSCCEncoder
   DeepJSCCFeedbackDecoder
   DeepJSCCFeedbackEncoder
   DeepJSCCFeedbackModel
   Tung2022DeepJSCCQ2Decoder
   Tung2022DeepJSCCQ2Encoder
   Tung2022DeepJSCCQDecoder
   Tung2022DeepJSCCQEncoder
   Yilmaz2023DeepJSCCNOMADecoder
   Yilmaz2023DeepJSCCNOMAEncoder
   Yilmaz2023DeepJSCCNOMAModel
   Yilmaz2024DeepJSCCWZConditionalDecoder
   Yilmaz2024DeepJSCCWZConditionalEncoder
   Yilmaz2024DeepJSCCWZDecoder
   Yilmaz2024DeepJSCCWZEncoder
   Yilmaz2024DeepJSCCWZModel
   Yilmaz2024DeepJSCCWZSmallDecoder
   Yilmaz2024DeepJSCCWZSmallEncoder


Compressors
^^^^^^^^^^^

Image compressor models, including standard and neural network-based methods.

.. currentmodule:: kaira.models.image.compressors

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BPGCompressor
   NeuralCompressor


Modulations
-----------

Digital modulation schemes for wireless communications.

.. currentmodule:: kaira.modulations

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BPSKDemodulator
   BPSKModulator
   BaseDemodulator
   BaseModulator
   DBPSKDemodulator
   DBPSKModulator
   DPSKDemodulator
   DPSKModulator
   DQPSKDemodulator
   DQPSKModulator
   IdentityDemodulator
   IdentityModulator
   ModulationRegistry
   OQPSKDemodulator
   OQPSKModulator
   PAMDemodulator
   PAMModulator
   PSKDemodulator
   PSKModulator
   Pi4QPSKDemodulator
   Pi4QPSKModulator
   QAMDemodulator
   QAMModulator
   QPSKDemodulator
   QPSKModulator


.. currentmodule:: kaira.modulations

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   binary_array_to_gray
   binary_to_gray
   calculate_spectral_efficiency
   calculate_theoretical_ber
   gray_array_to_binary
   gray_to_binary
   plot_constellation


Utils
^^^^^

Utility functions for digital modulation schemes.

.. currentmodule:: kaira.modulations.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   binary_array_to_gray
   binary_to_gray
   calculate_spectral_efficiency
   calculate_theoretical_ber
   gray_array_to_binary
   gray_to_binary
   plot_constellation


Data
----

Data utilities for Kaira, including data generation and correlation models.

.. currentmodule:: kaira.data

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BinaryTensorDataset
   UniformTensorDataset
   WynerZivCorrelationDataset


.. currentmodule:: kaira.data

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   create_binary_tensor
   create_uniform_tensor
   load_sample_images

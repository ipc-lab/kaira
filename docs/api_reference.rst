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

This package provides various channel models for simulating communication systems, including analog
and digital channels, with support for various noise models, distortions, and fading patterns.

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
   UplinkMACChannel


Constraints
-----------

Constraints module for Kaira.

This module contains various constraints that can be applied to transmitted signals in
wireless communication systems. These constraints ensure signals meet practical requirements
such as power limitations, hardware capabilities, and regulatory specifications.

Available constraint categories:
- Base constraint definitions: Abstract base classes for all constraints
- Power constraints: Control total power, average power, and PAPR
- Antenna constraints: Manage power distribution across multiple antennas
- Signal constraints: Handle amplitude limitations and spectral properties
- Constraint composition: Combine multiple constraints sequentially

The module also provides factory functions for creating common constraint combinations
and utilities for testing and validating constraint effectiveness.

Example:
    >>> from kaira.constraints import TotalPowerConstraint, PAPRConstraint
    >>> from kaira.constraints.utils import combine_constraints
    >>>
    >>> # Create individual constraints
    >>> power_constr = TotalPowerConstraint(total_power=1.0)
    >>> papr_constr = PAPRConstraint(max_papr=4.0)
    >>>
    >>> # Combine constraints into a single operation
    >>> combined = combine_constraints([power_constr, papr_constr])
    >>>
    >>> # Apply to a signal
    >>> constrained_signal = combined(input_signal)

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

This module provides helper functions for creating, testing, validating, and working with
constraints in wireless communication systems. These utilities streamline the process of
configuring common constraint combinations and verifying constraint effectiveness.

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

This module contains various metrics for evaluating the performance of communication systems.

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

This module contains metrics for evaluating image quality.

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

This module contains metrics for evaluating signal processing performance.

.. currentmodule:: kaira.metrics.signal

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BER
   BLER
   BitErrorRate
   BlockErrorRate
   EVM
   ErrorVectorMagnitude
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


Soft Bit Thresholding
^^^^^^^^^^^^^^^^^^^^^

Soft bit thresholding module for binary data processing.

This module provides various thresholding techniques for converting soft bit representations
(probabilities, LLRs, etc.) to hard decisions. These thresholders can be used with soft decoders or
as standalone components in signal processing pipelines.

Soft bit processing is crucial in modern communication systems to extract maximum information from
the received signals. The techniques implemented here are based on established methods in
communication theory.

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
   Projection
   ProjectionType


Decoders
^^^^^^^^

Forward Error Correction (FEC) decoders for Kaira.

This module provides various decoder implementations for forward error correction codes.
The decoders in this module are designed to work seamlessly with the corresponding encoders
from the `kaira.models.fec.encoders` module.

Decoders
--------
- BlockDecoder: Base class for all block code decoders
- SyndromeLookupDecoder: Decoder using syndrome lookup tables for efficient error correction
- BerlekampMasseyDecoder: Implementation of Berlekamp-Massey algorithm for decoding BCH and Reed-Solomon codes
- ReedMullerDecoder: Implementation of Reed-Muller decoding algorithm for Reed-Muller codes
- WagnerSoftDecisionDecoder: Implementation of Wagner's soft-decision decoder for single-parity check codes
- BruteForceMLDecoder: Maximum likelihood decoder that searches through all possible codewords
- BeliefPropagationDecoder: Implementation of belief propagation algorithm for decoding LDPC codes

These decoders can be used to recover original messages from possibly corrupted codewords
that have been transmitted over noisy channels. Each decoder has specific strengths and
is optimized for particular types of codes or error patterns.

Examples
--------
>>> from kaira.models.fec.encoders import BCHCodeEncoder
>>> from kaira.models.fec.decoders import BerlekampMasseyDecoder
>>> encoder = BCHCodeEncoder(15, 7)
>>> decoder = BerlekampMasseyDecoder(encoder)
>>> # Example decoding
>>> received = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
>>> decoded = decoder(received)

.. currentmodule:: kaira.models.fec.decoders

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseBlockDecoder
   BeliefPropagationDecoder
   BeliefPropagationPolarDecoder
   BerlekampMasseyDecoder
   BruteForceMLDecoder
   ReedMullerDecoder
   SuccessiveCancellationDecoder
   SyndromeLookupDecoder
   WagnerSoftDecisionDecoder


Encoders
^^^^^^^^

Forward Error Correction encoders for Kaira.

This module provides various encoder implementations for forward error correction, including:
- Block codes: Fundamental error correction codes that operate on fixed-size blocks
- Linear block codes: Codes with linear algebraic structure allowing matrix operations
- LDPC codes: Low-Density Parity-Check codes with sparse parity-check matrices
- Cyclic codes: Special class of linear codes with cyclic shift properties
- BCH codes: Powerful algebraic codes with precise error-correction capabilities
- Reed-Solomon codes: Widely-used subset of BCH codes for burst error correction
- Hamming codes: Simple single-error-correcting codes with efficient implementation
- Repetition codes: Basic codes that repeat each bit multiple times
- Golay codes: Perfect codes with specific error correction properties
- Single parity-check codes: Simple error detection through parity bit addition

These encoders can be used to add redundancy to data for enabling error detection and correction
in communication systems, storage devices, and other applications requiring reliable data
transmission over noisy channels :cite:`lin2004error,moon2005error`.

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
   LDPCCodeEncoder
   LinearBlockCodeEncoder
   PolarCodeEncoder
   ReedMullerCodeEncoder
   ReedSolomonCodeEncoder
   RepetitionCodeEncoder
   SingleParityCheckCodeEncoder
   SystematicLinearBlockCodeEncoder


Generic
^^^^^^^

Generic model implementations for Kaira.

This module provides generic model implementations that can be used as building blocks for more
complex models, such as sequential, parallel, and branching models.

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

This module provides models specifically designed for image data transmission.

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
   Xie2023DTDeepJSCCDecoder
   Xie2023DTDeepJSCCEncoder
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

This package provides implementations of common digital modulation and demodulation techniques used
in modern communication systems, including PSK, QAM, PAM, and differential modulation schemes.

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


Losses
------

Kaira Losses Package.

This package provides various loss functions for different modalities.

.. currentmodule:: kaira.losses

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseLoss
   CompositeLoss
   LossRegistry


Adversarial
^^^^^^^^^^^

Adversarial Losses module for Kaira.

This module contains various adversarial loss functions for GAN-based training.

.. currentmodule:: kaira.losses.adversarial

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   FeatureMatchingLoss
   HingeLoss
   LSGANLoss
   R1GradientPenalty
   VanillaGANLoss
   WassersteinGANLoss


Audio
^^^^^

Audio Losses module for Kaira.

This module contains various loss functions for training audio-based communication systems.

.. currentmodule:: kaira.losses.audio

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AudioContrastiveLoss
   FeatureMatchingLoss
   L1AudioLoss
   LogSTFTMagnitudeLoss
   MelSpectrogramLoss
   MultiResolutionSTFTLoss
   STFTLoss
   SpectralConvergenceLoss


Image
^^^^^

Losses module for Kaira.

This module contains various loss functions for training communication systems, including MSE loss,
LPIPS loss, and SSIM loss. These loss functions are widely used in image processing and
computer vision tasks :cite:`wang2009mean` :cite:`zhang2018unreasonable`.

.. currentmodule:: kaira.losses.image

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   CombinedLoss
   FocalLoss
   GradientLoss
   L1Loss
   LPIPSLoss
   MSELPIPSLoss
   MSELoss
   MSSSIMLoss
   PSNRLoss
   SSIMLoss
   StyleLoss
   TotalVariationLoss
   VGGLoss


Multimodal
^^^^^^^^^^

Multimodal Losses module for Kaira.

This module contains various loss functions for training multimodal systems.

.. currentmodule:: kaira.losses.multimodal

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AlignmentLoss
   CMCLoss
   ContrastiveLoss
   InfoNCELoss
   TripletLoss


Text
^^^^

Text Losses module for Kaira.

This module contains various loss functions for training text-based systems.

.. currentmodule:: kaira.losses.text

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   CosineSimilarityLoss
   CrossEntropyLoss
   LabelSmoothingLoss
   Word2VecLoss


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


Utils
-----

General utility functions for the Kaira library.

.. currentmodule:: kaira.utils

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   CapacityAnalyzer


.. currentmodule:: kaira.utils

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   add_noise_for_snr
   calculate_num_filters_factor_image
   calculate_snr
   estimate_signal_power
   noise_power_to_snr
   snr_db_to_linear
   snr_linear_to_db
   snr_to_noise_power
   to_tensor


Snr
^^^

Utility functions for Signal-to-Noise Ratio (SNR) calculations and conversions.

.. currentmodule:: kaira.utils.snr

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   add_noise_for_snr
   calculate_snr
   estimate_signal_power
   noise_power_to_snr
   snr_db_to_linear
   snr_linear_to_db
   snr_to_noise_power


Benchmarks
----------

Kaira Benchmarking System.

This module provides standardized benchmarks for evaluating communication system components and
deep learning models in Kaira.

.. currentmodule:: kaira.benchmarks

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseBenchmark
   BenchmarkConfig
   BenchmarkRegistry
   BenchmarkResult
   BenchmarkResultsManager
   BenchmarkSuite
   BenchmarkVisualizer
   ComparisonRunner
   ParallelRunner
   ParametricRunner
   StandardMetrics
   StandardRunner


.. currentmodule:: kaira.benchmarks

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   create_benchmark
   get_benchmark
   get_config
   list_benchmarks
   list_configs
   register_benchmark

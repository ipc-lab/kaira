Changelog
=========

All notable changes to the Kaira project are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.




Unreleased Changes
------------------


Added
^^^^^


* Future features and improvements will be listed here

Version 0.2.0 (2025-06-07)
--------------------------


Added
^^^^^


* **Forward Error Correction (FEC) Support**: Complete implementation of LDPC and Polar codes
* LDPC Code Encoder with configurable parameters and RPTU database integration
* Belief Propagation Decoder for LDPC codes with enhanced error handling
* Polar Code Encoder with systematic and non-systematic encoding options
* Successive Cancellation Decoder for Polar codes
* Advanced visualization tools for Polar code decoding animations
* Comprehensive unit tests for all FEC components
* **Enhanced Metrics System**
* Error Vector Magnitude (EVM) metric with corresponding tests
* Improved MultiScaleSSIM implementation using torchmetrics
* Enhanced benchmark configurations with block_length parameter support
* **Channel-Aware Models**
* Introduced ChannelAwareBaseModel for standardized Channel State Information (CSI) handling
* Improved neural compressor with better error handling for mock models
* **Documentation and Examples**
* Comprehensive FEC examples demonstrating LDPC and Polar code usage
* Enhanced API reference documentation for decoders and encoders
* Added example code sections with visual demonstrations
* Improved theoretical analysis documentation with better formatting

Changed
^^^^^^^


* **Modulation Improvements**
* Fixed complex_output parameter behavior across PSK modulators
* Improved demodulation logic for better performance
* **Benchmark System Enhancements**
* Refactored benchmark configurations to use standardized parameters
* Improved result filtering in BenchmarkResultsManager
* Enhanced benchmark examples with clearer documentation
* **Code Quality and Testing**
* Comprehensive refactoring of FEC encoder modules for better readability
* Enhanced pre-commit workflows and CI/CD pipeline improvements
* Expanded test coverage for neural networks and benchmark components
* Improved type hints and error messages across modules

Fixed
^^^^^


* Corrected assertion messages in decoders to reference appropriate parameters
* Fixed import paths for polar code indices
* Resolved parameter naming inconsistencies in LDPC experiments
* Improved error handling in neural compressor forward pass
* Fixed GitHub Actions workflow configurations for API documentation

Removed
^^^^^^^


* Cleaned up outdated GitHub Actions workflows
* Removed redundant build artifacts and temporary files

Version 0.1.1 (2025-05-22)
--------------------------


Changed
^^^^^^^


* Updated Python version requirement to >=3.10 in setup configuration
* Fixed installation command in documentation to use 'pykaira' instead of 'kaira'

Removed
^^^^^^^


* Removed CircleCI configuration, release drafter settings, and changelog update script
* Cleaned up build artifacts and enhanced deployment script
* **Note**: Changelog update script removal is intentional - changelog is now maintained manually for better control over release notes

Improved
^^^^^^^^


* Enhanced project metadata and version extraction
* Refactored deployment script for better reliability

Version 0.1.0 (2025-05-09)
--------------------------


Added
^^^^^


* Initial release of Kaira framework v0.1.0
* Core modules for wireless communication simulation
* DeepJSCC implementation
* Channel models and modulation schemes
* Metrics for performance evaluation
* Documentation framework
* CI/CD pipeline integration

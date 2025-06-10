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

Version 0.2.0 (2025-06-10)
--------------------------


Added
^^^^^


* **Forward Error Correction (FEC) Framework**: Comprehensive error correction implementation
* **LDPC Codes**: Implementation with RPTU database integration
  * Belief Propagation Decoder with configurable iteration count
  * Min-Sum LDPC Decoder with scaling factor and offset parameters
  * LDPC benchmarking suite with multiple code configurations
  * Educational and practical LDPC code examples
* **Polar Codes**: Implementation following 5G standard specifications
  * Systematic and non-systematic Polar Code Encoders with rank-based frozen bit selection
  * Successive Cancellation Decoder for polar code decoding
  * Belief Propagation Polar Decoder with message passing algorithm
  * Polar code visualization with decoding step demonstrations
* **Reed-Muller Codes**: Algebraic block code implementation with decoder
* **Algebraic Codes**: BCH, Reed-Solomon, and Hamming code implementations
* **Decoders**: Berlekamp-Massey, Brute Force ML, and Wagner soft-decision decoders
* **Channel Models and Communication Systems**
* Uplink MAC Channel implementation for multiple access scenarios
* AWGN channel improvements with better noise modeling
* Enhanced integration between channels, modulators, and FEC components
* **Neural Network Components**
* BPG Compressor integration for image compression applications
* Neural compressor with fixed quality modes and early stopping
* DeepJSCC implementations with channel awareness
* ChannelAwareBaseModel for Channel State Information (CSI) handling
* **Metrics and Benchmarking**
* Error Vector Magnitude (EVM) metric implementation
* MultiScaleSSIM implementation using torchmetrics
* Benchmark configurations with variable block length support
* LDPC benchmarking suite with decoder performance comparison
* **Documentation and Examples**
* FEC tutorial examples covering encoder and decoder usage
* Visualization examples for decoding algorithms
* API reference documentation for FEC components
* Performance comparison guides and educational materials

Changed
^^^^^^^


* **Development Infrastructure**
* Python 3.13.2 support added across codebase
* GitHub Actions workflows updated with artifact handling and token management
* Auto-examples generation system with incremental builds
* Documentation pipeline with Read the Docs integration
* MyPy type checking with type annotations
* **FEC Module Architecture**
* Encoder module refactoring for better maintainability
* Decoder base classes with improved error handling
* Belief propagation algorithm memory optimization
* Code structure reorganization for educational vs. practical use
* **Modulation and Signal Processing**
* PSK modulator complex_output parameter fixes
* Demodulation logic with better noise variance handling
* LLR (Log-Likelihood Ratio) calculation improvements
* Integration improvements between modulators, channels, and decoders
* **Benchmark System**
* Benchmark configurations using standardized parameter formats
* Result filtering and analysis in BenchmarkResultsManager
* Performance analysis examples with detailed metrics
* Statistical reliability improvements with configurable parameters
* **Testing Framework**
* Mock tests replaced with integration tests for external dependencies
* Neural compressor testing with realistic scenarios
* FEC component test coverage expansion
* Test documentation and contributor guidelines

Fixed
^^^^^


* **Code Quality**
* MyPy type annotations for JSON response handling in GitHub API integration
* GitHub Actions workflow conditional logic and artifact access permissions
* Assertion messages in decoders to reference correct parameter names
* Error handling in neural compressor forward pass
* **FEC Implementation**
* Import paths for polar code indices and rank-based selection
* Parameter naming inconsistencies in LDPC experiments and examples
* Belief propagation convergence detection and message passing
* Systematic encoding behavior in FEC encoders
* **Documentation and Build System**
* GitHub Actions workflow configurations for API documentation generation
* Auto_examples release upload logic for existing releases
* Extraction process for problematic files during documentation builds
* Error handling in download_and_extract_examples functionality
* **Testing Framework**
* Integration test configurations for BPG compressor and external dependencies
* Test isolation issues in neural network benchmarks
* Test stability with random seed management
* Performance metric calculations in benchmark test suites

Performance Improvements
^^^^^^^^^^^^^^^^^^^^^^^^


* **FEC Decoding**
* Belief propagation memory usage optimization for large LDPC codes
* Polar code decoding speed with vectorized operations
* Convergence detection reducing iteration overhead
* Sparse matrix handling in parity check operations
* **Neural Networks**
* BPG compressor integration with optimized quality settings
* DeepJSCC forward pass efficiency with tensor management
* Channel-aware model implementations for faster inference
* **Benchmarks**
* Benchmark execution with parallel processing
* Statistical analysis with faster BER/BLER calculations
* Memory management in large-scale simulation runs

Removed
^^^^^^^


* **Legacy Code**
* Outdated GitHub Actions workflows replaced by updated CI/CD pipeline
* Redundant build artifacts and temporary files
* Deprecated API endpoints and legacy benchmark configurations
* ePub format option from documentation build
* **Development Workflow**
* Conflict resolution complexity in auto_examples extraction process
* Download step enforcement in favor of full rebuild strategy
* Problematic file handling in auto_examples generation pipeline

Breaking Changes
^^^^^^^^^^^^^^^^


* **Python Version**: Minimum Python version increased to 3.10
* **FEC API**: Decoder initialization parameter changes
* `bp_iterations` renamed to `bp_iters` in belief propagation decoders
* Parameter validation raises specific exceptions for invalid configurations
* **Benchmark Configuration**: Legacy parameter formats deprecated
* **Import Paths**: Internal FEC utility imports reorganized

Migration Guide
^^^^^^^^^^^^^^^


For users upgrading from v0.1.x to v0.2.0:

1. Update Python environment to ≥3.10
2. Update FEC decoder configurations to use new parameter names
3. Update benchmark configurations to use standardized format
4. Review imports from `kaira.models.fec.utils` for changes

Security Improvements
^^^^^^^^^^^^^^^^^^^^^


* GitHub token management for CI/CD workflows
* Dependency updates to latest secure versions
* Parameter validation across FEC components

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

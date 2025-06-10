:orphan:

Benchmarks
==========

Benchmarking tools and performance comparisons for different algorithms, models, and system configurations.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the basic usage of the Kaira benchmarking system, including running individual benchmarks, creating and running benchmark suites, and saving/analyzing results. The Kaira benchmarking system provides tools for: * Running individual benchmarks with different configurations * Creating and executing benchmark suites * Analyzing and visualizing benchmark results * Comparing performance across different algorithms and parameters">

.. only:: html

    .. image:: /auto_examples/benchmarks/images/thumb/sphx_glr_plot_basic_usage_thumb.png
      :alt: Basic Benchmark Usage

    :ref:`sphx_glr_auto_examples_benchmarks_plot_basic_usage.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Basic Benchmark Usage</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the Kaira benchmarking system to compare the performance of different approaches, such as various modulation schemes, using parameter sweeps and result visualization. The comparison framework allows you to: * Compare multiple algorithms or configurations side-by-side * Run parameter sweeps to explore performance across different settings * Visualize comparative results with unified plotting * Generate comprehensive performance summaries">

.. only:: html

    .. image:: /auto_examples/benchmarks/images/thumb/sphx_glr_plot_comparison_example_thumb.png
      :alt: Benchmark Comparison Example

    :ref:`sphx_glr_auto_examples_benchmarks_plot_comparison_example.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Benchmark Comparison Example</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the new organized results management system in Kaira, showcasing automatic directory structuring, experiment naming, suite management, result comparison, and maintenance features. The results management system provides: * Automatic directory organization for benchmark results * Experiment naming and metadata tracking * Suite-level result aggregation and comparison * Result maintenance and cleanup utilities * Comprehensive result analysis and reporting">

.. only:: html

    .. image:: /auto_examples/benchmarks/images/thumb/sphx_glr_plot_demo_new_results_system_thumb.png
      :alt: New Results Management System Demo

    :ref:`sphx_glr_auto_examples_benchmarks_plot_demo_new_results_system.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">New Results Management System Demo</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates a comprehensive benchmark for Forward Error Correction (FEC) codes using the Kaira benchmarking system. It evaluates multiple ECC algorithms across different parameters and provides detailed performance comparison. The comprehensive ECC benchmark includes: * Multiple error correction codes (Hamming, BCH, Golay, Repetition, Single Parity Check) * Block Error Rate (BLER) and Bit Error Rate (BER) evaluation * Coding gain analysis * Error correction capability evaluation * Comparison across different code rates and block lengths Note: Individual benchmarks use only repetition codes (the only code type currently supported by ChannelCodingBenchmark), while the comprehensive benchmark tests all available ECC implementations directly using the FEC encoder/decoder classes.">

.. only:: html

    .. image:: /auto_examples/benchmarks/images/thumb/sphx_glr_plot_ecc_comprehensive_benchmark_thumb.png
      :alt: Comprehensive Error Correction Codes Benchmark

    :ref:`sphx_glr_auto_examples_benchmarks_plot_ecc_comprehensive_benchmark.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comprehensive Error Correction Codes Benchmark</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This benchmark compares different LDPC (Low-Density Parity-Check) codes :cite:`gallager1962low` across various metrics including: - Bit Error Rate (BER) performance - Block Error Rate (BLER) performance - Decoding convergence behavior with belief propagation :cite:`kschischang2001factor` - Computational complexity - Code rate efficiency We test multiple LDPC code configurations with different: - Parity check matrix structures - Code rates - Block lengths - Belief propagation iteration counts">

.. only:: html

    .. image:: /auto_examples/benchmarks/images/thumb/sphx_glr_plot_ldpc_codes_comparison_thumb.png
      :alt: LDPC Codes Comparison Benchmark

    :ref:`sphx_glr_auto_examples_benchmarks_plot_ldpc_codes_comparison.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">LDPC Codes Comparison Benchmark</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates comprehensive benchmark result visualization in Kaira, including BER curve plotting, throughput performance, modulation comparisons, and performance summary generation. The visualization system provides: * BER curve plotting with theoretical and simulated results * Throughput performance analysis across different payload sizes * Comparative visualization of multiple algorithms or configurations * Automated report generation with statistical summaries * Customizable plotting styles and formats">

.. only:: html

    .. image:: /auto_examples/benchmarks/images/thumb/sphx_glr_plot_visualization_example_thumb.png
      :alt: Benchmark Visualization Example

    :ref:`sphx_glr_auto_examples_benchmarks_plot_visualization_example.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Benchmark Visualization Example</div>
    </div>

.. raw:: html

    </div>


.. toctree:
   :hidden:

   /auto_examples/benchmarks/plot_basic_usage
   /auto_examples/benchmarks/plot_comparison_example
   /auto_examples/benchmarks/plot_demo_new_results_system
   /auto_examples/benchmarks/plot_ecc_comprehensive_benchmark
   /auto_examples/benchmarks/plot_ldpc_codes_comparison
   /auto_examples/benchmarks/plot_visualization_example

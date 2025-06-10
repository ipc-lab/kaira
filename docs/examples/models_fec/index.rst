:orphan:

Models Fec
==========

Forward Error Correction (FEC) models and coding techniques, including modern deep learning approaches to error correction and classical coding schemes.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the fundamental binary operations used in forward error correction (FEC) coding. We'll explore how to calculate Hamming distances, Hamming weights, and convert between binary and integer representations using utility functions from the FEC module.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_binary_operations_thumb.png
      :alt: Basic Binary Operations for FEC

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_binary_operations.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Basic Binary Operations for FEC</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to perform block-wise processing of data for forward error correction (FEC) using the `apply_blockwise` utility function. Block-wise processing is essential in many coding schemes like block codes, systematic codes, and interleaved coding.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_blockwise_processing_thumb.png
      :alt: Block-wise Processing for FEC

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_blockwise_processing.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Block-wise Processing for FEC</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use various Forward Error Correction (FEC) decoders from the kaira.models.fec.decoders module. FEC decoders recover original messages from possibly corrupted codewords that have been transmitted over noisy channels. We'll explore: - Basic concepts in FEC decoding - Hard-decision vs. soft-decision decoding - Syndrome-based decoding - Advanced algebraic decoders - Maximum likelihood decoding - Performance evaluation and error correction capabilities">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_decoders_tutorial_thumb.png
      :alt: FEC Decoders Tutorial

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_decoders_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FEC Decoders Tutorial</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example provides a detailed visualization of how forward error correction (FEC) decoding algorithms work. We'll create animated, step-by-step visualizations of several popular decoding algorithms to help understand their inner workings and relative performance characteristics.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_decoding_visualization_thumb.png
      :alt: Visualizing the FEC Decoding Process

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_decoding_visualization.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualizing the FEC Decoding Process</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use various Forward Error Correction (FEC) encoders from the kaira.models.fec.encoders module. FEC codes add redundancy to transmitted data, allowing receivers to detect and correct errors without retransmission. We'll explore: - Basic block codes (Repetition, Single Parity Check) - Linear block codes (Hamming) - Cyclic codes and BCH codes - Reed-Solomon codes - Advanced features and performance evaluation">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_encoders_tutorial_thumb.png
      :alt: FEC Encoders Tutorial

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_encoders_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FEC Encoders Tutorial</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the algebraic foundations of forward error correction (FEC) codes using the binary polynomial and finite field utilities in the `kaira.models.fec.algebra` module. These algebraic structures are essential for advanced codes like BCH, Reed-Solomon, and other algebraic error correction codes.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_finite_field_algebra_thumb.png
      :alt: Finite Field Algebra for FEC Codes

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_finite_field_algebra.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Finite Field Algebra for FEC Codes</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates advanced visualizations for Low-Density Parity-Check (LDPC) codes :cite:`gallager1962low`, including animated belief propagation :cite:`kschischang2001factor`, Tanner graph analysis, and performance comparisons with different decoder configurations.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_ldpc_advanced_visualization_thumb.png
      :alt: Advanced LDPC Code Visualization with Belief Propagation Animation

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_ldpc_advanced_visualization.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Advanced LDPC Code Visualization with Belief Propagation Animation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates Low-Density Parity-Check (LDPC) codes :cite:`gallager1962low` (via RPTU database) and belief propagation decoding :cite:`kschischang2001factor`. We'll simulate a complete communication system using LDPC codes over an AWGN channel and analyze the error performance at different SNR levels.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_ldpc_rptu_simulation_thumb.png
      :alt: LDPC Coding and Belief Propagation Decoding via RPTU Database

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_ldpc_rptu_simulation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">LDPC Coding and Belief Propagation Decoding via RPTU Database</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates Low-Density Parity-Check (LDPC) codes and belief propagation decoding :cite:`gallager1962low` :cite:`kschischang2001factor`. We'll simulate a complete communication system using LDPC codes over an AWGN channel and analyze the error performance at different SNR levels.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_ldpc_simulation_thumb.png
      :alt: LDPC Coding and Belief Propagation Decoding

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_ldpc_simulation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">LDPC Coding and Belief Propagation Decoding</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates advanced visualizations for Polar codes :cite:`arikan2008channel`, including channel polarization visualization, successive cancellation decoding :cite:`arikan2009channel` steps, and performance comparisons between different decoders including belief propagation :cite:`arikan2011systematic`.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_polar_advanced_visualization_thumb.png
      :alt: Advanced Polar Code Visualization with Decoding Animations

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_polar_advanced_visualization.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Advanced Polar Code Visualization with Decoding Animations</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates Polar codes :cite:`arikan2008channel` with successive cancellation :cite:`arikan2009channel` and belief propagation decoding :cite:`arikan2011systematic`. We'll simulate a complete communication system using Polar codes over an AWGN channel and analyze the error performance at different SNR levels.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_polar_simulation_thumb.png
      :alt: Polar Coding and Decoding: Successive Cancellation and Belief Propagation

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_polar_simulation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar Coding and Decoding: Successive Cancellation and Belief Propagation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example showcases various forward error correction (FEC) coding schemes in real-world applications through interactive, visually rich demonstrations. We'll explore how different codes perform under various channel conditions and application constraints, with a focus on the practical tradeoffs between redundancy, error correction capability, and computational complexity.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_practical_applications_thumb.png
      :alt: Interactive FEC Code Comparison for Real-World Applications

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_practical_applications.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Interactive FEC Code Comparison for Real-World Applications</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates syndrome decoding, a key technique in forward error correction (FEC) that efficiently corrects errors using a parity-check matrix. We'll visualize the syndrome computation and the error correction process with animated, interactive graphics.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_syndrome_decoding_thumb.png
      :alt: Syndrome Decoding Visualization

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_syndrome_decoding.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Syndrome Decoding Visualization</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example provides a visually rich demonstration of forward error correction (FEC) techniques in action. We'll visualize how different coding schemes protect data as it passes through noisy channels, with animated and interactive visualizations that show the encoding, transmission, corruption, and decoding processes.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_fec_visual_error_correction_thumb.png
      :alt: Visualizing Error Correction in Action

    :ref:`sphx_glr_auto_examples_models_fec_plot_fec_visual_error_correction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualizing Error Correction in Action</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the fundamental differences between educational hand-crafted LDPC codes :cite:`gallager1962low` and professional RPTU database codes, explaining why direct performance comparison is inappropriate and misleading. We focus on: - Educational value of small hand-crafted codes - Real-world applications of RPTU database codes - Why these serve different purposes - Proper evaluation methodologies for each type **Key Message**: These code types serve different purposes and should not be directly compared for "performance" - it's like comparing a bicycle to an airplane for transportation efficiency.">

.. only:: html

    .. image:: /auto_examples/models_fec/images/thumb/sphx_glr_plot_ldpc_educational_vs_professional_thumb.png
      :alt: Educational vs Professional LDPC Codes: Understanding the Differences

    :ref:`sphx_glr_auto_examples_models_fec_plot_ldpc_educational_vs_professional.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Educational vs Professional LDPC Codes: Understanding the Differences</div>
    </div>

.. raw:: html

    </div>


.. toctree:
   :hidden:

   /auto_examples/models_fec/plot_fec_binary_operations
   /auto_examples/models_fec/plot_fec_blockwise_processing
   /auto_examples/models_fec/plot_fec_decoders_tutorial
   /auto_examples/models_fec/plot_fec_decoding_visualization
   /auto_examples/models_fec/plot_fec_encoders_tutorial
   /auto_examples/models_fec/plot_fec_finite_field_algebra
   /auto_examples/models_fec/plot_fec_ldpc_advanced_visualization
   /auto_examples/models_fec/plot_fec_ldpc_rptu_simulation
   /auto_examples/models_fec/plot_fec_ldpc_simulation
   /auto_examples/models_fec/plot_fec_polar_advanced_visualization
   /auto_examples/models_fec/plot_fec_polar_simulation
   /auto_examples/models_fec/plot_fec_practical_applications
   /auto_examples/models_fec/plot_fec_syndrome_decoding
   /auto_examples/models_fec/plot_fec_visual_error_correction
   /auto_examples/models_fec/plot_ldpc_educational_vs_professional

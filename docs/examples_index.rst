:orphan:

.. _kaira_examples_gallery:

Examples and Tutorials
======================

.. raw:: html

    <div class="gallery-intro">
        <p>This gallery provides a comprehensive collection of examples demonstrating Kaira's capabilities for communications, signal processing, and machine learning applications. Each example includes:</p>
        <ul>
            <li>Well-documented code that you can run directly</li>
            <li>Clear explanations of the underlying concepts</li>
            <li>Visualizations of results and performance metrics</li>
            <li>Tips for adapting the examples to your specific use cases</li>
        </ul>
        
        <h2>Getting Started</h2>
        <ul>
            <li><strong>New to Kaira?</strong> Start with the basic examples to understand core concepts</li>
            <li><strong>Seeking specific features?</strong> Use the categories below to find relevant examples</li>
            <li><strong>Want to learn more?</strong> Each example links to related documentation topics</li>
        </ul>
    </div>

.. raw:: html

    <div class="category-grid">
        <div class="category-card">
            <a href="examples/channel_models/index.html">
                <h3>Channel Models</h3>
                <p>Explore various communication channel implementations</p>
            </a>
        </div>
        
        <div class="category-card">
            <a href="examples/constraints/index.html">
                <h3>Constraints</h3>
                <p>Examples demonstrating various constraints for communications systems</p>
            </a>
        </div>
        
        <div class="category-card">
            <a href="examples/modulation/index.html">
                <h3>Modulation</h3>
                <p>Digital modulation schemes and constellation analysis</p>
            </a>
        </div>
        
        <div class="category-card">
            <a href="examples/deep_jscc/index.html">
                <h3>Deep JSCC</h3>
                <p>End-to-end deep learning for joint source-channel coding</p>
            </a>
        </div>
        
        <div class="category-card">
            <a href="examples/metrics/index.html">
                <h3>Metrics</h3>
                <p>Comprehensive evaluation metrics for signal quality, error rates, and perceptual quality, including PSNR, SSIM, BER, SNR, and LPIPS</p>
            </a>
        </div>
        
        <div class="category-card">
            <a href="examples/performance_analysis/index.html">
                <h3>Performance Analysis</h3>
                <p>Tools for measuring and optimizing system performance</p>
            </a>
        </div>
    </div>

.. raw:: html

    <div class="gallery-outro">
        <p>We encourage you to modify these examples for your own research and applications. If you develop an interesting example that might benefit others, please consider contributing it back to the community!</p>
        <p><em>Happy experimenting with Kaira!</em></p>
    </div>

.. toctree::
   :maxdepth: 2
   :hidden:
   
   examples/channel_models/index
   examples/constraints/index
   examples/modulation/index
   examples/deep_jscc/index
   examples/metrics/index
   examples/performance_analysis/index

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of Additive White Gaussian Noise (AWGN) channel in the Kaira library. AWGN is one of the most common communication channel models, which adds Gaussian noise to the input signal.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_awgn_channel_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_awgn_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Simulating AWGN Channels with Kaira</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the LaplacianChannel in Kaira, which models channels with impulsive noise that follows the Laplacian distribution. Unlike Gaussian noise, Laplacian noise has heavier tails, making it suitable for modeling environments with occasional large noise spikes.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_laplacian_channel_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_laplacian_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Impulsive Noise with Laplacian Channel</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of basic power constraints in Kaira. We&#x27;ll explore how to apply various constraints to signals and visualize their effects.">

.. only:: html

    .. image:: /auto_examples/constraints/images/thumb/sphx_glr_plot_basic_constraints_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_constraints_plot_basic_constraints.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Understanding Basic Power Constraints in Kaira</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the PhaseNoiseChannel in Kaira, which simulates phase noise commonly encountered in oscillators and frequency synthesizers. Phase noise is a critical impairment in high-frequency communication systems and can severely degrade performance even when signal amplitude remains intact.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_phase_noise_channel_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_phase_noise_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Phase Noise Effects on Signal Constellations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of binary channel models in Kaira. Binary channels are fundamental in digital communications as they represent the transmission of binary data (0s and 1s) through a noisy medium.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_binary_channels_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_binary_channels.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Digital Binary Channels in Kaira</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to combine multiple constraints in Kaira to satisfy complex signal requirements. We&#x27;ll explore the composition utilities and see how constraints can be sequentially applied to meet practical transmission specifications.">

.. only:: html

    .. image:: /auto_examples/constraints/images/thumb/sphx_glr_plot_constraint_composition_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_constraints_plot_constraint_composition.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Composing Constraints for Complex Signal Requirements</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to simulate and analyze fading channels using Kaira. Fading channels model signal attenuation and phase shifts that occur in wireless communications due to multipath propagation and other environmental factors.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_fading_channels_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_fading_channels.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fading Channels in Wireless Communications</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the NonlinearChannel in Kaira, which allows modeling various nonlinear signal distortions commonly encountered in communication systems. Nonlinearities occur in many components such as amplifiers, mixers, and converters, and can significantly impact system performance.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_nonlinear_channel_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_nonlinear_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear Channel Distortion Effects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to compose multiple channel effects in Kaira to simulate complex transmission scenarios. In real communication systems, signals often pass through multiple channel impairments simultaneously, such as fading, phase noise, and additive noise. Kaira makes it easy to chain these effects together for realistic simulations.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_composite_channels_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_composite_channels.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Composing Multiple Channel Effects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the PoissonChannel in Kaira, which models signal-dependent noise commonly found in optical systems and photon-counting detectors. Unlike AWGN where noise is independent of signal intensity, Poisson noise increases with signal strength, making it essential for accurate modeling of optical communications and imaging systems.">

.. only:: html

    .. image:: /auto_examples/channel_models/images/thumb/sphx_glr_plot_poisson_channel_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_channel_models_plot_poisson_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Poisson Channel for Signal-Dependent Noise</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates practical applications of Kaira&#x27;s constraints in realistic wireless communication scenarios, focusing on OFDM and MIMO systems. We&#x27;ll explore how to configure and apply appropriate constraints for these systems.">

.. only:: html

    .. image:: /auto_examples/constraints/images/thumb/sphx_glr_plot_practical_constraints_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_constraints_plot_practical_constraints.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Practical Applications of Constraints in Wireless Communication Systems</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/channel_models/plot_awgn_channel
   /auto_examples/channel_models/plot_laplacian_channel
   /auto_examples/constraints/plot_basic_constraints
   /auto_examples/channel_models/plot_phase_noise_channel
   /auto_examples/channel_models/plot_binary_channels
   /auto_examples/constraints/plot_constraint_composition
   /auto_examples/channel_models/plot_fading_channels
   /auto_examples/channel_models/plot_nonlinear_channel
   /auto_examples/channel_models/plot_composite_channels
   /auto_examples/channel_models/plot_poisson_channel
   /auto_examples/constraints/plot_practical_constraints

.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

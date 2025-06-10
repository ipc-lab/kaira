:orphan:

Channels
========

Channel models for wireless communications, including AWGN, fading channels, and composite channel effects.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of Additive White Gaussian Noise (AWGN) channel in the Kaira library. AWGN is one of the most common communication channel models, which adds Gaussian noise to the input signal. We'll visualize how different noise levels (SNR) affect signal transmission.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_awgn_channel_thumb.png
      :alt: Simulating AWGN Channels with Kaira

    :ref:`sphx_glr_auto_examples_channels_plot_awgn_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Simulating AWGN Channels with Kaira</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of binary channel models in Kaira. Binary channels are fundamental in digital communications as they represent the transmission of binary data (0s and 1s) through a noisy medium. We'll explore the three main binary channel models: 1. Binary Symmetric Channel (BSC) 2. Binary Erasure Channel (BEC) 3. Binary Z-Channel">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_binary_channels_thumb.png
      :alt: Digital Binary Channels in Kaira

    :ref:`sphx_glr_auto_examples_channels_plot_binary_channels.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Digital Binary Channels in Kaira</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to create a visually appealing comparison of different communication channels in Kaira. We'll visualize the effects of various channels on transmitted signals and compare their characteristics.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_channel_comparison_thumb.png
      :alt: Channel Comparison

    :ref:`sphx_glr_auto_examples_channels_plot_channel_comparison.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Channel Comparison</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to compose multiple channel effects in Kaira to simulate complex transmission scenarios. In real communication systems, signals often pass through multiple channel impairments simultaneously, such as fading, phase noise, and additive noise. Kaira makes it easy to chain these effects together for realistic simulations.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_composite_channels_thumb.png
      :alt: Composing Multiple Channel Effects

    :ref:`sphx_glr_auto_examples_channels_plot_composite_channels.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Composing Multiple Channel Effects</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to simulate and analyze fading channels using Kaira. Fading channels model signal attenuation and phase shifts that occur in wireless communications due to multipath propagation and other environmental factors. In this example, we'll focus on the FlatFadingChannel model, which simulates flat fading where all frequency components of the signal experience the same magnitude of fading.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_fading_channels_thumb.png
      :alt: Fading Channels in Wireless Communications

    :ref:`sphx_glr_auto_examples_channels_plot_fading_channels.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fading Channels in Wireless Communications</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the LaplacianChannel in Kaira, which models channels with impulsive noise that follows the Laplacian distribution. Unlike Gaussian noise, Laplacian noise has heavier tails, making it suitable for modeling environments with occasional large noise spikes.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_laplacian_channel_thumb.png
      :alt: Impulsive Noise with Laplacian Channel

    :ref:`sphx_glr_auto_examples_channels_plot_laplacian_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Impulsive Noise with Laplacian Channel</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the NonlinearChannel in Kaira, which allows modeling various nonlinear signal distortions commonly encountered in communication systems. Nonlinearities occur in many components such as amplifiers, mixers, and converters, and can significantly impact system performance.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_nonlinear_channel_thumb.png
      :alt: Nonlinear Channel Distortion Effects

    :ref:`sphx_glr_auto_examples_channels_plot_nonlinear_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear Channel Distortion Effects</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the PhaseNoiseChannel in Kaira, which simulates phase noise commonly encountered in oscillators and frequency synthesizers. Phase noise is a critical impairment in high-frequency communication systems and can severely degrade performance even when signal amplitude remains intact.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_phase_noise_channel_thumb.png
      :alt: Phase Noise Effects on Signal Constellations

    :ref:`sphx_glr_auto_examples_channels_plot_phase_noise_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Phase Noise Effects on Signal Constellations</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the PoissonChannel in Kaira, which models signal-dependent noise commonly found in optical systems and photon-counting detectors. Unlike AWGN where noise is independent of signal intensity, Poisson noise increases with signal strength, making it essential for accurate modeling of optical communications and imaging systems.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_poisson_channel_thumb.png
      :alt: Poisson Channel for Signal-Dependent Noise

    :ref:`sphx_glr_auto_examples_channels_plot_poisson_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Poisson Channel for Signal-Dependent Noise</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the difference between Rician and Rayleigh fading channels in Kaira. While both model multipath propagation in wireless communications, Rician fading includes a dominant line-of-sight component, making it suitable for modeling wireless channels where there is a direct path between transmitter and receiver. We'll visualize the effect of different K-factors in Rician fading and compare with Rayleigh fading.">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_rician_fading_thumb.png
      :alt: Rician Fading vs Rayleigh Fading Channels

    :ref:`sphx_glr_auto_examples_channels_plot_rician_fading.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rician Fading vs Rayleigh Fading Channels</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the UplinkMACChannel class to simulate uplink communication scenarios with multiple users transmitting simultaneously. It demonstrates both shared channel and per-user channel configurations, and shows how to dynamically update channel parameters. Key Features Demonstrated: - Using a single shared channel for all users - Using different channels for each user - Dynamic parameter updates during transmission - Signal visualization and analysis">

.. only:: html

    .. image:: /auto_examples/channels/images/thumb/sphx_glr_plot_uplink_mac_channel_thumb.png
      :alt: UplinkMACChannel Usage with Different Channel Types

    :ref:`sphx_glr_auto_examples_channels_plot_uplink_mac_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UplinkMACChannel Usage with Different Channel Types</div>
    </div>

.. raw:: html

    </div>


.. toctree:
   :hidden:

   /auto_examples/channels/plot_awgn_channel
   /auto_examples/channels/plot_binary_channels
   /auto_examples/channels/plot_channel_comparison
   /auto_examples/channels/plot_composite_channels
   /auto_examples/channels/plot_fading_channels
   /auto_examples/channels/plot_laplacian_channel
   /auto_examples/channels/plot_nonlinear_channel
   /auto_examples/channels/plot_phase_noise_channel
   /auto_examples/channels/plot_poisson_channel
   /auto_examples/channels/plot_rician_fading
   /auto_examples/channels/plot_uplink_mac_channel

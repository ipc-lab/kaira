:orphan:

Channel Models
==============

Examples demonstrating various communication channel models and their characteristics.

.. raw:: html

    <div class="sphx-glr-thumbnails">

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

    </div>

.. toctree::
   :hidden:
   
   /auto_examples/channel_models/plot_awgn_channel
   /auto_examples/channel_models/plot_laplacian_channel
   /auto_examples/channel_models/plot_phase_noise_channel
   /auto_examples/channel_models/plot_binary_channels
   /auto_examples/channel_models/plot_fading_channels
   /auto_examples/channel_models/plot_nonlinear_channel
   /auto_examples/channel_models/plot_composite_channels
   /auto_examples/channel_models/plot_poisson_channel

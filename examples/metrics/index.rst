:orphan:

Metrics
==============

Examples demonstrating Kaira's comprehensive metrics suite for evaluating signal
quality, error rates, and image quality in communication and signal processing
systems.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates signal quality metrics including SNR, BER, SER, and FER. Shows how to evaluate communication system performance using multiple metrics.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_signal_metrics_thumb.png
      :alt: Signal Quality Metrics

    :ref:`sphx_glr_auto_examples_metrics_plot_signal_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Signal Quality Metrics</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Explores image quality metrics including PSNR, SSIM, MS-SSIM, and LPIPS. Compares their effectiveness for different types of image distortions.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_image_metrics_thumb.png
      :alt: Image Quality Metrics

    :ref:`sphx_glr_auto_examples_metrics_plot_image_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Image Quality Metrics</div>
    </div>

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/metrics/plot_signal_metrics
   /auto_examples/metrics/plot_image_metrics

Detailed Topics
-------------------------------------------

Signal Metrics
~~~~~~~~~~~~~
The signal metrics examples demonstrate evaluation of communication system performance:

* Bit Error Rate (BER) - Fundamental measure of digital transmission accuracy
* Symbol Error Rate (SER) - Error rate for multi-bit symbols
* Block Error Rate (BLER) - Error rate for blocks of bits
* Frame Error Rate (FER) - Error rate for larger data frames
* Signal-to-Noise Ratio (SNR) - Basic measure of signal quality

Image Metrics
~~~~~~~~~~~~
The image quality metrics examples cover both traditional and modern approaches:

* PSNR - Peak Signal-to-Noise Ratio for pixel-level fidelity
* SSIM - Structural Similarity for perceptual quality
* MS-SSIM - Multi-scale extension of SSIM
* LPIPS - Learned Perceptual Image Patch Similarity

Each example includes:

* Detailed implementation and usage examples
* Visualization of metric behavior
* Analysis of trade-offs between different metrics
* Practical considerations for real-world applications
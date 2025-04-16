:orphan:

Performance Metrics
===================
Examples demonstrating various metrics for performance evaluation in communications and signal processing.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of signal and error rate metrics in the Kaira library, including BER (Bit Error Rate), BLER (Block Error Rate), SER (Symbol Error Rate), FER (Frame Error Rate), and SNR (Signal-to-Noise Ratio). These metrics are essential for evaluating the performance of communication systems.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_signal_metrics_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_metrics_plot_signal_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Signal and Error Rate Metrics</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the image quality metrics available in Kaira, including PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), MS-SSIM (Multi-Scale SSIM), and LPIPS (Learned Perceptual Image Patch Similarity). These metrics are particularly useful for evaluating image compression algorithms, assessing deep learning-based image processing, and quality control in image transmission systems.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_image_metrics_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_metrics_plot_image_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Image Quality Metrics</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to create custom metrics by extending the BaseMetric class in the Kaira library. Custom metrics allow you to implement specialized performance measurements for your particular communication system requirements.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_custom_metrics_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_metrics_plot_custom_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Creating Custom Metrics</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to create and use composite metrics in Kaira. Composite metrics combine multiple individual metrics into a single evaluation measure, which is useful for multi-objective assessment and performance optimization.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_composite_metrics_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_metrics_plot_composite_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Composite Metrics for Multi-Objective Evaluation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the MetricRegistry in Kaira to manage, create, and utilize different metrics efficiently. The registry provides a centralized way to register and access metrics, making it easy to create standardized evaluation pipelines.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_metrics_registry_thumb.png
      :alt:

    :ref:`sphx_glr_auto_examples_metrics_plot_metrics_registry.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using the Metrics Registry</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates how to create visually appealing visualizations of performance metrics used in communication systems including BER, SNR, capacity, and other important measures.">

.. only:: html

    .. image:: /auto_examples/metrics/images/thumb/sphx_glr_plot_performance_metrics_thumb.png
      :alt: Performance Metrics Visualization

    :ref:`sphx_glr_auto_examples_metrics_plot_performance_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Performance Metrics Visualization</div>
    </div>

.. raw:: html

    </div>

.. toctree::
   :hidden:

   /auto_examples/metrics/plot_signal_metrics
   /auto_examples/metrics/plot_image_metrics
   /auto_examples/metrics/plot_custom_metrics
   /auto_examples/metrics/plot_composite_metrics
   /auto_examples/metrics/plot_metrics_registry

Signal and Error Rate Metrics
-----------------------------
BER, BLER, SER, and other signal quality metrics are fundamental for assessing the performance of communication systems.

Image Quality Metrics
---------------------
PSNR, SSIM, MS-SSIM and LPIPS metrics for evaluating image quality and perceptual similarity.

Custom Metrics
--------------
Learn how to create custom metrics tailored to your specific application requirements.

Composite Metrics
-----------------
Examples showing how to combine multiple metrics for comprehensive evaluation.

Metrics Registry
----------------
The metrics registry provides a flexible way to register and manage various metrics in your projects.

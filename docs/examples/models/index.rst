:orphan:

Models
======

Neural network models and architectures for communications, including deep learning approaches to channel coding, modulation, and signal processing.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="maps by explicitly modeling interdependencies between channel state information and input features. The AFModule allows the same model to be used during training and testing across channels with different signal-to-noise ratios without significant performance degradation. It is particularly useful in wireless communication scenarios where channel conditions vary.">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_afmodule_thumb.png
      :alt: Attention-Feature Module (AFModule)

    :ref:`sphx_glr_auto_examples_models_plot_afmodule.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attention-Feature Module (AFModule)</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the original DeepJSCC model from Bourtsoulatze et al. (2019), which pioneered deep learning-based joint source-channel coding for image transmission over wireless channels.">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_bourtsoulatze_deepjscc_thumb.png
      :alt: Original DeepJSCC Model (Bourtsoulatze 2019)

    :ref:`sphx_glr_auto_examples_models_plot_bourtsoulatze_deepjscc.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Original DeepJSCC Model (Bourtsoulatze 2019)</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of the ChannelAwareBaseModel abstract base class, which standardizes how Channel State Information (CSI) is handled across different models in the Kaira framework. The ChannelAwareBaseModel provides: - Standardized CSI validation and normalization - Utility methods for CSI transformation - Helper functions for passing CSI to submodules - Consistent interface for channel-aware models">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_channel_aware_base_model_thumb.png
      :alt: Channel-Aware Base Model Example

    :ref:`sphx_glr_auto_examples_models_plot_channel_aware_base_model.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Channel-Aware Base Model Example</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of complex-valued projections in Kaira for dimensionality reduction in wireless communication systems. Complex projections are essential for efficiently representing signals with in-phase (I) and quadrature (Q) components commonly found in wireless communications. We'll visualize and compare: 1. Real-valued projections (Rademacher, Gaussian, Orthogonal) 2. Complex-valued projections (Complex Gaussian, Complex Orthogonal) 3. Applications to wireless channel modeling and signal compression">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_complex_projections_thumb.png
      :alt: Complex Projections for Wireless Communications

    :ref:`sphx_glr_auto_examples_models_plot_complex_projections.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Complex Projections for Wireless Communications</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the DeepJSCC model for image transmission over a noisy channel. DeepJSCC is an end-to-end approach that jointly optimizes source compression and channel coding using deep neural networks, providing robust performance in varying channel conditions.">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_deepjscc_model_thumb.png
      :alt: Deep Joint Source-Channel Coding (DeepJSCC) Model

    :ref:`sphx_glr_auto_examples_models_plot_deepjscc_model.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Deep Joint Source-Channel Coding (DeepJSCC) Model</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the MultipleAccessChannelModel for transmitting information from multiple users over a shared channel. This model simulates scenarios where multiple transmitters send signals simultaneously and a single receiver tries to recover all messages.">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_multiple_access_channel_thumb.png
      :alt: Multiple Access Channel Model for Joint Encoding

    :ref:`sphx_glr_auto_examples_models_plot_multiple_access_channel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multiple Access Channel Model for Joint Encoding</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the usage of projections in Kaira for dimensionality reduction in communication systems, along with techniques to evaluate projection quality using cover tests. Projections are critical for efficient signal representation and transmission in bandwidth-constrained channels. We'll visualize three types of projections: 1. Rademacher projections (random binary matrices) 2. Gaussian projections (random Gaussian matrices) 3. Orthogonal projections (matrices with orthogonal columns) and evaluate their effectiveness using cover tests and reconstruction quality metrics. These projections have been previously used in (and adapted from) :cite:`yilmaz2025learning,yilmaz2025private`.">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_projections_and_cover_tests_thumb.png
      :alt: Projections and Cover Tests for Communication Systems

    :ref:`sphx_glr_auto_examples_models_plot_projections_and_cover_tests.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Projections and Cover Tests for Communication Systems</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the SequentialModel as a foundation for building modular neural network architectures. The SequentialModel allows you to compose multiple modules together, similar to PyTorch's nn.Sequential but with additional features for communication system modeling.">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_sequential_model_thumb.png
      :alt: Sequential Model for Modular Neural Network Design

    :ref:`sphx_glr_auto_examples_models_plot_sequential_model.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sequential Model for Modular Neural Network Design</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the Discrete Task-Oriented Deep JSCC (DT-DeepJSCC) model from Xie et al. (2023). Unlike traditional DeepJSCC which focuses on image reconstruction, DT-DeepJSCC is designed for task-oriented semantic communications, specifically for image classification tasks. It uses a discrete bottleneck for robustness against channel impairments.">

.. only:: html

    .. image:: /auto_examples/models/images/thumb/sphx_glr_plot_xie2023_dt_deepjscc_thumb.png
      :alt: Discrete Task-Oriented Deep JSCC Model (Xie 2023)

    :ref:`sphx_glr_auto_examples_models_plot_xie2023_dt_deepjscc.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Discrete Task-Oriented Deep JSCC Model (Xie 2023)</div>
    </div>

.. raw:: html

    </div>


.. toctree:
   :hidden:

   /auto_examples/models/plot_afmodule
   /auto_examples/models/plot_bourtsoulatze_deepjscc
   /auto_examples/models/plot_channel_aware_base_model
   /auto_examples/models/plot_complex_projections
   /auto_examples/models/plot_deepjscc_model
   /auto_examples/models/plot_multiple_access_channel
   /auto_examples/models/plot_projections_and_cover_tests
   /auto_examples/models/plot_sequential_model
   /auto_examples/models/plot_xie2023_dt_deepjscc

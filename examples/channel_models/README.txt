Channel Models
==============

These examples demonstrate how to use different communication channel models in the Kaira library.

Communication channels represent the medium through which signals are transmitted, and they introduce
various forms of noise and distortion. Kaira provides implementations for common channel models used
in information theory and communications.

Examples
--------

.. toctree::
   :maxdepth: 1

   plot_awgn_channel
   plot_binary_channels
   plot_composite_channels
   plot_fading_channels
   plot_laplacian_channel
   plot_nonlinear_channel
   plot_phase_noise_channel
   plot_poisson_channel

Key Concepts
-----------

* **Channel Capacity**: The maximum rate at which information can be transmitted through a channel
* **Signal-to-Noise Ratio (SNR)**: The ratio of signal power to noise power, often expressed in decibels
* **Fading**: Time-varying attenuation of signal strength in wireless communications
* **Additive White Gaussian Noise (AWGN)**: The most common noise model, where Gaussian noise is added to the signal

See Also
--------

* :mod:`kaira.channels` module documentation
* :ref:`Channel models reference <ch_reference>`


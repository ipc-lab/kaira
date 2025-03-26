Deep Joint Source-Channel Coding
================================

These examples demonstrate Deep Joint Source-Channel Coding (Deep JSCC) techniques in the Kaira library.

Deep JSCC combines traditional source and channel coding into a single end-to-end optimized neural network,
providing an alternative to separate source and channel coding approaches. These techniques are particularly
valuable for communication over noisy or bandwidth-limited channels.

Examples
--------

.. toctree::
   :maxdepth: 1

   # Add plot examples filenames here without .py extension
   # e.g., plot_jscc_autoencoder

Key Concepts
------------

* **End-to-End Optimization**: Joint optimization of source and channel coding for specific channel conditions
* **Autoencoder Architecture**: Neural network architectures that encode and decode signals in a single model
* **Rate-Distortion Trade-off**: The relationship between compression rate and reconstruction quality
* **Channel-Aware Training**: Training neural networks with awareness of channel characteristics

See Also
--------

* :mod:`kaira.models` module documentation


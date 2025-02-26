.. _basic_usage_example:

Basic Usage Example
==================

This example demonstrates how to use Kaira for image transmission over a noisy channel.

.. include:: ../../examples/basic_usage.py
   :literal:
   :start-after: """Basic Usage of Kaira
   :end-before: import matplotlib

The example covers:

* Loading and preprocessing an image
* Creating JSCC model components
* Setting up a channel with specific SNR
* Transmitting the image through the channel
* Evaluating reconstruction quality

Full Source Code
---------------

.. literalinclude:: ../../examples/basic_usage.py
   :language: python
   :linenos:

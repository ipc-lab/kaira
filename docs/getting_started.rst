Getting Started with Kaira
===========================

.. warning::
   Kaira is currently in beta. The API is subject to change as we refine the library based on user feedback and evolving research needs.

This guide will help you get up and running with Kaira quickly, demonstrating its core capabilities through simple examples.

Installation
------------

Install Kaira using pip:

.. code-block:: bash

    pip install pykaira

Basic Usage
-----------

Here's a simple example demonstrating how to use Kaira to simulate a basic communication system:

.. code-block:: python

    import torch
    import kaira

    # Create a simple AWGN channel
    channel = kaira.channels.AWGNChannel(snr_db=10.0)

    # Generate some random data to transmit
    data = torch.randn(100, 8)

    # Pass the data through the channel
    received_data = channel(data)

    print(f"Original data shape: {data.shape}")
    print(f"Received data shape: {received_data.shape}")

Deep Learning Example
---------------------

Kaira integrates seamlessly with PyTorch for deep learning applications:

.. code-block:: python

    import torch
    import torch.nn as nn
    import kaira

    # Define a simple autoencoder model
    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )

        def forward(self, x, channel):
            # Encode the input
            encoded = self.encoder(x)

            # Pass through the channel
            channel_out = channel(encoded)

            # Decode the channel output
            decoded = self.decoder(channel_out)
            return decoded

    # Create a model, channel, and some data
    model = SimpleAutoencoder(input_dim=28*28, latent_dim=16)
    channel = kaira.channels.AWGNChannel(snr_db=15.0)
    dummy_data = torch.randn(32, 28*28)  # Batch of 32 images

    # Forward pass through the model and channel
    output = model(dummy_data, channel)

    print(f"Input shape: {dummy_data.shape}")
    print(f"Output shape: {output.shape}")

Next Steps
----------

- Check out the :doc:`API Reference </api_reference>` for detailed information on Kaira's modules
- Browse through :ref:`kaira_examples_gallery` for more advanced use cases
- Learn about :doc:`Best Practices </best_practices>` for using Kaira effectively

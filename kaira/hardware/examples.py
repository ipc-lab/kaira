"""Example usage of Kaira GNU Radio integration.

This script demonstrates how to use Kaira models with GNU Radio for hardware-in-the-loop testing
and deployment.
"""

import torch

from kaira.hardware import FrequencyBand, GNURadioBridge, SDRConfig


def create_deepjscc_models(input_dim: int = 128, latent_dim: int = 64, channel_dim: int = 32):
    """Create example DeepJSCC encoder and decoder models."""

    # Simple encoder model
    encoder = torch.nn.Sequential(torch.nn.Linear(input_dim, latent_dim), torch.nn.ReLU(), torch.nn.Linear(latent_dim, channel_dim), torch.nn.Tanh())  # Normalize output for transmission

    # Simple decoder model
    decoder = torch.nn.Sequential(torch.nn.Linear(channel_dim, latent_dim), torch.nn.ReLU(), torch.nn.Linear(latent_dim, input_dim), torch.nn.Sigmoid())  # Output between 0 and 1

    return encoder, decoder


def example_transmission():
    """Example of setting up a transmission system."""

    # Create models
    encoder, decoder = create_deepjscc_models()

    # Configure SDR for 915 MHz ISM band
    config = SDRConfig(center_frequency=FrequencyBand.ISM_915.value, sample_rate=1e6, tx_gain=20.0, rx_gain=30.0, buffer_size=1024)  # 1 MHz

    print(f"Configuring SDR for {config.center_frequency/1e6:.1f} MHz")
    print(f"Sample rate: {config.sample_rate/1e6:.1f} MHz")

    # Create GNU Radio bridge
    bridge = GNURadioBridge(config)

    # Create flowgraph
    bridge.create_flowgraph()

    # Add source (random data for testing)
    source = bridge.add_source("signal_generator", frequency=1000, amplitude=0.1)

    # Add torch encoder
    torch_encoder = bridge.add_torch_encoder(encoder)

    # Add sink (file for testing, USRP for real hardware)
    sink = bridge.add_sink("file", filename="transmitted_signal.dat")

    # Connect blocks
    bridge.connect(source, torch_encoder)
    bridge.connect(torch_encoder, sink)

    print("Starting transmission...")
    bridge.start()

    # Run for a few seconds
    import time

    time.sleep(5)

    bridge.stop()
    print("Transmission complete")


def example_reception():
    """Example of setting up a reception system."""

    # Create models
    encoder, decoder = create_deepjscc_models()

    # Configure SDR
    config = SDRConfig(center_frequency=FrequencyBand.ISM_915.value, sample_rate=1e6, rx_gain=30.0, buffer_size=1024)

    # Create GNU Radio bridge
    bridge = GNURadioBridge(config)

    # Create flowgraph
    bridge.create_flowgraph()

    # Add source (file for testing, USRP for real hardware)
    source = bridge.add_source("file", filename="transmitted_signal.dat")

    # Add torch decoder
    torch_decoder = bridge.add_torch_decoder(decoder)

    # Add sink
    sink = bridge.add_sink("file", filename="decoded_data.bin")

    # Connect blocks
    bridge.connect(source, torch_decoder)
    bridge.connect(torch_decoder, sink)

    print("Starting reception...")
    bridge.start()

    # Run for a few seconds
    import time

    time.sleep(5)

    bridge.stop()
    print("Reception complete")


def example_full_loop():
    """Example of a complete transmission and reception loop."""

    try:
        print("=== Kaira GNU Radio Integration Example ===")
        print()

        print("1. Setting up transmission...")
        example_transmission()
        print()

        print("2. Setting up reception...")
        example_reception()
        print()

        print("Example completed successfully!")

    except ImportError as e:
        print(f"GNU Radio not available: {e}")
        print("Please install GNU Radio with: pip install gnuradio")

    except Exception as e:
        print(f"Error during example: {e}")


if __name__ == "__main__":
    example_full_loop()

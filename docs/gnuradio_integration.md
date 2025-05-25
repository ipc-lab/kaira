# GNU Radio Integration for Kaira

This module provides hardware integration capabilities for Kaira, specifically focusing on Software Defined Radio (SDR) integration through GNU Radio.

## Overview

The GNU Radio integration allows you to:

- Use Kaira's PyTorch-based models (encoders/decoders) in real-time GNU Radio flowgraphs
- Interface with USRP and other SDR hardware
- Perform hardware-in-the-loop testing of deep learning communication systems
- Deploy trained models to real wireless environments

## Installation

### Prerequisites

1. **GNU Radio 3.9+**:

   ```bash
   # Ubuntu/Debian
   sudo apt-get install gnuradio gnuradio-dev

   # Or via conda
   conda install -c conda-forge gnuradio

   # Or via pip (limited functionality)
   pip install gnuradio
   ```

2. **USRP Hardware Drivers** (if using USRP):

   ```bash
   sudo apt-get install libuhd-dev uhd-host
   ```

3. **Updated Kaira dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Configuration

```python
from kaira.hardware import SDRConfig, FrequencyBand, GNURadioBridge
import torch

# Configure SDR parameters
config = SDRConfig(
    center_frequency=FrequencyBand.ISM_915.value,  # 915 MHz
    sample_rate=1e6,  # 1 MHz
    tx_gain=20.0,     # 20 dB
    rx_gain=30.0,     # 30 dB
    buffer_size=1024
)

# Create your PyTorch models
encoder = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.Tanh()
)

decoder = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 128),
    torch.nn.Sigmoid()
)
```

### Setting up a Transmitter

```python
# Create GNU Radio bridge
bridge = GNURadioBridge(config)

# Create flowgraph
flowgraph = bridge.create_flowgraph()

# Add data source
source = bridge.add_source("file", filename="input_data.bin")

# Add PyTorch encoder block
torch_encoder = bridge.add_torch_encoder(encoder)

# Add USRP sink (or file sink for testing)
sink = bridge.add_sink("usrp")  # or "file" for testing

# Connect the blocks
bridge.connect(source, torch_encoder)
bridge.connect(torch_encoder, sink)

# Start transmission
bridge.start()
```

### Setting up a Receiver

```python
# Create GNU Radio bridge
bridge = GNURadioBridge(config)

# Create flowgraph
flowgraph = bridge.create_flowgraph()

# Add USRP source (or file source for testing)
source = bridge.add_source("usrp")  # or "file" for testing

# Add PyTorch decoder block
torch_decoder = bridge.add_torch_decoder(decoder)

# Add data sink
sink = bridge.add_sink("file", filename="output_data.bin")

# Connect the blocks
bridge.connect(source, torch_decoder)
bridge.connect(torch_decoder, sink)

# Start reception
bridge.start()
```

## API Reference

### SDRConfig

Configuration class for SDR hardware parameters.

**Parameters:**

- `center_frequency` (float): Center frequency in Hz
- `sample_rate` (float): Sample rate in Hz
- `bandwidth` (float, optional): Bandwidth in Hz (defaults to sample_rate)
- `tx_gain` (float): Transmit gain in dB (0-100)
- `rx_gain` (float): Receive gain in dB (0-100)
- `antenna` (str): Antenna selection ("TX/RX", "RX2", etc.)
- `buffer_size` (int): Buffer size for processing
- `device_args` (str, optional): Device-specific arguments
- `stream_args` (dict, optional): Stream-specific arguments

### FrequencyBand

Enumeration of common frequency bands:

- `ISM_433`: 433 MHz ISM band
- `ISM_915`: 915 MHz ISM band
- `ISM_2400`: 2.4 GHz ISM band
- `WIFI_2400`: WiFi 2.4 GHz
- `WIFI_5000`: WiFi 5 GHz
- `GPS_L1`: GPS L1 band
- `FM_BROADCAST`: FM radio band

### GNURadioBridge

Main bridge class for GNU Radio integration.

**Methods:**

- `create_flowgraph()`: Create a new GNU Radio flowgraph
- `add_source(source_type, **kwargs)`: Add signal source
- `add_sink(sink_type, **kwargs)`: Add signal sink
- `add_torch_encoder(model)`: Add PyTorch encoder block
- `add_torch_decoder(model)`: Add PyTorch decoder block
- `connect(src, dst)`: Connect two blocks
- `start()`: Start the flowgraph
- `stop()`: Stop the flowgraph

## Advanced Usage

### Custom GNU Radio Blocks

You can create custom GNU Radio blocks that integrate with PyTorch:

```python
from kaira.hardware.gnuradio_bridge import GNURadioBlock
import numpy as np

class CustomProcessingBlock(GNURadioBlock):
    def __init__(self, model):
        super().__init__(
            name="custom_processing",
            input_signature=[np.complex64],
            output_signature=[np.complex64]
        )
        self.set_model(model)

    def work(self, input_items, output_items):
        input_data = input_items[0]

        # Custom processing with PyTorch model
        with torch.no_grad():
            processed = self._model(torch.from_numpy(input_data))
            output_data = processed.numpy()

        output_items[0][:len(output_data)] = output_data
        return len(output_data)
```

### Hardware-in-the-Loop Testing

For complete hardware-in-the-loop testing:

```python
from kaira.hardware import GNURadioTransmitter, GNURadioReceiver

# Create specialized transmitter and receiver
transmitter = GNURadioTransmitter(config, encoder_model)
receiver = GNURadioReceiver(config, decoder_model)

# Start both simultaneously for loop testing
transmitter.start()
receiver.start()

# Run test
import time
time.sleep(10)  # Run for 10 seconds

# Stop both
transmitter.stop()
receiver.stop()
```

## Examples

See `kaira/hardware/examples.py` for complete working examples including:

- Basic transmission and reception
- DeepJSCC model integration
- File-based testing
- Hardware loop testing

## Troubleshooting

### Common Issues

1. **GNU Radio not found**: Make sure GNU Radio is properly installed and accessible
2. **USRP not detected**: Check USB connections and UHD drivers
3. **Permission errors**: May need to add user to `usrp` group or run with sudo
4. **Frequency out of range**: Check SDR hardware specifications
5. **Sample rate too high**: Reduce sample rate or check hardware capabilities

### Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check GNU Radio version:

```python
import gnuradio
print(gnuradio.version())
```

## Hardware Compatibility

### Tested Hardware

- USRP B200/B210
- USRP N200/N210
- RTL-SDR dongles (receive only)
- BladeRF
- HackRF (with gr-osmosdr)

### Frequency Ranges

- Most USRP devices: 70 MHz - 6 GHz
- RTL-SDR: 24 MHz - 1.7 GHz
- Check your specific hardware documentation

## Performance Considerations

- **Real-time constraints**: GNU Radio operates in real-time, ensure models are fast enough
- **GPU acceleration**: PyTorch models can use GPU, but data transfer adds latency
- **Buffer sizes**: Larger buffers improve throughput but increase latency
- **Sample rates**: Higher rates require more computational power

## Contributing

To contribute to the GNU Radio integration:

1. Test with different hardware configurations
2. Add support for additional GNU Radio blocks
3. Improve error handling and robustness
4. Add more comprehensive examples
5. Update documentation

For issues and feature requests, please use the Kaira GitHub repository.

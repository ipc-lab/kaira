"""GNU Radio integration bridge for Kaira.

This module provides interfaces to connect Kaira's PyTorch-based models with GNU Radio flowgraphs
for hardware-in-the-loop testing and deployment.
"""

import threading
import time
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

from .sdr_utils import HardwareError, SDRConfig

# Type checking imports - only for mypy/IDE support
if TYPE_CHECKING:
    try:
        from gnuradio import analog, audio, blocks, digital, gr, uhd  # type: ignore
    except ImportError:
        # Create type stubs for type checking when GNU Radio is not installed
        gr = Any
        blocks = Any
        analog = Any
        digital = Any
        uhd = Any
        audio = Any

# Runtime import with graceful fallback
GNURADIO_AVAILABLE = False
try:
    from gnuradio import analog, audio, blocks, digital, gr, uhd  # type: ignore

    GNURADIO_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"GNU Radio not available: {e}. " "Hardware integration features will be disabled. " "Install gnuradio>=3.9.0 to enable SDR functionality.", ImportWarning, stacklevel=2)
    # Create dummy modules for runtime when GNU Radio is not available
    if not TYPE_CHECKING:

        class _DummyModule:
            """Dummy module for when GNU Radio is not available."""

            def __getattr__(self, name: str) -> Any:
                """Raise ImportError for any attribute access."""
                raise ImportError("GNU Radio is not available")

        gr = _DummyModule()
        blocks = _DummyModule()
        analog = _DummyModule()
        digital = _DummyModule()
        uhd = _DummyModule()
        audio = _DummyModule()


# Define GNURadioBlockBase using TYPE_CHECKING to make mypy happy
# For type checking, define a base class interface that both implementations will follow
if TYPE_CHECKING:

    class _GNURadioBlockBaseType:
        """Type interface for GNU Radio block base class.

        This class provides a type annotation interface for the underlying GNURadioBlockBase
        implementations.
        """

        def __init__(self, name: str, input_signature: Any, output_signature: Any) -> None:
            """Initialize a GNU Radio block.

            Args:
                name: Block name
                input_signature: Input data type
                output_signature: Output data type
            """
            ...

        def set_model(self, model: torch.nn.Module) -> None:
            """Set PyTorch model for signal processing.

            Args:
                model: PyTorch model to use
            """
            ...

        def work(self, input_items: Any, output_items: Any) -> int:
            """Process input data and produce output.

            Args:
                input_items: Input data arrays
                output_items: Output data arrays

            Returns:
                Number of items produced
            """
            ...


# Runtime implementations based on whether GNU Radio is available
if GNURADIO_AVAILABLE:
    # Real GNU Radio block implementation
    class GNURadioBlockBase(gr.sync_block):  # type: ignore
        """Base GNU Radio block that interfaces with PyTorch models.

        This class provides the foundation for creating GNU Radio blocks that can
        execute PyTorch models in real-time signal processing pipelines.

        Attributes:
            _torch_device: The device (CPU/GPU) where PyTorch computations are performed
            _model: The PyTorch model to execute in this block
        """

        def __init__(self, name: str, input_signature, output_signature):
            """Initialize the GNU Radio block.

            Args:
                name: Name of the GNU Radio block
                input_signature: Input data type signature for GNU Radio
                output_signature: Output data type signature for GNU Radio
            """
            super().__init__(name=name, in_sig=input_signature, out_sig=output_signature)
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = None

        def set_model(self, model: torch.nn.Module):
            """Set the PyTorch model for this block.

            Args:
                model: PyTorch model to use for signal processing
            """
            self._model = model
            if self._model is not None:
                self._model.eval()
                self._model.to(self._torch_device)

        def work(self, input_items, output_items):
            """Main processing function - to be implemented by subclasses.

            Args:
                input_items: List of input numpy arrays from GNU Radio
                output_items: List of output numpy arrays to GNU Radio

            Returns:
                Number of items processed
            """
            raise NotImplementedError

else:
    # Stub implementation when GNU Radio is not available
    class GNURadioBlockBase:  # type: ignore
        """Dummy GNU Radio block when GNU Radio is not available.

        This class provides a fallback implementation that raises appropriate errors when GNU Radio
        is not installed.
        """

        def __init__(self, name: str, input_signature, output_signature):
            """Initialize dummy block - raises ImportError.

            Args:
                name: Name of the GNU Radio block
                input_signature: Input data type signature
                output_signature: Output data type signature

            Raises:
                ImportError: Always raised since GNU Radio is not available
            """
            raise ImportError("GNU Radio is not available. Please install gnuradio>=3.9.0")

        def set_model(self, model: torch.nn.Module):
            """Dummy set_model method - raises ImportError.

            Args:
                model: PyTorch model (ignored)

            Raises:
                ImportError: Always raised since GNU Radio is not available
            """
            raise ImportError("GNU Radio is not available. Please install gnuradio>=3.9.0")

        def work(self, input_items, output_items):
            """Dummy work method - raises ImportError.

            Args:
                input_items: Input items (ignored)
                output_items: Output items (ignored)

            Raises:
                ImportError: Always raised since GNU Radio is not available
            """
            raise ImportError("GNU Radio is not available. Please install gnuradio>=3.9.0")


# Alias for backwards compatibility
GNURadioBlock = GNURadioBlockBase


class TorchEncoderBlock(GNURadioBlock):
    """GNU Radio block that wraps a PyTorch encoder model.

    This block takes input data as bytes and processes it through a PyTorch
    encoder model, producing complex symbols suitable for transmission.

    Attributes:
        _model: The PyTorch encoder model
    """

    def __init__(self, model: Optional[torch.nn.Module] = None):
        """Initialize the encoder block.

        Args:
            model: Optional PyTorch encoder model to use
        """
        super().__init__(name="torch_encoder", input_signature=[np.byte], output_signature=[np.complex64])  # Input bytes  # Output complex symbols
        if model is not None:
            self.set_model(model)

    def work(self, input_items, output_items):
        """Process input bytes through the encoder model.

        Args:
            input_items: List containing input byte arrays
            output_items: List containing output complex symbol arrays

        Returns:
            Number of output items produced

        Raises:
            RuntimeError: If no model has been set
        """
        if self._model is None:
            raise RuntimeError("No model set for encoder block")

        input_data = input_items[0]
        num_items = len(input_data)

        if num_items == 0:
            return 0

        # Convert to torch tensor
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data).float().to(self._torch_device)

            # Process through model
            encoded = self._model(input_tensor)

            # Convert back to numpy
            if torch.is_complex(encoded):
                output_data = encoded.cpu().numpy().astype(np.complex64)
            else:
                # Assume real output should be converted to complex
                output_data = encoded.cpu().numpy().astype(np.float32)
                output_data = output_data.astype(np.complex64)

        output_items[0][: len(output_data)] = output_data
        return len(output_data)


class TorchDecoderBlock(GNURadioBlock):
    """GNU Radio block that wraps a PyTorch decoder model.

    This block takes complex symbols as input and processes them through
    a PyTorch decoder model to recover the original bytes.

    Attributes:
        _model: The PyTorch decoder model
    """

    def __init__(self, model: Optional[torch.nn.Module] = None):
        """Initialize the decoder block.

        Args:
            model: Optional PyTorch decoder model to use
        """
        super().__init__(name="torch_decoder", input_signature=[np.complex64], output_signature=[np.byte])  # Input complex symbols  # Output bytes
        if model is not None:
            self.set_model(model)

    def work(self, input_items, output_items):
        """Process input complex symbols through the decoder model.

        Args:
            input_items: List containing input complex symbol arrays
            output_items: List containing output byte arrays

        Returns:
            Number of output items produced

        Raises:
            RuntimeError: If no model has been set
        """
        if self._model is None:
            raise RuntimeError("No model set for decoder block")

        input_data = input_items[0]
        num_items = len(input_data)

        if num_items == 0:
            return 0

        # Convert to torch tensor
        with torch.no_grad():
            if np.iscomplexobj(input_data):
                input_tensor = torch.from_numpy(input_data).to(self._torch_device)
            else:
                input_tensor = torch.from_numpy(input_data).float().to(self._torch_device)

            # Process through model
            decoded = self._model(input_tensor)

            # Convert back to numpy bytes
            output_data = decoded.cpu().numpy().astype(np.uint8)

        output_items[0][: len(output_data)] = output_data
        return len(output_data)


class GNURadioBridge:
    """Main bridge class for integrating Kaira models with GNU Radio."""

    def __init__(self, config: SDRConfig):
        """Initialize GNU Radio bridge.

        Args:
            config: SDR configuration parameters
        """
        if not GNURADIO_AVAILABLE:
            raise ImportError("GNU Radio is not available. Please install gnuradio>=3.9.0")

        self.config = config
        self._flowgraph = None
        self._running = False
        self._thread = None

    def create_flowgraph(self):
        """Create a GNU Radio flowgraph."""
        if not GNURADIO_AVAILABLE:
            raise ImportError("GNU Radio is not available. Please install gnuradio>=3.9.0")

        if self._flowgraph is not None:
            self.stop()

        self._flowgraph = gr.top_block("Kaira GNU Radio Bridge")
        return self._flowgraph

    def add_source(self, source_type: str = "usrp", **kwargs) -> Any:
        """Add a signal source to the flowgraph.

        Args:
            source_type: Type of source ("usrp", "file", "signal_generator")
            **kwargs: Additional arguments for the source

        Returns:
            GNU Radio source block
        """
        if self._flowgraph is None:
            raise RuntimeError("Must create flowgraph first")

        if source_type == "usrp":
            source = uhd.usrp_source(device_addr=self.config.device_args or "", stream_args=uhd.stream_args(cpu_format="fc32", channels=list(range(self.config.num_channels))))
            source.set_sample_rate(self.config.sample_rate)
            source.set_center_freq(self.config.center_frequency)
            source.set_gain(self.config.rx_gain)

        elif source_type == "file":
            filename = kwargs.get("filename", "input.dat")
            source = blocks.file_source(gr.sizeof_gr_complex, filename, True)

        elif source_type == "signal_generator":
            freq = kwargs.get("frequency", 1000)
            amplitude = kwargs.get("amplitude", 0.1)
            source = analog.sig_source_c(self.config.sample_rate, analog.GR_COS_WAVE, freq, amplitude)

        else:
            raise ValueError(f"Unknown source type: {source_type}")

        return source

    def add_sink(self, sink_type: str = "usrp", **kwargs) -> Any:
        """Add a signal sink to the flowgraph.

        Args:
            sink_type: Type of sink ("usrp", "file", "null")
            **kwargs: Additional arguments for the sink

        Returns:
            GNU Radio sink block
        """
        if self._flowgraph is None:
            raise RuntimeError("Must create flowgraph first")

        if sink_type == "usrp":
            sink = uhd.usrp_sink(device_addr=self.config.device_args or "", stream_args=uhd.stream_args(cpu_format="fc32", channels=list(range(self.config.num_channels))))
            sink.set_sample_rate(self.config.sample_rate)
            sink.set_center_freq(self.config.center_frequency)
            sink.set_gain(self.config.tx_gain)

        elif sink_type == "file":
            filename = kwargs.get("filename", "output.dat")
            sink = blocks.file_sink(gr.sizeof_gr_complex, filename)

        elif sink_type == "null":
            sink = blocks.null_sink(gr.sizeof_gr_complex)

        else:
            raise ValueError(f"Unknown sink type: {sink_type}")

        return sink

    def add_torch_encoder(self, model: torch.nn.Module) -> TorchEncoderBlock:
        """Add a PyTorch encoder block to the flowgraph."""
        encoder_block = TorchEncoderBlock(model)
        return encoder_block

    def add_torch_decoder(self, model: torch.nn.Module) -> TorchDecoderBlock:
        """Add a PyTorch decoder block to the flowgraph."""
        decoder_block = TorchDecoderBlock(model)
        return decoder_block

    def connect(self, src, dst, src_port: int = 0, dst_port: int = 0):
        """Connect two blocks in the flowgraph."""
        if self._flowgraph is None:
            raise RuntimeError("Must create flowgraph first")
        self._flowgraph.connect((src, src_port), (dst, dst_port))

    def start(self):
        """Start the GNU Radio flowgraph."""
        if self._flowgraph is None:
            raise RuntimeError("Must create flowgraph first")

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_flowgraph)
        self._thread.start()

    def stop(self):
        """Stop the GNU Radio flowgraph."""
        if not self._running:
            return

        self._running = False
        if self._flowgraph is not None:
            self._flowgraph.stop()
            self._flowgraph.wait()

        if self._thread is not None:
            self._thread.join()

    def _run_flowgraph(self):
        """Internal method to run the flowgraph."""
        try:
            self._flowgraph.start()
            while self._running:
                time.sleep(0.1)
        except Exception as e:
            raise HardwareError(f"Flowgraph execution failed: {e}")
        finally:
            if self._flowgraph is not None:
                self._flowgraph.stop()
                self._flowgraph.wait()


class GNURadioTransmitter(GNURadioBridge):
    """Specialized transmitter using GNU Radio."""

    def __init__(self, config: SDRConfig, encoder_model: torch.nn.Module):
        super().__init__(config)
        self.encoder_model = encoder_model
        self._setup_tx_flowgraph()

    def _setup_tx_flowgraph(self):
        """Setup transmitter flowgraph."""
        self.create_flowgraph()

        # Create data source (file or random data)
        self.data_source = blocks.file_source(gr.sizeof_char, "/dev/urandom", False)

        # Add torch encoder
        self.encoder = self.add_torch_encoder(self.encoder_model)

        # Add channel coding/modulation if needed

        # Add USRP sink
        self.usrp_sink = self.add_sink("usrp")

        # Connect blocks
        self.connect(self.data_source, self.encoder)
        self.connect(self.encoder, self.usrp_sink)

    def transmit_data(self, data: Union[np.ndarray, bytes]):
        """Transmit data through the encoder and SDR."""
        # Implementation would depend on specific data injection method
        pass


class GNURadioReceiver(GNURadioBridge):
    """Specialized receiver using GNU Radio."""

    def __init__(self, config: SDRConfig, decoder_model: torch.nn.Module):
        super().__init__(config)
        self.decoder_model = decoder_model
        self._setup_rx_flowgraph()

    def _setup_rx_flowgraph(self):
        """Setup receiver flowgraph."""
        self.create_flowgraph()

        # Add USRP source
        self.usrp_source = self.add_source("usrp")

        # Add signal conditioning (filters, AGC, etc.)

        # Add torch decoder
        self.decoder = self.add_torch_decoder(self.decoder_model)

        # Add data sink
        self.data_sink = blocks.file_sink(gr.sizeof_char, "received_data.bin")

        # Connect blocks
        self.connect(self.usrp_source, self.decoder)
        self.connect(self.decoder, self.data_sink)

    def receive_data(self) -> Optional[np.ndarray]:
        """Receive and decode data from SDR."""
        # Implementation would depend on specific data extraction method
        pass

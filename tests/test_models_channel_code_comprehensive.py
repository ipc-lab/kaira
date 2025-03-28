"""Comprehensive tests for the channel code model."""
import pytest
import torch
import torch.nn as nn
from kaira.models.channel_code import ChannelCodeModel
from kaira.models.base import BaseModel
from kaira.channels import BaseChannel, PerfectChannel, AWGNChannel
from kaira.constraints import BaseConstraint, AveragePowerConstraint
from kaira.modulations import BaseModulator, BaseDemodulator
from kaira.modulations import PSKModulator, PSKDemodulator


class SimpleEncoder(BaseModel):
    """A simple encoder for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        
    def forward(self, x, *args, **kwargs):
        return self.fc(x)


class SimpleDecoder(BaseModel):
    """A simple decoder for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)
        
    def forward(self, x, *args, **kwargs):
        decoded = self.fc(x)
        # Return both decoded data and a soft estimate
        return decoded, torch.sigmoid(decoded)


class SimpleModulator(BaseModulator):
    """A simple modulator for testing."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, *args, **kwargs):
        # Simple identity modulation
        return x
    
    @property
    def bits_per_symbol(self) -> int:
        return 1


class SimpleDemodulator(BaseDemodulator):
    """A simple demodulator for testing."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, *args, **kwargs):
        # Simple identity demodulation
        return x
    
    @property
    def bits_per_symbol(self) -> int:
        return 1
        

class SimpleConstraint(BaseConstraint):
    """A simple constraint for testing."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, *args, **kwargs):
        # Simple identity constraint
        return x


@pytest.fixture
def simple_channel_code_model():
    """Create a simple channel code model for testing."""
    encoder = SimpleEncoder()
    modulator = SimpleModulator()
    constraint = SimpleConstraint()
    channel = PerfectChannel()
    demodulator = SimpleDemodulator()
    decoder = SimpleDecoder()
    
    return ChannelCodeModel(
        encoder=encoder,
        modulator=modulator,
        constraint=constraint,
        channel=channel,
        demodulator=demodulator,
        decoder=decoder
    )


@pytest.fixture
def realistic_channel_code_model():
    """Create a more realistic channel code model for testing."""
    encoder = SimpleEncoder()
    modulator = PSKModulator(order=4)  # QPSK
    constraint = AveragePowerConstraint(average_power=1.0)
    channel = AWGNChannel(snr_db=10.0)
    demodulator = PSKDemodulator(order=4)
    decoder = SimpleDecoder()
    
    return ChannelCodeModel(
        encoder=encoder,
        modulator=modulator,
        constraint=constraint,
        channel=channel,
        demodulator=demodulator,
        decoder=decoder
    )


class TestChannelCodeModel:
    """Tests for the ChannelCodeModel."""
    
    def test_initialization(self, simple_channel_code_model):
        """Test model initialization."""
        model = simple_channel_code_model
        
        # Check component assignment
        assert isinstance(model.encoder, SimpleEncoder)
        assert isinstance(model.modulator, SimpleModulator)
        assert isinstance(model.constraint, SimpleConstraint)
        assert isinstance(model.channel, PerfectChannel)
        assert isinstance(model.demodulator, SimpleDemodulator)
        assert isinstance(model.decoder, SimpleDecoder)
        
        # Check steps in the sequential model
        assert len(model.steps) == 6
        
    def test_forward_perfect_channel(self, simple_channel_code_model):
        """Test forward pass with a perfect channel."""
        model = simple_channel_code_model
        batch_size = 5
        input_data = torch.randn(batch_size, 10)
        
        output = model(input_data)
        
        # Check output structure
        assert isinstance(output, dict)
        assert "final_output" in output
        assert "history" in output
        
        # Check output shapes
        assert output["final_output"].shape == (batch_size, 10)
        assert len(output["history"]) == 1
        
        # Check history content
        history_item = output["history"][0]
        assert "encoded" in history_item
        assert "received" in history_item
        assert "decoded" in history_item
        assert "soft_estimate" in history_item
        
        # With a perfect channel, decoded should be deterministically related to input
        # but not identical because of the encoding/decoding transformations
        assert not torch.allclose(output["final_output"], input_data, atol=1e-5)
        
    def test_forward_with_noisy_channel(self, realistic_channel_code_model):
        """Test forward pass with a realistic noisy channel."""
        model = realistic_channel_code_model
        batch_size = 5
        input_data = torch.randn(batch_size, 10)
        
        output = model(input_data)
        
        # Check output structure
        assert isinstance(output, dict)
        assert "final_output" in output
        assert "history" in output
        
        # Check output shapes
        assert output["final_output"].shape == (batch_size, 10)
        assert len(output["history"]) == 1
        
        # With a noisy channel, each run should give slightly different results
        first_run = output["final_output"].clone()
        second_output = model(input_data)
        second_run = second_output["final_output"]
        
        # The outputs should be different due to the random noise in the channel
        # This is only true if the channel adds noise, which the AWGNChannel does
        assert not torch.allclose(first_run, second_run, atol=1e-5)
        
    def test_history_tracking(self, simple_channel_code_model):
        """Test that the history is correctly tracked."""
        model = simple_channel_code_model
        input_data = torch.randn(3, 10)
        
        output = model(input_data)
        history = output["history"]
        
        # Check that history contains one entry (for a single iteration)
        assert len(history) == 1
        
        # Check that all expected elements are in the history
        history_item = history[0]
        assert "encoded" in history_item
        assert "received" in history_item
        assert "decoded" in history_item
        assert "soft_estimate" in history_item
        
        # Check shapes of history elements
        assert history_item["encoded"].shape == (3, 20)  # encoder output size
        assert history_item["received"].shape == (3, 20)  # channel output size
        assert history_item["decoded"].shape == (3, 10)  # decoder output size
        assert history_item["soft_estimate"].shape == (3, 10)  # soft estimate size
        
    def test_with_keyword_arguments(self, simple_channel_code_model):
        """Test model forward pass with additional keyword arguments."""
        model = simple_channel_code_model
        input_data = torch.randn(3, 10)
        
        # Pass some additional kwargs
        output = model(input_data, extra_param=42, another_param="test")
        
        # Ensure the forward pass completes successfully
        assert "final_output" in output
        assert output["final_output"].shape == (3, 10)
        
    def test_device_compatibility(self, simple_channel_code_model):
        """Test model compatibility with different devices."""
        model = simple_channel_code_model
        input_data = torch.randn(3, 10)
        
        # Move model to CPU explicitly
        model = model.to("cpu")
        input_data = input_data.to("cpu")
        
        # Forward pass should work on CPU
        output_cpu = model(input_data)
        assert output_cpu["final_output"].device.type == "cpu"
        
        # Skip GPU test if not available
        if torch.cuda.is_available():
            # Move model to GPU
            model = model.to("cuda")
            input_data = input_data.to("cuda")
            
            # Forward pass should work on GPU
            output_gpu = model(input_data)
            assert output_gpu["final_output"].device.type == "cuda"
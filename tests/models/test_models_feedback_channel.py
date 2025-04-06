# tests/test_models/test_feedback_channel.py
import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel, IdentityChannel
from kaira.models import FeedbackChannelModel
from kaira.models.registry import ModelRegistry


class SimpleEncoder(nn.Module):
    """Simple encoder for testing FeedbackChannelModel."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x, state=None):
        if state is not None:
            # Apply state as a multiplicative factor
            return self.layer(x) * state
        return self.layer(x)


class SimpleDecoder(nn.Module):
    """Simple decoder for testing FeedbackChannelModel."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(5, 10)

    def forward(self, x):
        return self.layer(x)


class SimpleFeedbackGenerator(nn.Module):
    """Simple feedback generator for testing FeedbackChannelModel."""

    def __init__(self, feedback_size=3):
        super().__init__()
        self.layer = nn.Linear(20, feedback_size)

    def forward(self, decoded, original):
        # Concatenate decoded output and original input
        combined = torch.cat([decoded, original], dim=1)
        return self.layer(combined)


class SimpleFeedbackProcessor(nn.Module):
    """Simple feedback processor for testing FeedbackChannelModel."""

    def __init__(self, input_size=3):
        super().__init__()
        self.input_size = input_size
        self.layer = nn.Linear(input_size, 1)

    def forward(self, feedback):
        # Process feedback to produce a scalar gain factor
        return torch.sigmoid(self.layer(feedback))


@pytest.fixture
def feedback_components():
    """Fixture providing components for FeedbackChannelModel testing."""
    encoder = SimpleEncoder()
    forward_channel = AWGNChannel(avg_noise_power=0.1)
    decoder = SimpleDecoder()
    feedback_size = 3
    feedback_generator = SimpleFeedbackGenerator(feedback_size=feedback_size)
    feedback_channel = IdentityChannel()  # Perfect feedback channel for testing
    feedback_processor = SimpleFeedbackProcessor(input_size=feedback_size)

    return {"encoder": encoder, "forward_channel": forward_channel, "decoder": decoder, "feedback_generator": feedback_generator, "feedback_channel": feedback_channel, "feedback_processor": feedback_processor, "max_iterations": 3}


def test_feedback_channel_model_initialization(feedback_components):
    """Test FeedbackChannelModel initialization with all components."""
    model = FeedbackChannelModel(**feedback_components)

    # Check all components are properly assigned
    assert model.encoder == feedback_components["encoder"]
    assert model.forward_channel == feedback_components["forward_channel"]
    assert model.decoder == feedback_components["decoder"]
    assert model.feedback_generator == feedback_components["feedback_generator"]
    assert model.feedback_channel == feedback_components["feedback_channel"]
    assert model.feedback_processor == feedback_components["feedback_processor"]
    assert model.max_iterations == feedback_components["max_iterations"]


def test_feedback_channel_model_single_iteration(feedback_components):
    """Test FeedbackChannelModel with a single iteration."""
    # Set max_iterations to 1
    components = feedback_components.copy()
    components["max_iterations"] = 1

    model = FeedbackChannelModel(**components)

    # Create test input
    batch_size = 4
    input_data = torch.randn(batch_size, 10)

    # Run model
    result = model(input_data)

    # Check output structure
    assert "final_output" in result
    assert "iterations" in result
    assert "feedback_history" in result

    # Check final output shape
    assert result["final_output"].shape == input_data.shape

    # Check iterations
    assert len(result["iterations"]) == 1

    # Check iteration structure
    iteration = result["iterations"][0]
    assert "encoded" in iteration
    assert "received" in iteration
    assert "decoded" in iteration
    assert "feedback" in iteration

    # Check feedback history
    assert len(result["feedback_history"]) == 1
    assert result["feedback_history"][0].shape == (batch_size, components["feedback_processor"].input_size)


def test_feedback_channel_model_multiple_iterations(feedback_components):
    """Test FeedbackChannelModel with multiple iterations."""
    model = FeedbackChannelModel(**feedback_components)

    # Create test input
    batch_size = 4
    input_data = torch.randn(batch_size, 10)

    # Run model
    result = model(input_data)

    # Check output structure
    assert "final_output" in result
    assert "iterations" in result
    assert "feedback_history" in result

    # Check final output shape
    assert result["final_output"].shape == input_data.shape

    # Check iterations
    assert len(result["iterations"]) == feedback_components["max_iterations"]

    # Check feedback history
    assert len(result["feedback_history"]) == feedback_components["max_iterations"]


def test_feedback_channel_model_adaptation(feedback_components):
    """Test that FeedbackChannelModel adapts transmission based on feedback."""
    model = FeedbackChannelModel(**feedback_components)

    # Create test input
    batch_size = 4
    input_data = torch.randn(batch_size, 10)

    # Run model
    result = model(input_data)

    # Get encoded signals from each iteration
    encoded_signals = [iteration["encoded"] for iteration in result["iterations"]]

    # The encoded signals should change after the first iteration due to feedback
    # In our setup, the first iteration has no feedback influence
    if len(encoded_signals) > 1:
        # Calculate differences between consecutive encoded signals
        differences = [torch.mean(torch.abs(encoded_signals[i] - encoded_signals[i - 1])) for i in range(1, len(encoded_signals))]

        # There should be some difference in encoded signals due to adaptation
        for diff in differences:
            assert diff > 0


def test_feedback_channel_model_registry():
    """Test that FeedbackChannelModel is correctly registered in ModelRegistry."""
    # Check registration
    assert "feedback_channel" in ModelRegistry._models

    # Basic components for creating through registry
    encoder = SimpleEncoder()
    forward_channel = AWGNChannel(avg_noise_power=0.1)
    decoder = SimpleDecoder()
    feedback_size = 3
    feedback_generator = SimpleFeedbackGenerator(feedback_size=feedback_size)
    feedback_channel = IdentityChannel()
    feedback_processor = SimpleFeedbackProcessor(input_size=feedback_size)

    # Create model through registry
    model = ModelRegistry.create("feedback_channel", encoder=encoder, forward_channel=forward_channel, decoder=decoder, feedback_generator=feedback_generator, feedback_channel=feedback_channel, feedback_processor=feedback_processor)

    assert isinstance(model, FeedbackChannelModel)
    # Default max_iterations should be 1
    assert model.max_iterations == 1


def test_feedback_channel_model_with_kwargs(feedback_components):
    """Test FeedbackChannelModel with additional keyword arguments."""
    model = FeedbackChannelModel(**feedback_components)

    # Create test input
    batch_size = 4
    input_data = torch.randn(batch_size, 10)

    # Run model with additional kwargs
    result = model(input_data, additional_kwarg="test")

    # Basic checks to ensure the model runs with kwargs
    assert "final_output" in result
    assert "iterations" in result
    assert result["final_output"].shape == input_data.shape


def test_feedback_channel_model_device_compatibility(feedback_components):
    """Test FeedbackChannelModel compatibility with different devices."""
    model = FeedbackChannelModel(**feedback_components)

    # Create test input
    batch_size = 4
    input_data = torch.randn(batch_size, 10)

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


def test_feedback_channel_model_zero_iterations(feedback_components):
    """Test FeedbackChannelModel with zero iterations (edge case)."""
    # Set max_iterations to 0
    components = feedback_components.copy()
    components["max_iterations"] = 0

    model = FeedbackChannelModel(**components)

    # Create test input
    batch_size = 4
    input_data = torch.randn(batch_size, 10)

    # Run model - should still function but produce empty results
    result = model(input_data)

    # Check output structure
    assert "final_output" not in result  # Final output should not exist
    assert "iterations" in result
    assert "feedback_history" in result

    # Iterations and feedback history should be empty
    assert len(result["iterations"]) == 0
    assert len(result["feedback_history"]) == 0

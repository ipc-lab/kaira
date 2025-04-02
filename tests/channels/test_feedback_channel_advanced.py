"""Tests for advanced Feedback Channel model scenarios."""
import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel, RayleighFadingChannel, PhaseNoiseChannel
from kaira.models import FeedbackChannelModel


class EncoderWithMemory(nn.Module):
    """Encoder that incorporates feedback state in its processing."""
    
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.main_layer = nn.Linear(input_dim, output_dim)
        self.state_layer = nn.Linear(output_dim, output_dim)
    
    def forward(self, x, state=None):
        # Main encoding
        encoded = self.main_layer(x)
        
        # Apply feedback state if available
        if state is not None:
            state_effect = self.state_layer(state)
            encoded = encoded + state_effect
            
        return encoded


class AdvancedDecoder(nn.Module):
    """More complex decoder with multiple layers."""
    
    def __init__(self, input_dim=5, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class DetailedFeedbackGenerator(nn.Module):
    """Feedback generator that computes more detailed feedback."""
    
    def __init__(self, input_dim=10, feedback_dim=3):
        super().__init__()
        # Takes both decoded output and original input
        combined_dim = input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 8),
            nn.ReLU(),
            nn.Linear(8, feedback_dim)
        )
    
    def forward(self, decoded, original):
        # Compute error metrics and generate feedback
        combined = torch.cat([decoded, original], dim=1)
        return self.net(combined)


class AdaptiveFeedbackProcessor(nn.Module):
    """Feedback processor with adaptive behavior based on feedback values."""
    
    def __init__(self, feedback_dim=3, output_dim=1):
        super().__init__()
        self.input_size = feedback_dim
        self.net = nn.Sequential(
            nn.Linear(feedback_dim, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim),
            nn.Sigmoid()  # Output between 0 and 1 for scaling
        )
    
    def forward(self, feedback):
        return self.net(feedback)


@pytest.fixture
def feedback_model_components():
    """Fixture providing components for testing advanced feedback channel models."""
    input_dim = 10
    latent_dim = 5
    feedback_dim = 3
    
    encoder = EncoderWithMemory(input_dim=input_dim, output_dim=latent_dim)
    forward_channel = AWGNChannel(snr_db=15)
    decoder = AdvancedDecoder(input_dim=latent_dim, output_dim=input_dim)
    feedback_generator = DetailedFeedbackGenerator(input_dim=input_dim, feedback_dim=feedback_dim)
    feedback_channel = AWGNChannel(snr_db=20)  # Usually feedback channel is better than forward
    feedback_processor = AdaptiveFeedbackProcessor(feedback_dim=feedback_dim)
    
    return {
        "encoder": encoder,
        "forward_channel": forward_channel,
        "decoder": decoder,
        "feedback_generator": feedback_generator,
        "feedback_channel": feedback_channel,
        "feedback_processor": feedback_processor,
        "max_iterations": 3
    }


def test_feedback_model_convergence(feedback_model_components):
    """Test that the feedback model improves reconstruction over iterations."""
    # Create model
    model = FeedbackChannelModel(**feedback_model_components)
    
    # Create test input
    batch_size = 16
    input_dim = 10
    input_data = torch.randn(batch_size, input_dim)
    
    # Run model
    result = model(input_data)
    
    # Extract outputs from each iteration
    iteration_outputs = [iter_result["decoded"] for iter_result in result["iterations"]]
    
    # Calculate error for each iteration
    errors = [torch.mean((output - input_data) ** 2).item() for output in iteration_outputs]
    
    # Check that errors decrease over iterations (allowing for some tolerance)
    # The last iteration should have lower error than the first
    assert errors[-1] < errors[0]
    
    # Check that at least one intermediate step shows improvement
    decreasing_steps = sum(errors[i] > errors[i+1] for i in range(len(errors)-1))
    assert decreasing_steps > 0


def test_feedback_model_with_different_channels(feedback_model_components):
    """Test feedback model with different channel models for forward and feedback paths."""
    # Baseline: both channels are AWGN
    model_baseline = FeedbackChannelModel(**feedback_model_components)
    
    # Test 1: Forward channel with fading, feedback channel AWGN
    components1 = feedback_model_components.copy()
    components1["forward_channel"] = RayleighFadingChannel()
    model1 = FeedbackChannelModel(**components1)
    
    # Test 2: Forward channel with phase noise, feedback channel AWGN
    components2 = feedback_model_components.copy()
    components2["forward_channel"] = PhaseNoiseChannel(phase_noise_std=0.1)
    model2 = FeedbackChannelModel(**components2)
    
    # Test 3: Forward channel AWGN, feedback channel with fading
    components3 = feedback_model_components.copy()
    components3["feedback_channel"] = RayleighFadingChannel()
    model3 = FeedbackChannelModel(**components3)
    
    # Create fixed test input for fair comparison
    torch.manual_seed(42)
    batch_size = 16
    input_dim = 10
    input_data = torch.randn(batch_size, input_dim)
    
    # Run all models
    result_baseline = model_baseline(input_data)
    result1 = model1(input_data)
    result2 = model2(input_data)
    result3 = model3(input_data)
    
    # All models should complete successfully
    assert "final_output" in result_baseline
    assert "final_output" in result1
    assert "final_output" in result2
    assert "final_output" in result3
    
    # The feedback history should be different across models
    for i in range(feedback_model_components["max_iterations"]):
        # Compare feedback history between baseline and other models
        assert not torch.allclose(
            result_baseline["feedback_history"][i], 
            result1["feedback_history"][i]
        )
        assert not torch.allclose(
            result_baseline["feedback_history"][i], 
            result2["feedback_history"][i]
        )
        assert not torch.allclose(
            result_baseline["feedback_history"][i], 
            result3["feedback_history"][i]
        )


def test_feedback_model_training_compatibility(feedback_model_components):
    """Test that the feedback channel model can be trained with backpropagation."""
    # Create model with fewer iterations for faster training
    components = feedback_model_components.copy()
    components["max_iterations"] = 2
    model = FeedbackChannelModel(**components)
    
    # Create test input
    batch_size = 16
    input_dim = 10
    input_data = torch.randn(batch_size, input_dim)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Initial loss (MSE between final output and input)
    result = model(input_data)
    initial_loss = torch.mean((result["final_output"] - input_data) ** 2)
    
    # Run a few optimization steps
    for _ in range(5):
        optimizer.zero_grad()
        result = model(input_data)
        loss = torch.mean((result["final_output"] - input_data) ** 2)
        loss.backward()
        optimizer.step()
    
    # Final loss
    result = model(input_data)
    final_loss = torch.mean((result["final_output"] - input_data) ** 2)
    
    # Loss should decrease
    assert final_loss < initial_loss


def test_feedback_model_with_custom_iteration_count(feedback_model_components):
    """Test feedback model with different numbers of iterations."""
    # Create base input
    batch_size = 8
    input_dim = 10
    input_data = torch.randn(batch_size, input_dim)
    
    # Test with 1, 3, and 5 iterations
    for iterations in [1, 3, 5]:
        components = feedback_model_components.copy()
        components["max_iterations"] = iterations
        model = FeedbackChannelModel(**components)
        
        # Run model
        result = model(input_data)
        
        # Check that we get the correct number of iterations
        assert len(result["iterations"]) == iterations
        assert len(result["feedback_history"]) == iterations


def test_feedback_model_with_batch_variation(feedback_model_components):
    """Test feedback model with varying batch sizes."""
    # Create model
    model = FeedbackChannelModel(**feedback_model_components)
    input_dim = 10
    
    # Test with different batch sizes
    for batch_size in [1, 8, 32]:
        # Create input
        input_data = torch.randn(batch_size, input_dim)
        
        # Run model
        result = model(input_data)
        
        # Check output shapes
        assert result["final_output"].shape == (batch_size, input_dim)
        
        # Check iteration results
        for iteration in result["iterations"]:
            assert iteration["encoded"].shape[0] == batch_size
            assert iteration["received"].shape[0] == batch_size
            assert iteration["decoded"].shape == (batch_size, input_dim)
            assert iteration["feedback"].shape[0] == batch_size


def test_explicit_feedback_state_changes(feedback_model_components):
    """Test that the feedback state changes correctly over iterations."""
    # Create model
    model = FeedbackChannelModel(**feedback_model_components)
    
    # Create input
    batch_size = 8
    input_dim = 10
    input_data = torch.randn(batch_size, input_dim)
    
    # Run model
    result = model(input_data)
    
    # Extract encoded outputs from each iteration to see adaptation
    encoded_signals = [iter_result["encoded"] for iter_result in result["iterations"]]
    
    # First iteration has no feedback effect
    # After that, encoded signals should change due to feedback
    for i in range(1, len(encoded_signals)):
        # Check that encoded signals differ between iterations
        assert not torch.allclose(encoded_signals[i-1], encoded_signals[i])
        
        # We expect difference due to the feedback adaptation
        diff = torch.norm(encoded_signals[i] - encoded_signals[i-1]).item()
        assert diff > 0.01  # Ensure meaningful difference
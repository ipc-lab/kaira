import pytest
import torch
from kaira.channels.lambda_channel import LambdaChannel, PerfectChannel

@pytest.fixture
def random_tensor():
    """Fixture providing a random tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(4, 100)

def test_lambda_channel():
    """Test LambdaChannel with a simple doubling function."""
    # Define a simple function that doubles the input
    def double_fn(x):
        return 2 * x

    # Create LambdaChannel with the doubling function
    channel = LambdaChannel(fn=double_fn)

    # Apply channel to input
    input_tensor = random_tensor()
    output = channel(input_tensor)

    # Check output shape matches input
    assert output.shape == input_tensor.shape

    # Check output values are double the input values
    assert torch.allclose(output, 2 * input_tensor)

def test_lambda_channel_complex():
    """Test LambdaChannel with a complex function."""
    # Define a function that adds a constant complex value
    def add_complex_fn(x):
        return x + torch.complex(torch.tensor(1.0), torch.tensor(1.0))

    # Create LambdaChannel with the complex function
    channel = LambdaChannel(fn=add_complex_fn)

    # Apply channel to input
    input_tensor = torch.complex(random_tensor(), random_tensor())
    output = channel(input_tensor)

    # Check output shape matches input
    assert output.shape == input_tensor.shape

    # Check output values are correctly transformed
    expected_output = input_tensor + torch.complex(torch.tensor(1.0), torch.tensor(1.0))
    assert torch.allclose(output, expected_output)

def test_perfect_channel():
    """Test PerfectChannel (identity channel)."""
    # Create PerfectChannel
    channel = PerfectChannel()

    # Apply channel to input
    input_tensor = random_tensor()
    output = channel(input_tensor)

    # Check output shape matches input
    assert output.shape == input_tensor.shape

    # Check output values are identical to input values
    assert torch.allclose(output, input_tensor)

# tests/test_channels_digital.py
import pytest
import torch

from kaira.channels import (
    BinaryErasureChannel,
    BinarySymmetricChannel,
    BinaryZChannel,
    ChannelRegistry,
)


@pytest.fixture
def binary_input():
    """Fixture providing a binary tensor for testing."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (1000,)).float()


@pytest.fixture
def bipolar_input():
    """Fixture providing a bipolar tensor {-1, 1} for testing."""
    torch.manual_seed(42)
    return 2 * torch.randint(0, 2, (1000,)).float() - 1


@pytest.mark.parametrize("crossover_prob", [0.0, 0.1, 0.5])
def test_binary_symmetric_channel(binary_input, crossover_prob):
    """Test BinarySymmetricChannel with different crossover probabilities."""
    torch.manual_seed(42)
    # Initialize channel
    channel = BinarySymmetricChannel(crossover_prob=crossover_prob)

    # Process through channel
    output = channel(binary_input)

    # Check output dimensions match input
    assert output.shape == binary_input.shape

    # Check output values remain binary (0 or 1)
    assert torch.all((output == 0) | (output == 1))

    # Calculate bit error rate
    bit_errors = (output != binary_input).float().mean().item()

    # For large samples, bit error rate should be close to crossover probability
    assert abs(bit_errors - crossover_prob) < 0.03  # Allow small statistical variation


@pytest.mark.parametrize("crossover_prob", [0.0, 0.1, 0.5])
def test_binary_symmetric_channel_bipolar(bipolar_input, crossover_prob):
    """Test BinarySymmetricChannel with bipolar {-1, 1} inputs."""
    torch.manual_seed(42)
    # Initialize channel
    channel = BinarySymmetricChannel(crossover_prob=crossover_prob)

    # Process through channel
    output = channel(bipolar_input)

    # Check output dimensions match input
    assert output.shape == bipolar_input.shape

    # Check output values remain bipolar (-1 or 1)
    assert torch.all((output == -1) | (output == 1))

    # Calculate bit error rate
    bit_errors = (output != bipolar_input).float().mean().item()

    # For large samples, bit error rate should be close to crossover probability
    assert abs(bit_errors - crossover_prob) < 0.03  # Allow small statistical variation


@pytest.mark.parametrize("erasure_prob", [0.0, 0.2, 0.7])
def test_binary_erasure_channel(binary_input, erasure_prob):
    """Test BinaryErasureChannel with different erasure probabilities."""
    torch.manual_seed(42)
    # Initialize channel
    erasure_symbol = -1
    channel = BinaryErasureChannel(erasure_prob=erasure_prob, erasure_symbol=erasure_symbol)

    # Process through channel
    output = channel(binary_input)

    # Check output dimensions match input
    assert output.shape == binary_input.shape

    # Check output values are either original binary values or erasure symbol
    assert torch.all((output == 0) | (output == 1) | (output == erasure_symbol))

    # Calculate erasure rate
    erasures = (output == erasure_symbol).float().mean().item()

    # For large samples, erasure rate should be close to erasure probability
    assert abs(erasures - erasure_prob) < 0.03  # Allow small statistical variation


@pytest.mark.parametrize("error_prob", [0.0, 0.3, 0.8])
def test_binary_z_channel(binary_input, error_prob):
    """Test BinaryZChannel with different error probabilities."""
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize channel
    channel = BinaryZChannel(error_prob=error_prob)

    # Create a test input with all ones to ensure we can measure error rate
    test_input = torch.ones_like(binary_input)

    # Process through channel
    output = channel(test_input)

    # Check output dimensions match input
    assert output.shape == test_input.shape

    # Check output values remain binary (0 or 1)
    assert torch.all((output == 0) | (output == 1))

    # For all ones input, the error rate equals the proportion of zeros in the output
    if error_prob > 0:
        errors = (output == 0).float().mean().item()
        # Error rate should be close to error probability
        assert abs(errors - error_prob) < 0.05
    else:
        # With zero error probability, no bits should flip
        assert torch.all(output == 1)


@pytest.mark.parametrize("error_prob", [0.0, 0.3, 0.8])
def test_binary_z_channel_bipolar(bipolar_input, error_prob):
    """Test BinaryZChannel with bipolar {-1, 1} inputs."""
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize channel
    channel = BinaryZChannel(error_prob=error_prob)

    # Create a test input with all ones to ensure we can measure error rate
    test_input = torch.ones_like(bipolar_input)

    # Process through channel
    output = channel(test_input)

    # Check output dimensions match input
    assert output.shape == test_input.shape

    # Check output values - in bipolar representation, 1 can become -1 or stay 1
    # (1→0 in binary becomes 1→-1 in bipolar)
    assert torch.all((output == -1) | (output == 1) | (output == 0))

    # For all ones input, the error rate equals the proportion of -1s in the output
    if error_prob > 0:
        errors = (output != 1).float().mean().item()
        # Error rate should be close to error probability
        assert abs(errors - error_prob) < 0.05
    else:
        # With zero error probability, no bits should flip
        assert torch.all(output == 1)


def test_invalid_parameters():
    """Test that invalid channel parameters raise appropriate errors."""
    # Test invalid crossover probability
    with pytest.raises(ValueError):
        BinarySymmetricChannel(crossover_prob=1.5)

    # Test invalid erasure probability
    with pytest.raises(ValueError):
        BinaryErasureChannel(erasure_prob=-0.2)

    # Test invalid error probability
    with pytest.raises(ValueError):
        BinaryZChannel(error_prob=2.0)


def test_channel_registry():
    """Test that digital channels are correctly registered."""
    torch.manual_seed(42)
    # Check that channels are registered with their correct names (lowercase class names)
    assert "binarysymmetricchannel" in ChannelRegistry._channels
    assert "binaryerasurechannel" in ChannelRegistry._channels
    assert "binaryzchannel" in ChannelRegistry._channels

    # Test creating channels through registry
    bsc = ChannelRegistry.create("binarysymmetricchannel", crossover_prob=0.1)
    assert isinstance(bsc, BinarySymmetricChannel)
    assert abs(bsc.crossover_prob.item() - 0.1) < 1e-6  # Use approximate comparison for float

    bec = ChannelRegistry.create("binaryerasurechannel", erasure_prob=0.2)
    assert isinstance(bec, BinaryErasureChannel)
    assert abs(bec.erasure_prob.item() - 0.2) < 1e-6  # Use approximate comparison for float

    bz = ChannelRegistry.create("binaryzchannel", error_prob=0.3)
    assert isinstance(bz, BinaryZChannel)
    assert abs(bz.error_prob.item() - 0.3) < 1e-6  # Use approximate comparison for float


def test_binary_z_channel_neg_one_format():
    """Test BinaryZChannel with {-1, 1} input format."""
    torch.manual_seed(42)
    input_tensor = torch.tensor([-1, 1, -1, 1], dtype=torch.float32)
    channel = BinaryZChannel(error_prob=0.5)
    output = channel(input_tensor)
    assert torch.all((output == -1) | (output == 1))


def test_binary_z_channel_no_errors():
    """Test BinaryZChannel with error probability of 0."""
    torch.manual_seed(42)
    input_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    channel = BinaryZChannel(error_prob=0.0)
    output = channel(input_tensor)
    assert torch.equal(input_tensor, output)

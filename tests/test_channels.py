# tests/test_channels.py
import pytest
import torch

from kaira.channels import AWGNChannel, PerfectChannel


def test_awgn_channel_initialization():
    """Test AWGN channel initialization with valid parameters."""
    avg_noise_power = 0.1
    channel = AWGNChannel(avg_noise_power=avg_noise_power)
    assert channel.avg_noise_power == torch.tensor(avg_noise_power)


@pytest.mark.parametrize("avg_noise_power", [0.1, 0.5, 1.0])
def test_awgn_channel_forward(random_tensor, avg_noise_power):
    """Test AWGN channel forward pass with different noise powers."""
    channel = AWGNChannel(avg_noise_power=avg_noise_power)
    output = channel(random_tensor)

    # Check output shape matches input
    assert output.shape == random_tensor.shape

    # Check noise has been added (output should be different from input)
    assert not torch.allclose(output, random_tensor)

    # Check noise variance is approximately as expected
    noise = output - random_tensor
    measured_variance = torch.var(noise)
    assert torch.isclose(measured_variance, torch.tensor(avg_noise_power), rtol=0.1)


def test_awgn_channel_complex():
    """Test AWGN channel with complex inputs."""
    avg_noise_power = 0.1
    channel = AWGNChannel(avg_noise_power=avg_noise_power)
    x = torch.complex(torch.randn(4, 2, 32, 32), torch.randn(4, 2, 32, 32))
    output = channel(x)

    # Check output shape and type
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.is_complex(output)

    # Check noise variance for real and imaginary parts
    noise = output - x
    real_variance = torch.var(noise.real)
    imag_variance = torch.var(noise.imag)
    assert torch.isclose(real_variance, torch.tensor(avg_noise_power/2), rtol=0.1)
    assert torch.isclose(imag_variance, torch.tensor(avg_noise_power/2), rtol=0.1)


def test_perfect_channel():
    """Test perfect channel (should pass through input unchanged)."""
    channel = PerfectChannel()
    x = torch.randn(4, 3, 32, 32)
    output = channel(x)

    assert torch.all(output == x)

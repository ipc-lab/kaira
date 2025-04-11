# tests/conftest.py
import pytest
import torch


@pytest.fixture
def random_tensor():
    """Fixture that provides a random tensor for testing."""
    return torch.randn(4, 3, 32, 32)  # Common image-like tensor shape


@pytest.fixture
def binary_tensor():
    """Fixture that provides a binary (0 or 1) tensor for testing digital channels."""
    torch.manual_seed(42)  # For reproducibility
    return torch.randint(0, 2, (1000,))  # 1000 random binary values


@pytest.fixture
def bipolar_tensor():
    """Fixture that provides a bipolar (-1 or 1) tensor for testing digital channels."""
    torch.manual_seed(42)  # For reproducibility
    return torch.randint(0, 2, (1000,)) * 2 - 1  # 1000 random values of -1 or 1


@pytest.fixture
def complex_tensor():
    """Fixture that provides a complex tensor for testing channels with complex inputs."""
    torch.manual_seed(42)  # For reproducibility
    real_part = torch.randn(1000)
    imag_part = torch.randn(1000)
    return torch.complex(real_part, imag_part)


@pytest.fixture
def device():
    """Fixture that provides the compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def snr_values():
    """Fixture providing common SNR test values."""
    return [-10.0, 0.0, 10.0, 20.0]

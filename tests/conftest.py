# tests/conftest.py
import pytest
import torch


@pytest.fixture
def random_tensor():
    """Fixture that provides a random tensor for testing."""
    return torch.randn(4, 3, 32, 32)  # Common image-like tensor shape


@pytest.fixture
def device():
    """Fixture that provides the compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def snr_values():
    """Fixture providing common SNR test values."""
    return [-10.0, 0.0, 10.0, 20.0]

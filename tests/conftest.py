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
def binary_source():
    """Fixture providing binary source data for testing."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (100,), dtype=torch.float32)


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
def continuous_source():
    """Fixture that provides a continuous-valued tensor (e.g., Gaussian)."""
    torch.manual_seed(43)  # Different seed
    return torch.randn(2000)  # Larger 1D tensor


@pytest.fixture
def large_binary_source():
    """Fixture that provides a large binary (0 or 1) tensor for statistical tests."""
    torch.manual_seed(44)  # Different seed
    return torch.randint(0, 2, (10000,))  # 10000 random binary values


@pytest.fixture
def multidimensional_source():
    """Fixture that provides a multi-dimensional tensor."""
    torch.manual_seed(45)  # Different seed
    return torch.randn(10, 5, 5)  # Example: Batch of 10, 5x5 tensors


@pytest.fixture
def device():
    """Fixture that provides the compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def snr_values():
    """Fixture providing common SNR test values."""
    return [-10.0, 0.0, 10.0, 20.0]

"""Common fixtures for metrics tests."""
import pytest
import torch


@pytest.fixture
def binary_data():
    """Fixture providing test data with known errors."""
    torch.manual_seed(42)
    n_bits = 1000

    # Create true bits
    true_bits = torch.randint(0, 2, (1, n_bits)).float()

    # Create received bits with some errors
    error_mask = torch.rand(1, n_bits) < 0.1  # 10% bit error rate
    received_bits = torch.logical_xor(true_bits, error_mask).float()

    return true_bits, received_bits


@pytest.fixture
def random_binary_data():
    """Fixture providing random binary data for testing with batches."""
    torch.manual_seed(42)
    n_bits = 1000

    # Create true bits
    true_bits = torch.randint(0, 2, (2, n_bits)).float()

    # Create received bits with some errors
    error_mask = torch.rand(2, n_bits) < 0.1  # 10% bit error rate
    received_bits = torch.logical_xor(true_bits, error_mask).float()

    return true_bits, received_bits


@pytest.fixture
def signal_data():
    """Fixture for creating sample signal data for SNR tests."""
    torch.manual_seed(42)
    signal = torch.randn(1, 1000)  # Original signal
    noise = 0.1 * torch.randn(1, 1000)  # Noise
    noisy_signal = signal + noise  # Noisy signal
    return signal, noisy_signal


@pytest.fixture
def sample_preds():
    """Fixture for creating sample predictions tensor for image metrics."""
    # Increased size to accommodate multi-scale operations (at least 256x256)
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_targets():
    """Fixture for creating sample targets tensor for image metrics."""
    # Increased size to accommodate multi-scale operations (at least 256x256)
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_images():
    """Fixture for creating sample image pairs."""
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    return img1, img2

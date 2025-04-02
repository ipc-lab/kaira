import torch
import pytest
import numpy as np
from kaira.channels.base import BaseChannel


class DummyChannel(BaseChannel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def random_tensor():
    """Fixture providing a random tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(4, 100)


@pytest.fixture
def complex_tensor():
    """Fixture providing a random complex tensor for testing."""
    torch.manual_seed(42)
    real = torch.randn(4, 100)
    imag = torch.randn(4, 100)
    return torch.complex(real, imag)


@pytest.fixture
def binary_tensor():
    """Fixture providing a binary tensor for testing."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (1000,)).float()


@pytest.fixture
def bipolar_tensor():
    """Fixture providing a bipolar tensor {-1, 1} for testing."""
    torch.manual_seed(42)
    return 2 * torch.randint(0, 2, (1000,)).float() - 1


def test_base_channel_get_config():
    channel = DummyChannel()
    config = channel.get_config()
    assert isinstance(config, dict)
    assert "training" in config
    assert "avg_noise_power" not in config


def test_base_channel_forward():
    channel = DummyChannel()
    x = torch.randn(10, 10)
    y = channel(x)
    assert torch.equal(x, y)

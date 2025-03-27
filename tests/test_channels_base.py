import pytest
import torch
from kaira.channels.base import BaseChannel

class DummyChannel(BaseChannel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

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

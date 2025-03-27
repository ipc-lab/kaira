import pytest
import torch
from typing import Optional, Union
from kaira.modulations.base import BaseModulator, BaseDemodulator

class TestModulator(BaseModulator):
    """Test implementation of BaseModulator."""
    def __init__(self):
        super().__init__(bits_per_symbol=2)

    @property
    def bits_per_symbol(self):
        """Return the number of bits per symbol."""
        return self._bits_per_symbol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that simply returns the input."""
        return x

class TestDemodulator(BaseDemodulator):
    """Test implementation of BaseDemodulator."""
    def __init__(self):
        super().__init__(bits_per_symbol=2)

    @property
    def bits_per_symbol(self):
        """Return the number of bits per symbol."""
        return self._bits_per_symbol

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass that simply returns the input."""
        return y

@pytest.fixture
def modulator():
    """Fixture providing a test modulator instance."""
    return TestModulator()

@pytest.fixture
def demodulator():
    """Fixture providing a test demodulator instance."""
    return TestDemodulator()

def test_modulator_reset_state(modulator):
    """Test that reset_state method doesn't raise errors."""
    modulator.reset_state()  # Should do nothing and not raise an error

def test_demodulator_reset_state(demodulator):
    """Test that reset_state method doesn't raise errors."""
    demodulator.reset_state()  # Should do nothing and not raise an error

def test_modulator_bits_per_symbol(modulator):
    """Test bits_per_symbol property of modulator."""
    assert modulator.bits_per_symbol == 2
    
    modulator._bits_per_symbol = None
    with pytest.raises(NotImplementedError):
        _ = modulator.bits_per_symbol

def test_demodulator_bits_per_symbol(demodulator):
    """Test bits_per_symbol property of demodulator."""
    assert demodulator.bits_per_symbol == 2

    demodulator._bits_per_symbol = None
    with pytest.raises(NotImplementedError):
        _ = demodulator.bits_per_symbol

def test_modulator_forward(modulator):
    """Test forward method of modulator."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = modulator(x)
    assert torch.equal(result, x)

def test_demodulator_forward(demodulator):
    """Test forward method of demodulator."""
    y = torch.tensor([1.0, 2.0, 3.0])
    result = demodulator(y)
    assert torch.equal(result, y)
    
    # Test with noise variance
    noise_var = 0.1
    result_with_noise = demodulator(y, noise_var)
    assert torch.equal(result_with_noise, y)

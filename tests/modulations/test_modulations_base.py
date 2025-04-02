from typing import Optional, Union

import pytest
import torch

from kaira.modulations.base import BaseDemodulator, BaseModulator


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


class ConcreteModulator(BaseModulator):
    """Concrete implementation of BaseModulator for testing."""

    @property
    def bits_per_symbol(self) -> int:
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConcreteDemodulator(BaseDemodulator):
    """Concrete implementation of BaseDemodulator for testing."""

    @property
    def bits_per_symbol(self) -> int:
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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


def test_base_modulator_abstract_methods():
    """Test that BaseModulator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseModulator()


def test_base_demodulator_abstract_methods():
    """Test that BaseDemodulator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseDemodulator()


def test_reset_state():
    """Test the reset_state method of BaseModulator."""
    mod = ConcreteModulator()
    mod.reset_state()  # This should not raise an error

    demod = ConcreteDemodulator()
    demod.reset_state()  # This should not raise an error


def test_base_modulator_concrete_subclass():
    """Test that a concrete subclass of BaseModulator can be instantiated."""
    mod = ConcreteModulator()
    assert mod.bits_per_symbol == 2

    x = torch.tensor([0, 1, 2, 3])
    y = mod(x)
    assert torch.equal(x, y)


def test_base_demodulator_concrete_subclass():
    """Test that a concrete subclass of BaseDemodulator can be instantiated."""
    demod = ConcreteDemodulator()
    assert demod.bits_per_symbol == 2

    x = torch.tensor([0, 1, 2, 3])
    y = demod(x)
    assert torch.equal(x, y)

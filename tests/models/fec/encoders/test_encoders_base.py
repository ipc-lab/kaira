"""Tests for the base encoder classes in kaira.models.fec.encoders package."""

from typing import Any, Tuple, Union

import pytest
import torch

from kaira.models.fec.encoders.base import BaseBlockCodeEncoder


class MinimalBlockCodeEncoder(BaseBlockCodeEncoder):
    """Minimal implementation of BaseBlockCodeEncoder for testing purposes.

    Implements just enough functionality to instantiate the abstract class.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Implement abstract method."""
        return x

    def inverse_encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Implement abstract method."""
        return x

    def calculate_syndrome(self, x: torch.Tensor) -> torch.Tensor:
        """Implement abstract method."""
        return torch.zeros(x.shape[:-1] + (1,))


class IncompleteBlockCodeEncoder(BaseBlockCodeEncoder):
    """Incomplete implementation of BaseBlockCodeEncoder for testing abstract method
    enforcement."""

    pass


class TestBaseBlockCodeEncoder:
    """Test suite for BaseBlockCodeEncoder class."""

    def test_initialization(self):
        """Test initialization of BaseBlockCodeEncoder with valid parameters."""
        # Test with valid parameters
        encoder = MinimalBlockCodeEncoder(code_length=7, code_dimension=4)
        assert encoder.code_length == 7
        assert encoder.code_dimension == 4
        assert encoder.redundancy == 3
        assert encoder.parity_bits == 3
        assert encoder.code_rate == 4 / 7

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with non-positive code length
        with pytest.raises(ValueError, match="Code length must be positive"):
            MinimalBlockCodeEncoder(code_length=0, code_dimension=4)

        # Test with non-positive code dimension
        with pytest.raises(ValueError, match="Code dimension must be positive"):
            MinimalBlockCodeEncoder(code_length=7, code_dimension=0)

        # Test with code dimension > code length
        with pytest.raises(ValueError, match="Code dimension .* must not exceed code length"):
            MinimalBlockCodeEncoder(code_length=4, code_dimension=7)

    def test_abstract_method_enforcement(self):
        """Test that abstract methods are enforced."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteBlockCodeEncoder(code_length=7, code_dimension=4)

    def test_properties(self):
        """Test property getters."""
        encoder = MinimalBlockCodeEncoder(code_length=8, code_dimension=4)

        # Test all properties
        assert encoder.code_length == 8
        assert encoder.code_dimension == 4
        assert encoder.redundancy == 4
        assert encoder.parity_bits == 4
        assert encoder.code_rate == 0.5

        # Verify redundancy and parity_bits are the same
        assert encoder.redundancy == encoder.parity_bits

        # Test code_rate formula
        assert encoder.code_rate == encoder.code_dimension / encoder.code_length

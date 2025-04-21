"""Tests for the base decoder classes in kaira.models.fec.decoders package."""

from typing import Any, Tuple, Union

import pytest
import torch

from kaira.models.fec.decoders.base import BaseBlockDecoder
from kaira.models.fec.encoders.base import BaseBlockCodeEncoder


class MinimalBlockCodeEncoder(BaseBlockCodeEncoder):
    """Minimal implementation of BaseBlockCodeEncoder for testing purposes."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Implement abstract method."""
        return x

    def inverse_encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Implement abstract method."""
        return x

    def calculate_syndrome(self, x: torch.Tensor) -> torch.Tensor:
        """Implement abstract method."""
        return torch.zeros(x.shape[:-1] + (1,))


class MinimalBlockDecoder(BaseBlockDecoder[MinimalBlockCodeEncoder]):
    """Minimal implementation of BaseBlockDecoder for testing purposes."""

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Implement abstract method."""
        return received


class IncompleteBlockDecoder(BaseBlockDecoder[MinimalBlockCodeEncoder]):
    """Incomplete implementation of BaseBlockDecoder for testing abstract method enforcement."""

    pass


class TestBaseBlockDecoder:
    """Test suite for BaseBlockDecoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Create an encoder instance
        encoder = MinimalBlockCodeEncoder(code_length=7, code_dimension=4)

        # Initialize the decoder with this encoder
        decoder = MinimalBlockDecoder(encoder=encoder)

        # Verify that encoder is stored
        assert decoder.encoder is encoder

        # Verify properties are forwarded to the encoder
        assert decoder.code_length == 7
        assert decoder.code_dimension == 4
        assert decoder.redundancy == 3
        assert decoder.code_rate == 4 / 7

    def test_abstract_method_enforcement(self):
        """Test that abstract methods are enforced."""
        encoder = MinimalBlockCodeEncoder(code_length=7, code_dimension=4)

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteBlockDecoder(encoder=encoder)

    def test_properties(self):
        """Test property delegation to encoder."""
        # Create an encoder instance with different parameters
        encoder = MinimalBlockCodeEncoder(code_length=8, code_dimension=4)
        decoder = MinimalBlockDecoder(encoder=encoder)

        # Test all properties are correctly delegated
        assert decoder.code_length == encoder.code_length
        assert decoder.code_dimension == encoder.code_dimension
        assert decoder.redundancy == encoder.redundancy
        assert decoder.code_rate == encoder.code_rate

    def test_forward_method(self):
        """Test the forward method of the minimal implementation."""
        encoder = MinimalBlockCodeEncoder(code_length=7, code_dimension=4)
        decoder = MinimalBlockDecoder(encoder=encoder)

        # Test with a simple tensor
        received = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        decoded = decoder(received)

        # MinimalBlockDecoder just returns the input
        assert torch.all(decoded == received)

        # Test with batch dimension
        received_batch = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
        decoded_batch = decoder(received_batch)

        assert torch.all(decoded_batch == received_batch)

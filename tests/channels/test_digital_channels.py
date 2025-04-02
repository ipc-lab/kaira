"""Tests for digital channel implementations in Kaira."""
import pytest
import torch
import numpy as np
from kaira.channels import (
    BinarySymmetricChannel,
    BinaryErasureChannel,
    BinaryZChannel
)
from .test_channels_base import binary_tensor, bipolar_tensor


class TestBinarySymmetricChannel:
    """Test suite for BinarySymmetricChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Valid initialization
        channel = BinarySymmetricChannel(crossover_prob=0.1)
        assert channel.crossover_prob.item() == 0.1
        
        # Invalid probability
        with pytest.raises(ValueError):
            BinarySymmetricChannel(crossover_prob=1.5)
    
    @pytest.mark.parametrize("crossover_prob", [0.0, 0.1, 0.5])
    def test_binary_format(self, binary_tensor, crossover_prob):
        """Test with binary {0, 1} input."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        channel = BinarySymmetricChannel(crossover_prob=crossover_prob)
        output = channel(binary_tensor)
        
        # Check shape preservation
        assert output.shape == binary_tensor.shape
        
        # Check output values are binary
        assert torch.all((output == 0) | (output == 1))
        
        # Calculate bit error rate
        bit_errors = (output != binary_tensor).float().mean().item()
        
        # Should be close to crossover probability
        assert abs(bit_errors - crossover_prob) < 0.03
    
    @pytest.mark.parametrize("crossover_prob", [0.0, 0.1, 0.5])
    def test_bipolar_format(self, bipolar_tensor, crossover_prob):
        """Test with bipolar {-1, 1} input."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        channel = BinarySymmetricChannel(crossover_prob=crossover_prob)
        output = channel(bipolar_tensor)
        
        # Check shape preservation
        assert output.shape == bipolar_tensor.shape
        
        # Check output values are bipolar
        assert torch.all((output == -1) | (output == 1))
        
        # Calculate bit error rate
        bit_errors = (output != bipolar_tensor).float().mean().item()
        
        # Should be close to crossover probability
        assert abs(bit_errors - crossover_prob) < 0.03


class TestBinaryErasureChannel:
    """Test suite for BinaryErasureChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default erasure symbol
        channel1 = BinaryErasureChannel(erasure_prob=0.2)
        assert channel1.erasure_prob.item() == 0.2
        assert channel1.erasure_symbol == -1
        
        # Custom erasure symbol
        channel2 = BinaryErasureChannel(erasure_prob=0.3, erasure_symbol=2)
        assert channel2.erasure_prob.item() == 0.3
        assert channel2.erasure_symbol == 2
        
        # Invalid probability
        with pytest.raises(ValueError):
            BinaryErasureChannel(erasure_prob=-0.1)
    
    @pytest.mark.parametrize("erasure_prob", [0.0, 0.2, 0.7])
    def test_binary_format(self, binary_tensor, erasure_prob):
        """Test with binary {0, 1} input."""
        torch.manual_seed(42)
        channel = BinaryErasureChannel(erasure_prob=erasure_prob, erasure_symbol=-1)
        output = channel(binary_tensor)
        
        # Check shape preservation
        assert output.shape == binary_tensor.shape
        
        # Output values should be 0, 1, or erasure_symbol
        assert torch.all((output == 0) | (output == 1) | (output == -1))
        
        # Calculate erasure rate
        erasure_rate = (output == -1).float().mean().item()
        
        # Should be close to erasure probability
        assert abs(erasure_rate - erasure_prob) < 0.03
    
    def test_custom_erasure_symbol(self, binary_tensor):
        """Test with a custom erasure symbol."""
        erasure_prob = 0.3
        custom_symbol = 2
        torch.manual_seed(42)
        channel = BinaryErasureChannel(erasure_prob=erasure_prob, erasure_symbol=custom_symbol)
        output = channel(binary_tensor)
        
        # Calculate erasure rate with custom symbol
        erasure_rate = (output == custom_symbol).float().mean().item()
        
        # Should be close to erasure probability
        assert abs(erasure_rate - erasure_prob) < 0.03
    
    def test_bipolar_format(self, bipolar_tensor):
        """Test with bipolar {-1, 1} input and a different erasure symbol."""
        erasure_prob = 0.3
        erasure_symbol = 0  # Use 0 as erasure for bipolar input
        torch.manual_seed(42)
        channel = BinaryErasureChannel(erasure_prob=erasure_prob, erasure_symbol=erasure_symbol)
        output = channel(bipolar_tensor)
        
        # Check output values
        assert torch.all((output == -1) | (output == 1) | (output == erasure_symbol))
        
        # Calculate erasure rate
        erasure_rate = (output == erasure_symbol).float().mean().item()
        
        # Should be close to erasure probability
        assert abs(erasure_rate - erasure_prob) < 0.03


class TestBinaryZChannel:
    """Test suite for BinaryZChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        channel = BinaryZChannel(error_prob=0.2)
        assert channel.error_prob.item() == 0.2
        
        # Invalid probability
        with pytest.raises(ValueError):
            BinaryZChannel(error_prob=1.5)
    
    def test_binary_format(self, binary_tensor):
        """Test with binary {0, 1} input."""
        error_prob = 0.3
        torch.manual_seed(42)
        channel = BinaryZChannel(error_prob=error_prob)
        output = channel(binary_tensor)
        
        # Check shape preservation
        assert output.shape == binary_tensor.shape
        
        # Output should remain binary
        assert torch.all((output == 0) | (output == 1))
        
        # In Z-channel, only 1→0 errors occur
        # Find positions where input is 1
        ones_mask = binary_tensor == 1
        zeros_mask = binary_tensor == 0
        
        # Calculate error rate for 1→0
        if ones_mask.any():
            ones_error_rate = (output[ones_mask] == 0).float().mean().item()
            # Should be close to error probability
            assert abs(ones_error_rate - error_prob) < 0.05
        
        # Verify no 0→1 errors
        if zeros_mask.any():
            assert torch.all(output[zeros_mask] == 0)
    
    def test_bipolar_format(self, bipolar_tensor):
        """Test with bipolar {-1, 1} input."""
        error_prob = 0.3
        torch.manual_seed(42)
        channel = BinaryZChannel(error_prob=error_prob)
        output = channel(bipolar_tensor)
        
        # Check shape preservation
        assert output.shape == bipolar_tensor.shape
        
        # Output should remain bipolar
        assert torch.all((output == -1) | (output == 1))
        
        # In Z-channel with bipolar format, only 1→-1 errors occur
        # Find positions where input is 1
        ones_mask = bipolar_tensor == 1
        neg_ones_mask = bipolar_tensor == -1
        
        # Calculate error rate for 1→-1
        if ones_mask.any():
            ones_error_rate = (output[ones_mask] == -1).float().mean().item()
            # Should be close to error probability
            assert abs(ones_error_rate - error_prob) < 0.05
        
        # Verify no -1→1 errors
        if neg_ones_mask.any():
            assert torch.all(output[neg_ones_mask] == -1)


def test_channel_registry():
    """Test that all digital channels are properly registered."""
    from kaira.channels.registry import ChannelRegistry
    
    # Check that all channel classes can be retrieved from registry
    bsc_class = ChannelRegistry.get("binarysymmetricchannel")
    bec_class = ChannelRegistry.get("binaryerasurechannel")
    bz_class = ChannelRegistry.get("binaryzchannel")
    
    assert bsc_class is BinarySymmetricChannel
    assert bec_class is BinaryErasureChannel
    assert bz_class is BinaryZChannel
    
    # Test creation through registry
    bsc = ChannelRegistry.create("binarysymmetricchannel", crossover_prob=0.1)
    assert isinstance(bsc, BinarySymmetricChannel)
    assert bsc.crossover_prob.item() == 0.1
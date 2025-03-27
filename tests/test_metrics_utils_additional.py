import pytest
import torch
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import tempfile
import os
from kaira.metrics.utils import (
    check_same_shape, benchmark_metrics
)
@pytest.fixture
def sample_data():
    """Fixture for creating sample data tensors."""
    return torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)

def test_check_same_shape_with_different_shapes():
    """Test check_same_shape with tensors of different shapes."""
    # Create tensors with different shapes
    tensor1 = torch.randn(2, 3, 64, 64)
    tensor2 = torch.randn(2, 3, 32, 32)
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        check_same_shape(tensor1, tensor2)
    
    # Different batch sizes
    tensor3 = torch.randn(4, 3, 64, 64)
    with pytest.raises(ValueError):
        check_same_shape(tensor1, tensor3)

def test_benchmark_metrics_with_empty_metrics():
    """Test benchmark_metrics with empty metrics list."""
    preds = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    
    # Should raise ValueError with empty metrics list
    with pytest.raises(ValueError):
        benchmark_metrics([], preds, target)

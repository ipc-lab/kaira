import pytest
import torch

from kaira.metrics.base import BaseMetric


class ConcreteMetric(BaseMetric):
    """Concrete implementation of BaseMetric for testing."""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.total = 0.0
        self.count = 0
    
    def update(self, preds, targets):
        """Update metric state."""
        error = torch.abs(preds - targets).mean()
        self.total += error.item()
        self.count += 1
    
    def compute(self):
        """Compute final metric value."""
        if self.count == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.total / self.count)
    
    def forward(self, preds, targets):
        """Forward pass for metric computation."""
        error = torch.abs(preds - targets)
        if self.reduction == 'none':
            return error
        elif self.reduction == 'sum':
            return error.sum()
        else:  # default: 'mean'
            return error.mean()


class DummyMetric(BaseMetric):
    def reset(self):
        self.value = 0
        
    def update(self, preds, target):
        self.value += 1
        
    def compute(self):
        return self.value


def test_base_metric_stateful_computation():
    """Test stateful computation of metrics (update-compute pattern)."""
    metric = ConcreteMetric()
    
    # Create test data
    batch1 = (torch.tensor([1.0, 2.0]), torch.tensor([1.2, 1.8]))
    batch2 = (torch.tensor([3.0, 4.0]), torch.tensor([2.8, 4.2]))
    
    # First batch
    metric.update(*batch1)
    assert metric.count == 1
    
    # Second batch
    metric.update(*batch2)
    assert metric.count == 2
    
    # Compute final result
    result = metric.compute()
    assert isinstance(result, torch.Tensor)
    
    # Reset state
    metric.reset()
    assert metric.count == 0
    assert metric.total == 0.0


def test_base_metric_direct_computation():
    """Test direct computation of metrics (forward pass)."""
    metric = ConcreteMetric()
    
    # Create test data
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.2, 1.8, 3.1])
    
    # Compute directly
    result = metric(preds, targets)
    assert isinstance(result, torch.Tensor)
    
    # Test different reductions
    metric_none = ConcreteMetric(reduction='none')
    result_none = metric_none(preds, targets)
    assert result_none.shape == preds.shape
    
    metric_sum = ConcreteMetric(reduction='sum')
    result_sum = metric_sum(preds, targets)
    assert result_sum.shape == torch.Size([])


def test_base_metric_initialization():
    """Test BaseMetric initialization."""
    metric = DummyMetric()
    assert isinstance(metric, BaseMetric)


def test_base_metric_forward_calls_update_compute():
    """Test that BaseMetric forward method calls update and compute."""
    metric = DummyMetric()
    
    # Setup mock inputs
    preds = torch.randn(1, 3, 10, 10)
    target = torch.randn(1, 3, 10, 10)
    
    # Call forward
    result = metric(preds, target)
    
    # Check that update was called (since value should be 1)
    assert result == 1

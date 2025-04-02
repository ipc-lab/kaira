import torch

from kaira.losses import BaseLoss, CompositeLoss


class MockLoss(BaseLoss):
    """Mock loss that returns a predefined value."""

    def __init__(self, return_value=1.0):
        super().__init__()
        self.return_value = return_value

    def forward(self, x, y):
        return torch.tensor(self.return_value)


def test_composite_loss_forward_without_weights():
    """Test CompositeLoss forward pass without explicit weights."""
    loss1 = MockLoss(return_value=2.0)
    loss2 = MockLoss(return_value=3.0)
    
    # Initialize with empty list and then add losses
    composite = CompositeLoss(losses=[])
    composite.add_loss(loss1)
    composite.add_loss(loss2)
    
    # Without weights, it should be a simple average
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    result = composite(x, y)
    
    # Expected result: (2.0 + 3.0) / 2 = 2.5
    assert torch.isclose(result, torch.tensor(2.5))


def test_composite_loss_str_representation():
    """Test the string representation of CompositeLoss."""
    loss1 = MockLoss(return_value=1.0)
    loss2 = MockLoss(return_value=2.0)
    
    # Initialize with losses list
    composite = CompositeLoss(losses={})
    composite.add_loss("loss1", loss1, weight=0.3)
    composite.add_loss("loss2", loss2, weight=0.7)
    
    # Check string representation
    string_repr = str(composite)
    
    # Verify that the string representation includes key information
    assert "CompositeLoss" in string_repr
    assert "MockLoss" in string_repr
    assert "loss1" in string_repr
    assert "loss2" in string_repr
    assert "weight=" in string_repr  # Just check that weights are included


def test_composite_loss_empty():
    """Test CompositeLoss behavior when no losses are added."""
    # Initialize with empty list of losses
    composite = CompositeLoss(losses=[])

    # With no losses, it should return zero
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    result = composite(x, y)

    assert torch.isclose(result, torch.tensor(0.0))

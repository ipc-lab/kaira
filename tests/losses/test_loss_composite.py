import pytest
import torch

from kaira.losses import BaseLoss
from kaira.losses.composite import CompositeLoss


class MockLoss(BaseLoss):
    """Mock loss that returns a fixed value."""

    def __init__(self, return_value=1.0):
        super().__init__()
        self.return_value = return_value

    def forward(self, *args, **kwargs):
        return torch.tensor(self.return_value)


def test_composite_loss_creation():
    """Test creating CompositeLoss with various configurations."""
    # Simple case with two losses
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)

    losses = {"loss1": loss1, "loss2": loss2}
    composite = CompositeLoss(losses)

    # Default weights should be equal
    assert composite.weights["loss1"] == 0.5
    assert composite.weights["loss2"] == 0.5

    # With custom weights
    weights = {"loss1": 0.3, "loss2": 0.7}
    composite = CompositeLoss(losses, weights)

    assert composite.weights["loss1"] == 0.3
    assert composite.weights["loss2"] == 0.7

    # Invalid weights (should raise exception)
    with pytest.raises(ValueError):
        CompositeLoss(losses, {"loss1": 0.3, "invalid_key": 0.7})


def test_composite_loss_forward():
    """Test forward pass of CompositeLoss."""
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)

    # Equal weights
    losses = {"loss1": loss1, "loss2": loss2}
    composite = CompositeLoss(losses)

    # Forward with dummy inputs
    result = composite(torch.zeros(1), torch.zeros(1))

    # Should be weighted average: 1.0 * 0.5 + 2.0 * 0.5 = 1.5
    assert pytest.approx(result.item(), abs=1e-5) == 1.5

    # Custom weights
    weights = {"loss1": 0.8, "loss2": 0.2}
    composite = CompositeLoss(losses, weights)

    result = composite(torch.zeros(1), torch.zeros(1))

    # Should be weighted average: 1.0 * 0.8 + 2.0 * 0.2 = 1.2
    assert pytest.approx(result.item(), abs=1e-5) == 1.2


def test_composite_loss_individual_losses():
    """Test getting individual loss values."""
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)

    losses = {"loss1": loss1, "loss2": loss2}
    composite = CompositeLoss(losses)

    # Get individual losses
    individual = composite.get_individual_losses(torch.zeros(1), torch.zeros(1))

    assert "loss1" in individual
    assert "loss2" in individual
    assert individual["loss1"].item() == 1.0
    assert individual["loss2"].item() == 2.0


def test_add_loss():
    """Test adding a new loss to an existing CompositeLoss."""
    loss1 = MockLoss(1.0)

    # Start with single loss
    losses = {"loss1": loss1}
    composite = CompositeLoss(losses)

    # Add another loss
    loss2 = MockLoss(2.0)
    composite.add_loss("loss2", loss2, weight=0.3)

    # Check weights (should be normalized)
    assert composite.weights["loss1"] == 0.7
    assert composite.weights["loss2"] == 0.3

    # Adding duplicate should raise error
    with pytest.raises(ValueError):
        composite.add_loss("loss1", MockLoss(3.0))


def test_compute_individual():
    """Test compute_individual method returns the same as get_individual_losses."""
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)

    losses = {"loss1": loss1, "loss2": loss2}
    composite = CompositeLoss(losses)

    # Get individual losses with both methods
    individual1 = composite.get_individual_losses(torch.zeros(1), torch.zeros(1))
    individual2 = composite.compute_individual(torch.zeros(1), torch.zeros(1))

    # Results should be identical
    assert set(individual1.keys()) == set(individual2.keys())
    assert individual1["loss1"].item() == individual2["loss1"].item()
    assert individual1["loss2"].item() == individual2["loss2"].item()


def test_loss_return_tuple():
    """Test handling of losses that return tuples."""

    class TupleLoss(BaseLoss):
        def forward(self, *args, **kwargs):
            return (torch.tensor(3.0), torch.tensor(1.0))

    losses = {"tuple_loss": TupleLoss()}
    composite = CompositeLoss(losses)

    # Forward should take first element of tuple
    result = composite(torch.zeros(1), torch.zeros(1))
    assert pytest.approx(result.item(), abs=1e-5) == 3.0


def test_str_representation():
    """Test the string representation of CompositeLoss."""
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)

    losses = {"loss1": loss1, "loss2": loss2}
    weights = {"loss1": 0.3, "loss2": 0.7}
    composite = CompositeLoss(losses, weights)

    # Generate string representation
    str_repr = str(composite)

    # Check for essential components
    assert "CompositeLoss" in str_repr
    assert "ModuleDict" in str_repr
    assert "loss1" in str_repr
    assert "loss2" in str_repr
    assert "MockLoss" in str_repr
    assert "weight=0.300" in str_repr  # 0.3 formatted with 3 decimal places
    assert "weight=0.700" in str_repr  # 0.7 formatted with 3 decimal places

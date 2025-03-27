import pytest
import torch
import torch.nn as nn

from kaira.losses import BaseLoss, CompositeLoss


class MockLoss(BaseLoss):
    """Mock loss that returns a predefined value."""
    def __init__(self, return_value=1.0):
        super().__init__()
        self.return_value = return_value
        
    def forward(self, *args, **kwargs):
        return torch.tensor(self.return_value)


def test_composite_loss_initialization():
    """Test CompositeLoss initialization with different losses."""
    # Create component losses
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)
    loss3 = MockLoss(3.0)
    
    # Test with dictionary
    losses_dict = {"loss1": loss1, "loss2": loss2}
    comp_loss = CompositeLoss(losses_dict)
    assert comp_loss.losses["loss1"] == loss1
    assert comp_loss.losses["loss2"] == loss2
    
    # Test with dictionary and weights
    weights = {"loss1": 0.7, "loss2": 0.3}
    comp_loss = CompositeLoss(losses_dict, weights)
    assert comp_loss.weights["loss1"] == 0.7
    assert comp_loss.weights["loss2"] == 0.3
    
    # Test with automatic equal weighting
    comp_loss = CompositeLoss(losses_dict)
    assert comp_loss.weights["loss1"] == 0.5
    assert comp_loss.weights["loss2"] == 0.5
    
    # Test with incorrect weights keys
    with pytest.raises(ValueError):
        wrong_weights = {"loss1": 0.5, "wrong_key": 0.5}
        CompositeLoss(losses_dict, wrong_weights)


def test_composite_loss_forward():
    """Test CompositeLoss forward pass with weighted combination."""
    # Create component losses
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)
    
    # Create composite loss with equal weights
    losses_dict = {"loss1": loss1, "loss2": loss2}
    comp_loss = CompositeLoss(losses_dict)
    
    # Forward pass with dummy inputs
    result = comp_loss(torch.zeros(1), torch.zeros(1))
    assert pytest.approx(result.item(), abs=1e-5) == 1.5  # (1.0 * 0.5) + (2.0 * 0.5)
    
    # Test with custom weights
    weights = {"loss1": 0.8, "loss2": 0.2}
    comp_loss = CompositeLoss(losses_dict, weights)
    result = comp_loss(torch.zeros(1), torch.zeros(1))
    assert pytest.approx(result.item(), abs=1e-5) == 1.2  # (1.0 * 0.8) + (2.0 * 0.2)


def test_composite_loss_add_loss():
    """Test adding losses to a CompositeLoss."""
    # Create initial composite loss
    loss1 = MockLoss(1.0)
    comp_loss = CompositeLoss({"loss1": loss1})
    
    # Add a new loss
    loss2 = MockLoss(2.0)
    comp_loss.add_loss("loss2", loss2, weight=0.3)
    
    # Check if loss was added correctly
    assert comp_loss.losses["loss2"] == loss2
    assert pytest.approx(comp_loss.weights["loss2"], abs=1e-5) == 0.23076923076923075
    
    # Check if weights were rebalanced
    assert pytest.approx(sum(comp_loss.weights.values())) == 1.0
    
    # Test adding a loss with existing name
    with pytest.raises(ValueError):
        comp_loss.add_loss("loss1", MockLoss(3.0))


def test_composite_loss_get_individual_losses():
    """Test getting individual loss values."""
    # Create component losses
    loss1 = MockLoss(1.0)
    loss2 = MockLoss(2.0)
    
    # Create composite loss
    losses_dict = {"loss1": loss1, "loss2": loss2}
    comp_loss = CompositeLoss(losses_dict)
    
    # Get individual losses
    dummy_input = torch.zeros(1)
    individual_losses = comp_loss.get_individual_losses(dummy_input, dummy_input)
    
    assert individual_losses["loss1"].item() == 1.0
    assert individual_losses["loss2"].item() == 2.0

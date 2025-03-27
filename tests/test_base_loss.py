import torch

from kaira.losses.base import BaseLoss


class ConcreteLoss(BaseLoss):
    """Concrete implementation of BaseLoss for testing."""

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)


def test_base_loss_initialization():
    """Test BaseClass initialization."""
    loss = ConcreteLoss()
    assert isinstance(loss, BaseLoss)
    assert isinstance(loss, torch.nn.Module)


def test_base_loss_forward():
    """Test BaseLoss forward functionality."""
    loss = ConcreteLoss()

    # Create test data
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.5, 2.5, 3.5])

    # Compute loss
    result = loss(x, y)

    # Check result
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])  # Scalar output
    assert result.item() == 0.25  # Mean squared error of differences

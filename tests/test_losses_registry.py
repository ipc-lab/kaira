import pytest
import torch

from kaira.losses import BaseLoss, LossRegistry


class DummyLoss(BaseLoss):
    def __init__(self, value=0.5):
        super().__init__()
        self.value = value

    def forward(self, *args, **kwargs):
        return torch.tensor(self.value)


def test_loss_registry_register():
    """Test registering a loss with the LossRegistry."""
    # Clear existing registrations for this test
    original_losses = LossRegistry._losses.copy()
    LossRegistry._losses.clear()

    try:
        # Register a new loss
        LossRegistry.register("dummy", DummyLoss)
        assert "dummy" in LossRegistry._losses
        assert LossRegistry._losses["dummy"] == DummyLoss
    finally:
        # Restore original losses
        LossRegistry._losses = original_losses


def test_loss_registry_register_decorator():
    """Test using register_loss decorator."""
    original_losses = LossRegistry._losses.copy()
    LossRegistry._losses.clear()

    try:
        # Define and register a loss using decorator
        @LossRegistry.register_loss("decorator_test")
        class TestLoss(BaseLoss):
            def forward(self, *args, **kwargs):
                return torch.tensor(0.0)

        # Check registration
        assert "decorator_test" in LossRegistry._losses
        assert LossRegistry._losses["decorator_test"] == TestLoss

        # Test with default name
        @LossRegistry.register_loss()
        class ImplicitNameLoss(BaseLoss):
            def forward(self, *args, **kwargs):
                return torch.tensor(0.0)

        # Should use class name (lowercase)
        assert "implicitnameloss" in LossRegistry._losses
    finally:
        # Restore original losses
        LossRegistry._losses = original_losses


def test_loss_registry_create():
    """Test creating a loss instance from the registry."""
    original_losses = LossRegistry._losses.copy()
    LossRegistry._losses.clear()

    try:
        # Register a loss and create an instance
        LossRegistry.register("test_param", DummyLoss)
        loss = LossRegistry.create("test_param", value=0.75)

        # Verify the instance
        assert isinstance(loss, DummyLoss)
        assert loss.value == 0.75

        # Test with non-existent loss
        with pytest.raises(KeyError):
            LossRegistry.create("nonexistent_loss")
    finally:
        # Restore original losses
        LossRegistry._losses = original_losses


def test_loss_registry_list_losses():
    """Test listing registered losses."""
    original_losses = LossRegistry._losses.copy()
    LossRegistry._losses.clear()

    try:
        LossRegistry.register("loss1", DummyLoss)
        LossRegistry.register("loss2", DummyLoss)

        losses = LossRegistry.list_losses()
        assert "loss1" in losses
        assert "loss2" in losses
        assert len(losses) == 2
    finally:
        # Restore original losses
        LossRegistry._losses = original_losses

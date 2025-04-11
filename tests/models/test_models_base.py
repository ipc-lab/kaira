import pytest
import torch

from kaira.models.base import BaseModel, ConfigurableModel


class DummyModel(BaseModel):
    def forward(self, x):
        return x * 2


def test_base_model_forward():
    model = DummyModel()
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor * 2)


def test_configurable_model_add_step():
    model = ConfigurableModel()

    def step(x):
        return x + 1

    model.add_step(step)
    assert len(model.steps) == 1
    assert model.steps[0] == step


def test_configurable_model_remove_step():
    model = ConfigurableModel()

    def step1(x):
        return x + 1

    def step2(x):
        return x * 2

    model.add_step(step1).add_step(step2)
    model.remove_step(0)
    assert len(model.steps) == 1
    assert model.steps[0] == step2


def test_configurable_model_forward():
    model = ConfigurableModel()

    def step1(x):
        return x + 1

    def step2(x):
        return x * 2

    model.add_step(step1).add_step(step2)
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_tensor)
    expected_output = (input_tensor + 1) * 2
    assert torch.allclose(output, expected_output)


def test_configurable_model_remove_step_out_of_range():
    model = ConfigurableModel()
    with pytest.raises(IndexError):
        model.remove_step(0)


def test_base_model_abstract_forward():
    """Test that BaseModel's forward method raises NotImplementedError if not implemented in
    subclass."""

    # Create a class that inherits from BaseModel and implements forward
    # but calls the parent's implementation which should raise NotImplementedError
    class IncompleteModel(BaseModel):
        def forward(self, *args, **kwargs):
            # Call parent's forward method which should raise NotImplementedError
            return super().forward(*args, **kwargs)

    # Instantiate the incomplete model
    model = IncompleteModel()

    # Verify that calling forward raises NotImplementedError
    with pytest.raises(NotImplementedError):
        model(torch.randn(1, 10))


def test_configurable_model_forward_implementation():
    """Test that ConfigurableModel's forward method correctly processes input through steps."""
    model = ConfigurableModel()
    input_tensor = torch.tensor([1.0, 2.0, 3.0])

    # Test with no steps added
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor), "Forward should return input unchanged when no steps exist"

    # Test with empty steps list
    model.steps = []
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor), "Forward should return input unchanged with empty steps list"


def test_configurable_model_forward_with_kwargs():
    """Test that ConfigurableModel's forward method handles kwargs correctly."""
    model = ConfigurableModel()

    def step_with_kwargs(x, scale=1):
        return x * scale

    model.add_step(step_with_kwargs)
    input_tensor = torch.tensor([1.0, 2.0, 3.0])

    # Test with default kwargs
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor), "Should use default kwargs value"

    # Test with custom kwargs
    output = model(input_tensor, scale=2)
    assert torch.allclose(output, input_tensor * 2), "Should use provided kwargs value"

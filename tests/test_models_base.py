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
    step = lambda x: x + 1
    model.add_step(step)
    assert len(model.steps) == 1
    assert model.steps[0] == step

def test_configurable_model_remove_step():
    model = ConfigurableModel()
    step1 = lambda x: x + 1
    step2 = lambda x: x * 2
    model.add_step(step1).add_step(step2)
    model.remove_step(0)
    assert len(model.steps) == 1
    assert model.steps[0] == step2

def test_configurable_model_forward():
    model = ConfigurableModel()
    step1 = lambda x: x + 1
    step2 = lambda x: x * 2
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
    """Test that BaseModel's forward method raises NotImplementedError if not implemented in subclass."""
    # Create a class that inherits from BaseModel but doesn't implement forward
    class IncompleteModel(BaseModel):
        pass
    
    # Instantiate the incomplete model
    model = IncompleteModel()
    
    # Verify that calling forward raises NotImplementedError
    with pytest.raises(NotImplementedError):
        model.forward(torch.randn(1, 10))

import pytest
import torch
from kaira.models.generic.parallel import ParallelModel
from kaira.models.base import BaseModel

class DummyModel(BaseModel):
    """A simple dummy model for testing."""
    def forward(self, x, *args, **kwargs):
        return x * 2
        
class ErrorModel(BaseModel):
    """A model that raises an exception when called."""
    def forward(self, x, *args, **kwargs):
        raise ValueError("Simulated error")

@pytest.fixture
def input_tensor():
    """Fixture providing a sample input tensor."""
    return torch.tensor([[1., 2.], [3., 4.]])

def test_parallel_model_basic(input_tensor):
    """Test basic functionality of ParallelModel."""
    model = ParallelModel()
    model.add_step(DummyModel(), "double")
    model.add_step(lambda x: x + 1, "plus_one")
    
    results = model(input_tensor)
    
    assert len(results) == 2
    assert "double" in results
    assert "plus_one" in results
    assert torch.all(results["double"] == input_tensor * 2)
    assert torch.all(results["plus_one"] == input_tensor + 1)

def test_parallel_model_init_with_steps():
    """Test initializing ParallelModel with steps."""
    steps = [
        ("double", DummyModel()),
        ("identity", lambda x: x)
    ]
    model = ParallelModel(steps=steps)
    
    assert len(model.steps) == 2
    assert model.steps[0][0] == "double"
    assert model.steps[1][0] == "identity"
    
    input_data = torch.tensor([1.0, 2.0])
    results = model(input_data)
    
    assert "double" in results
    assert "identity" in results
    assert torch.all(results["double"] == input_data * 2)
    assert torch.all(results["identity"] == input_data)

def test_parallel_model_auto_naming():
    """Test auto-naming of steps when name is not provided."""
    model = ParallelModel()
    model.add_step(lambda x: x)
    model.add_step(lambda x: x)
    
    assert model.steps[0][0] == "step_0"
    assert model.steps[1][0] == "step_1"
    
    # Test that auto-naming continues from current count
    model.remove_step(1)
    model.add_step(lambda x: x)
    assert model.steps[1][0] == "step_2"

def test_parallel_model_non_callable_step():
    """Test adding a non-callable step raises TypeError."""
    model = ParallelModel()
    
    with pytest.raises(TypeError, match="Step must be callable"):
        model.add_step("not_callable")

def test_parallel_model_remove_step_out_of_range():
    """Test removing a step with out-of-range index raises IndexError."""
    model = ParallelModel()
    model.add_step(lambda x: x)
    
    with pytest.raises(IndexError, match="Step index 1 out of range"):
        model.remove_step(1)
    
    with pytest.raises(IndexError, match="Step index -1 out of range"):
        model.remove_step(-1)

def test_parallel_model_empty_steps(input_tensor):
    """Test calling forward with no steps returns empty dict."""
    model = ParallelModel()
    results = model(input_tensor)
    
    assert results == {}

def test_parallel_model_with_error(input_tensor):
    """Test that errors in steps are caught and reported."""
    model = ParallelModel()
    model.add_step(ErrorModel(), "error_step")
    
    results = model(input_tensor)
    
    assert "error_step" in results
    assert isinstance(results["error_step"], str)
    assert "Error: Simulated error" in results["error_step"]

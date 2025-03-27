import pytest
import torch
from kaira.models.generic.branching import BranchingModel
from kaira.models.base import BaseModel

class DummyModel(BaseModel):
    def forward(self, x):
        return x * 2

class AnotherDummyModel(BaseModel):
    def forward(self, x):
        return x + 3

@pytest.fixture
def branching_model():
    model = BranchingModel()
    model.add_branch("branch1", condition=lambda x: x.sum() > 10, model=DummyModel())
    model.add_branch("branch2", condition=lambda x: x.sum() <= 10, model=AnotherDummyModel())
    return model

def test_branching_model_forward(branching_model):
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output, branch = branching_model(input_tensor, return_branch=True)
    assert torch.allclose(output, input_tensor + 3)
    assert branch == "branch2"

    input_tensor = torch.tensor([5.0, 6.0])
    output, branch = branching_model(input_tensor, return_branch=True)
    assert torch.allclose(output, input_tensor * 2)
    assert branch == "branch1"

def test_branching_model_default_branch():
    model = BranchingModel()
    model.set_default_branch(DummyModel())
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output, branch = model(input_tensor, return_branch=True)
    assert torch.allclose(output, input_tensor * 2)
    assert branch == "default"

def test_branching_model_remove_branch(branching_model):
    branching_model.remove_branch("branch1")
    input_tensor = torch.tensor([5.0, 6.0])
    with pytest.raises(RuntimeError):
        branching_model(input_tensor)

def test_branching_model_get_branch(branching_model):
    condition, model = branching_model.get_branch("branch1")
    assert condition(torch.tensor([5.0, 6.0])) is True
    assert isinstance(model, DummyModel)

# New tests to increase coverage

def test_branching_model_add_branch_duplicate():
    """Test adding a branch with a duplicate name raises ValueError."""
    model = BranchingModel()
    dummy = DummyModel()
    condition = lambda x: True
    
    # Add first branch
    model.add_branch("test", condition, dummy)
    
    # Try to add branch with same name
    with pytest.raises(ValueError, match="Branch 'test' already exists"):
        model.add_branch("test", condition, dummy)

def test_branching_model_default_branch_execution():
    """Test that default branch is executed when no conditions match."""
    model = BranchingModel()
    
    # Add a branch with a condition that will never match
    model.add_branch("never_match", lambda x: False, DummyModel())
    
    # Set a default branch that we can verify was called
    default_model = DummyModel()
    model.set_default_branch(default_model)
    
    # Test with return_branch=True to check branch name
    result, branch_name = model(torch.tensor([1.0]), return_branch=True)
    assert branch_name == "default"
    assert result is not None

def test_branching_model_get_branch_nonexistent():
    """Test that get_branch raises KeyError for nonexistent branch."""
    model = BranchingModel()
    
    with pytest.raises(KeyError, match="Branch 'nonexistent' not found"):
        model.get_branch("nonexistent")

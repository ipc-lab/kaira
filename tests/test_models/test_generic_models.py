# tests/test_models/test_generic_models.py
import pytest
import torch
import torch.nn as nn

from kaira.models.generic import (
    BranchingModel,
    IdentityModel,
    LambdaModel,
    ParallelModel,
    SequentialModel,
)


class SimpleLayer(nn.Module):
    """Simple layer that adds a constant to the input."""
    
    def __init__(self, add_value=1.0):
        super().__init__()
        self.add_value = add_value
        
    def forward(self, x):
        return x + self.add_value


class ScaleLayer(nn.Module):
    """Simple layer that scales the input by a constant."""
    
    def __init__(self, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return x * self.scale_factor


@pytest.fixture
def input_tensor():
    """Fixture providing a simple input tensor for testing."""
    return torch.tensor([1.0, 2.0, 3.0])


def test_sequential_model(input_tensor):
    """Test sequential model processes steps in order."""
    # Create steps that will be applied in sequence
    step1 = SimpleLayer(add_value=1.0)  # Add 1
    step2 = ScaleLayer(scale_factor=2.0)  # Multiply by 2
    step3 = SimpleLayer(add_value=3.0)  # Add 3
    
    # Create sequential model and add steps
    model = SequentialModel([step1, step2])
    model.add_step(step3)
    
    # Check number of steps
    assert len(model.steps) == 3
    
    # Apply model to input
    output = model(input_tensor)
    
    # Expected: ((input + 1) * 2) + 3
    expected = (input_tensor + 1.0) * 2.0 + 3.0
    assert torch.allclose(output, expected)


def test_sequential_model_empty():
    """Test sequential model with no steps (identity behavior)."""
    model = SequentialModel()
    
    # Input should pass through unchanged
    input_data = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_data)
    
    assert torch.allclose(output, input_data)


def test_sequential_model_invalid_step():
    """Test sequential model rejects non-callable steps."""
    model = SequentialModel()
    
    # Try to add a non-callable step
    with pytest.raises(TypeError):
        model.add_step("not_callable")


def test_parallel_model(input_tensor):
    """Test parallel model processes branches and aggregates results."""
    # Create branches for parallel processing
    branch1 = SimpleLayer(add_value=1.0)  # Add 1
    branch2 = ScaleLayer(scale_factor=2.0)  # Multiply by 2
    branch3 = lambda x: x ** 2  # Square
    
    # Create parallel model with branches and sum aggregator
    model = ParallelModel(
        branches=[branch1, branch2, branch3],
        aggregator=lambda outputs: sum(outputs)
    )
    
    # Apply model to input
    output = model(input_tensor)
    
    # Expected: (input + 1) + (input * 2) + (input^2)
    expected = (input_tensor + 1.0) + (input_tensor * 2.0) + (input_tensor ** 2)
    assert torch.allclose(output, expected)


def test_parallel_model_custom_aggregator(input_tensor):
    """Test parallel model with custom aggregation function."""
    # Create branches
    branch1 = SimpleLayer(add_value=5.0)  
    branch2 = ScaleLayer(scale_factor=0.5)
    
    # Create custom aggregator that takes the max of all outputs
    def max_aggregator(outputs):
        return torch.max(torch.stack(outputs), dim=0)[0]
    
    # Create parallel model with custom aggregator
    model = ParallelModel(
        branches=[branch1, branch2],
        aggregator=max_aggregator
    )
    
    # Apply model to input
    output = model(input_tensor)
    
    # Expected: element-wise max of (input + 5) and (input * 0.5)
    expected = torch.max(
        torch.stack([input_tensor + 5.0, input_tensor * 0.5]), 
        dim=0
    )[0]
    assert torch.allclose(output, expected)


def test_branching_model(input_tensor):
    """Test branching model with condition-based routing."""
    # Create a condition function that routes based on sum of input
    def condition(x):
        return torch.sum(x) > 5.0
    
    # Create true and false branches
    true_branch = SimpleLayer(add_value=10.0)   # Add 10
    false_branch = ScaleLayer(scale_factor=0.5) # Multiply by 0.5
    
    # Create branching model
    model = BranchingModel(
        condition=condition,
        true_branch=true_branch,
        false_branch=false_branch
    )
    
    # Test with input that should take true branch
    true_input = torch.tensor([2.0, 2.0, 2.0])  # Sum = 6 > 5
    true_output = model(true_input)
    assert torch.allclose(true_output, true_input + 10.0)
    
    # Test with input that should take false branch
    false_input = torch.tensor([1.0, 1.0, 1.0])  # Sum = 3 < 5
    false_output = model(false_input)
    assert torch.allclose(false_output, false_input * 0.5)


def test_branching_model_default_branches(input_tensor):
    """Test branching model with default branches (identity functions)."""
    # Create a condition function
    def condition(x):
        return torch.sum(x) > 5.0
    
    # Create branching model with only condition
    model = BranchingModel(condition=condition)
    
    # Both branches should act as identity functions
    true_input = torch.tensor([2.0, 2.0, 2.0])  # Sum = 6 > 5
    true_output = model(true_input)
    assert torch.allclose(true_output, true_input)
    
    false_input = torch.tensor([1.0, 1.0, 1.0])  # Sum = 3 < 5
    false_output = model(false_input)
    assert torch.allclose(false_output, false_input)


def test_identity_model(input_tensor):
    """Test identity model passes input unchanged."""
    model = IdentityModel()
    output = model(input_tensor)
    
    # Output should be identical to input
    assert torch.allclose(output, input_tensor)
    
    # Test with different input types
    string_input = "test string"
    assert model(string_input) == string_input
    
    dict_input = {"key": "value"}
    assert model(dict_input) == dict_input


def test_lambda_model(input_tensor):
    """Test lambda model applies the provided function."""
    # Create a lambda model with a squaring function
    square_fn = lambda x: x ** 2
    model = LambdaModel(function=square_fn)
    
    # Apply model
    output = model(input_tensor)
    
    # Expected: input^2
    expected = input_tensor ** 2
    assert torch.allclose(output, expected)


def test_lambda_model_with_args(input_tensor):
    """Test lambda model with additional arguments."""
    # Create a lambda model with a function that uses additional args
    def add_and_scale(x, add_value, scale_factor):
        return (x + add_value) * scale_factor
    
    model = LambdaModel(function=add_and_scale)
    
    # Apply model with additional args
    output = model(input_tensor, 5.0, 2.0)
    
    # Expected: (input + 5) * 2
    expected = (input_tensor + 5.0) * 2.0
    assert torch.allclose(output, expected)


def test_model_composition(input_tensor):
    """Test composition of different generic models."""
    # Create components
    seq_model = SequentialModel([
        SimpleLayer(add_value=1.0),
        ScaleLayer(scale_factor=2.0)
    ])
    
    parallel_model = ParallelModel(
        branches=[
            lambda x: x + 3.0,
            lambda x: x * 0.5
        ],
        aggregator=lambda outputs: sum(outputs)
    )
    
    # Compose models: sequential followed by parallel
    composed_model = SequentialModel([seq_model, parallel_model])
    
    # Apply composed model
    output = composed_model(input_tensor)
    
    # Expected: Let's break it down
    # 1. Sequential: (input + 1) * 2
    intermediate = (input_tensor + 1.0) * 2.0
    # 2. Parallel: (intermediate + 3) + (intermediate * 0.5)
    expected = (intermediate + 3.0) + (intermediate * 0.5)
    
    assert torch.allclose(output, expected)
import pytest
import torch
from kaira.constraints import LambdaConstraint

def test_lambda_constraint():
    # Define a simple lambda function for testing
    lambda_function = lambda x: x * 2
    constraint = LambdaConstraint(lambda_function)
    
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    
    assert torch.allclose(y, x * 2)

def test_lambda_constraint_with_complex_function():
    # Define a more complex lambda function for testing
    lambda_function = lambda x: torch.where(x > 0, x, torch.zeros_like(x))
    constraint = LambdaConstraint(lambda_function)
    
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = constraint(x)
    
    expected = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(y, expected)

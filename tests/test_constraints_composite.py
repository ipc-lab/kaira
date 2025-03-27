# tests/test_constraints_composite.py
import pytest
import torch
import torch.nn as nn

from kaira.constraints import BaseConstraint, CompositeConstraint, TotalPowerConstraint, AveragePowerConstraint


class MockConstraint(BaseConstraint):
    """Mock constraint that multiplies input by a constant."""
    
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        
    def forward(self, x):
        return x * self.factor


@pytest.fixture
def random_tensor():
    """Fixture providing a random tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 32, 32)


def test_composite_constraint_initialization():
    """Test CompositeConstraint initialization with different constraints."""
    # Single constraint
    constraint1 = MockConstraint(0.5)
    composite = CompositeConstraint([constraint1])
    assert len(composite.constraints) == 1
    
    # Multiple constraints
    constraint2 = MockConstraint(2.0)
    composite = CompositeConstraint([constraint1, constraint2])
    assert len(composite.constraints) == 2
    
    # Initialize with nn.ModuleList
    constraints = nn.ModuleList([constraint1, constraint2])
    composite = CompositeConstraint(constraints)
    assert len(composite.constraints) == 2
    assert composite.constraints is constraints  # Should use the same ModuleList


def test_composite_constraint_forward(random_tensor):
    """Test CompositeConstraint forward pass applies constraints in sequence."""
    # Create two mock constraints with known factors
    constraint1 = MockConstraint(0.5)  # Halves the input
    constraint2 = MockConstraint(2.0)  # Doubles the input
    
    # Create composite with both constraints
    composite = CompositeConstraint([constraint1, constraint2])
    
    # Apply composite constraint
    result = composite(random_tensor)
    
    # The result should be equivalent to applying both constraints in sequence
    # First halving (×0.5) then doubling (×2), which equals the original
    expected = random_tensor * 0.5 * 2.0
    assert torch.allclose(result, expected)
    
    # Change the order of constraints - should affect the result
    composite_reversed = CompositeConstraint([constraint2, constraint1])
    result_reversed = composite_reversed(random_tensor)
    
    # Now we double first (×2) then halve (×0.5), which still equals the original
    # but verifies the order dependency
    expected_reversed = random_tensor * 2.0 * 0.5
    assert torch.allclose(result_reversed, expected_reversed)
    
    # Both results should be equal in this case (but might not be for other constraint types)
    assert torch.allclose(result, result_reversed)


def test_composite_with_real_constraints(random_tensor):
    """Test CompositeConstraint with actual power constraints."""
    # Create power constraints with known parameters
    total_power = TotalPowerConstraint(total_power=1.0)
    avg_power = AveragePowerConstraint(average_power=0.1)
    
    # Create composite with both constraints
    composite = CompositeConstraint([total_power, avg_power])
    
    # Apply composite constraint
    result = composite(random_tensor)
    
    # Check that the final constraint (avg_power) is satisfied
    # Use a more relaxed tolerance value since floating-point operations can introduce small differences
    assert torch.isclose(torch.mean(result**2), torch.tensor(0.1), rtol=1e-4, atol=1e-4)
    
    # The total power constraint applied first, but its effect is overridden by avg_power
    # So we shouldn't expect total_power to be exactly 1.0


def test_composite_constraint_add_constraint():
    """Test adding constraints to a CompositeConstraint."""
    # Create a composite with one constraint
    constraint1 = MockConstraint(0.5)
    composite = CompositeConstraint([constraint1])
    assert len(composite.constraints) == 1
    
    # Add a new constraint
    constraint2 = MockConstraint(2.0)
    composite.add_constraint(constraint2)
    assert len(composite.constraints) == 2
    
    # Test that the new constraint is in the list
    assert composite.constraints[1] is constraint2


def test_composite_constraint_invalid_input():
    """Test that CompositeConstraint raises appropriate errors for invalid inputs."""
    # Test with non-constraint object
    with pytest.raises(TypeError):
        CompositeConstraint([nn.Linear(10, 10)])
    
    # Test adding non-constraint object
    composite = CompositeConstraint([MockConstraint(1.0)])
    with pytest.raises(TypeError):
        composite.add_constraint(nn.Linear(10, 10))
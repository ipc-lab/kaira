import pytest
import torch

from kaira.constraints import BaseConstraint, ConstraintRegistry


class DummyConstraint(BaseConstraint):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x * self.value


def test_constraint_registry_register():
    """Test registering a constraint with the ConstraintRegistry."""
    # Clear existing registrations for this test
    original_constraints = ConstraintRegistry._constraints.copy()
    ConstraintRegistry._constraints.clear()
    
    try:
        # Register a new constraint
        ConstraintRegistry.register("dummy", DummyConstraint)
        assert "dummy" in ConstraintRegistry._constraints
        assert ConstraintRegistry._constraints["dummy"] == DummyConstraint
    finally:
        # Restore original constraints
        ConstraintRegistry._constraints = original_constraints


def test_constraint_registry_register_decorator():
    """Test using register_constraint decorator."""
    original_constraints = ConstraintRegistry._constraints.copy()
    ConstraintRegistry._constraints.clear()
    
    try:
        # Define and register a constraint using decorator
        @ConstraintRegistry.register_constraint("decorator_test")
        class TestConstraint(BaseConstraint):
            def forward(self, x):
                return x
        
        # Check registration
        assert "decorator_test" in ConstraintRegistry._constraints
        assert ConstraintRegistry._constraints["decorator_test"] == TestConstraint
        
        # Test with default name
        @ConstraintRegistry.register_constraint()
        class ImplicitNameConstraint(BaseConstraint):
            def forward(self, x):
                return x
        
        # Should use class name (lowercase)
        assert "implicitnameconstraint" in ConstraintRegistry._constraints
    finally:
        # Restore original constraints
        ConstraintRegistry._constraints = original_constraints


def test_constraint_registry_create():
    """Test creating a constraint instance from the registry."""
    original_constraints = ConstraintRegistry._constraints.copy()
    ConstraintRegistry._constraints.clear()
    
    try:
        # Register a constraint and create an instance
        ConstraintRegistry.register("test_param", DummyConstraint)
        constraint = ConstraintRegistry.create("test_param", value=3.0)
        
        # Verify the instance
        assert isinstance(constraint, DummyConstraint)
        assert constraint.value == 3.0
        
        # Test with non-existent constraint
        with pytest.raises(KeyError):
            ConstraintRegistry.create("nonexistent_constraint")
    finally:
        # Restore original constraints
        ConstraintRegistry._constraints = original_constraints


def test_constraint_registry_list_constraints():
    """Test listing registered constraints."""
    original_constraints = ConstraintRegistry._constraints.copy()
    ConstraintRegistry._constraints.clear()
    
    try:
        ConstraintRegistry.register("constraint1", DummyConstraint)
        ConstraintRegistry.register("constraint2", DummyConstraint)
        
        constraints = ConstraintRegistry.list_constraints()
        assert "constraint1" in constraints
        assert "constraint2" in constraints
        assert len(constraints) == 2
    finally:
        # Restore original constraints
        ConstraintRegistry._constraints = original_constraints

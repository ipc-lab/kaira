"""Tests for generic model classes (LambdaModel, ParallelModel, BranchingModel)."""
import pytest
import torch

from kaira.models.base import BaseModel
from kaira.models.generic import (
    IdentityModel,
    LambdaModel,
    ParallelModel,
    BranchingModel,
)


# Shared test fixtures and utility classes

class DummyModel(BaseModel):
    """A simple dummy model for testing."""

    def forward(self, x, *args, **kwargs):
        return x * 2


class ErrorModel(BaseModel):
    """A model that raises an exception when called."""

    def forward(self, x, *args, **kwargs):
        raise ValueError("Simulated error")


class AnotherDummyModel(BaseModel):
    """Another dummy model for testing."""

    def forward(self, x, *args, **kwargs):
        return x + 3


@pytest.fixture
def input_tensor():
    """Fixture providing a sample input tensor."""
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def branching_model():
    """Fixture providing a sample branching model."""
    model = BranchingModel()
    model.add_branch("branch1", condition=lambda x: x.sum() > 10, model=DummyModel())
    model.add_branch("branch2", condition=lambda x: x.sum() <= 10, model=AnotherDummyModel())
    return model


# LambdaModel tests

class TestLambdaModel:
    """Tests for the LambdaModel class."""

    def test_lambda_model_forward(self):
        """Test forward pass of LambdaModel with a simple lambda."""
        model = LambdaModel(lambda x: x * 2)
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = model(input_tensor)
        expected_output = input_tensor * 2
        assert torch.allclose(output, expected_output)

    def test_lambda_model_with_args(self):
        """Test that LambdaModel can pass additional args to the lambda function."""
        def add_and_scale(x, add_value, scale_factor):
            return (x + add_value) * scale_factor

        model = LambdaModel(add_and_scale)
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = model(input_tensor, 5.0, 2.0)
        expected_output = (input_tensor + 5.0) * 2.0
        assert torch.allclose(output, expected_output)

    def test_lambda_model_repr(self):
        """Test the string representation of LambdaModel."""
        model = LambdaModel(lambda x: x * 2, name="DoubleModel")
        expected_repr = "LambdaModel(name=DoubleModel, func=<lambda>)"
        assert repr(model) == expected_repr


# ParallelModel tests

class TestParallelModel:
    """Tests for the ParallelModel class."""

    def test_parallel_model_basic(self, input_tensor):
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

    def test_parallel_model_init_with_steps(self):
        """Test initializing ParallelModel with steps."""
        steps = [("double", DummyModel()), ("identity", lambda x: x)]
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

    def test_parallel_model_auto_naming(self):
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

    def test_parallel_model_non_callable_step(self):
        """Test adding a non-callable step raises TypeError."""
        model = ParallelModel()

        with pytest.raises(TypeError, match="Step must be callable"):
            model.add_step("not_callable")

    def test_parallel_model_remove_step_out_of_range(self):
        """Test removing a step with out-of-range index raises IndexError."""
        model = ParallelModel()
        model.add_step(lambda x: x)

        with pytest.raises(IndexError, match="Step index 1 out of range"):
            model.remove_step(1)

        with pytest.raises(IndexError, match="Step index -1 out of range"):
            model.remove_step(-1)

    def test_parallel_model_empty_steps(self, input_tensor):
        """Test calling forward with no steps returns empty dict."""
        model = ParallelModel()
        results = model(input_tensor)

        assert results == {}

    def test_parallel_model_with_error(self, input_tensor):
        """Test that errors in steps are caught and reported."""
        model = ParallelModel()
        model.add_step(ErrorModel(), "error_step")

        results = model(input_tensor)

        assert "error_step" in results
        assert isinstance(results["error_step"], str)
        assert "Error: Simulated error" in results["error_step"]


# BranchingModel tests

class TestBranchingModel:
    """Tests for the BranchingModel class."""

    def test_branching_model_forward(self, branching_model):
        """Test forward pass with branch selection based on input."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output, branch = branching_model(input_tensor, return_branch=True)
        assert torch.allclose(output, input_tensor + 3)
        assert branch == "branch2"

        input_tensor = torch.tensor([5.0, 6.0])
        output, branch = branching_model(input_tensor, return_branch=True)
        assert torch.allclose(output, input_tensor * 2)
        assert branch == "branch1"

    def test_branching_model_default_branch(self):
        """Test using a default branch when no conditions match."""
        model = BranchingModel()
        model.set_default_branch(DummyModel())
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output, branch = model(input_tensor, return_branch=True)
        assert torch.allclose(output, input_tensor * 2)
        assert branch == "default"

    def test_branching_model_remove_branch(self, branching_model):
        """Test removing a branch and handling requests that would use it."""
        branching_model.remove_branch("branch1")
        input_tensor = torch.tensor([5.0, 6.0])
        with pytest.raises(RuntimeError):
            branching_model(input_tensor)

    def test_branching_model_get_branch(self, branching_model):
        """Test getting branch components by name."""
        condition, model = branching_model.get_branch("branch1")
        assert condition(torch.tensor([5.0, 6.0])) is True
        assert isinstance(model, DummyModel)

    def test_branching_model_add_branch_duplicate(self):
        """Test adding a branch with a duplicate name raises ValueError."""
        model = BranchingModel()
        dummy = DummyModel()

        def condition(x):
            return True

        # Add first branch
        model.add_branch("test", condition, dummy)

        # Try to add branch with same name
        with pytest.raises(ValueError, match="Branch 'test' already exists"):
            model.add_branch("test", condition, dummy)

    def test_branching_model_default_branch_execution(self):
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

    def test_branching_model_get_branch_nonexistent(self):
        """Test that get_branch raises KeyError for nonexistent branch."""
        model = BranchingModel()

        with pytest.raises(KeyError, match="Branch 'nonexistent' not found"):
            model.get_branch("nonexistent")

    def test_branching_model_remove_nonexistent_branch(self):
        """Test that removing a nonexistent branch raises KeyError."""
        model = BranchingModel()
        
        # Attempt to remove a branch that doesn't exist
        with pytest.raises(KeyError, match="Branch 'nonexistent' not found"):
            model.remove_branch("nonexistent")
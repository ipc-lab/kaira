"""Tests for generic model classes (LambdaModel, ParallelModel, BranchingModel,
SequentialModel)."""
import pytest
import torch

from kaira.models.base import BaseModel
from kaira.models.generic import (
    BranchingModel,
    IdentityModel,
    LambdaModel,
    ParallelModel,
    SequentialModel,
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


class SimpleLayer(BaseModel):
    """Simple layer that adds a constant to the input."""

    def __init__(self, add_value=1.0):
        super().__init__()
        self.add_value = add_value

    def forward(self, x):
        return x + self.add_value


class ScaleLayer(BaseModel):
    """Simple layer that scales the input by a constant."""

    def __init__(self, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x * self.scale_factor


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


# SequentialModel tests


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

    def test_parallel_model_branches_and_aggregator(self, input_tensor):
        """Test parallel model with branches and custom aggregator."""
        # Create branches for parallel processing
        branch1 = SimpleLayer(add_value=1.0)  # Add 1
        branch2 = ScaleLayer(scale_factor=2.0)  # Multiply by 2

        def branch3(x):
            return x**2  # Square

        # Create parallel model with branches and sum aggregator
        model = ParallelModel(branches=[branch1, branch2, branch3], aggregator=lambda outputs: sum(outputs))

        # Apply model to input
        sample_input = torch.tensor([1.0, 2.0, 3.0])
        output = model(sample_input)

        # Expected: (input + 1) + (input * 2) + (input^2)
        expected = (sample_input + 1.0) + (sample_input * 2.0) + (sample_input**2)
        assert torch.allclose(output, expected)


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

    def test_branching_model_with_condition(self):
        """Test branching model with condition-based routing."""

        # Create a condition function that routes based on sum of input
        def condition(x):
            return torch.sum(x) > 5.0

        # Create true and false branches
        true_branch = SimpleLayer(add_value=10.0)  # Add 10
        false_branch = ScaleLayer(scale_factor=0.5)  # Multiply by 0.5

        # Create branching model
        model = BranchingModel(condition=condition, true_branch=true_branch, false_branch=false_branch)

        # Test with input that should take true branch
        true_input = torch.tensor([2.0, 2.0, 2.0])  # Sum = 6 > 5
        true_output = model(true_input)
        assert torch.allclose(true_output, true_input + 10.0)

        # Test with input that should take false branch
        false_input = torch.tensor([1.0, 1.0, 1.0])  # Sum = 3 < 5
        false_output = model(false_input)
        assert torch.allclose(false_output, false_input * 0.5)

    def test_branching_model_get_branch_condition_wrapper(self):
        """Test that condition wrapper returned by get_branch handles tensor and non-tensor
        results."""
        model = BranchingModel()

        # Test with a condition that returns a tensor with item() method
        def tensor_condition(x):
            # Use recommended clone().detach() instead of torch.tensor()
            return (x.sum() > 5).clone().detach()

        model.add_branch("tensor_branch", tensor_condition, DummyModel())
        wrapper, _ = model.get_branch("tensor_branch")

        # Verify tensor result is converted to Python bool
        result_true = wrapper(torch.tensor([3.0, 3.0]))  # Sum = 6 > 5
        result_false = wrapper(torch.tensor([2.0, 2.0]))  # Sum = 4 < 5
        assert isinstance(result_true, bool)
        assert isinstance(result_false, bool)
        assert result_true is True
        assert result_false is False

        # Test with a condition that returns a Python bool directly
        def bool_condition(x):
            return x.sum() > 5

        model.add_branch("bool_branch", bool_condition, DummyModel())
        wrapper, _ = model.get_branch("bool_branch")

        # Verify Python bool result is preserved
        result_true = wrapper(torch.tensor([3.0, 3.0]))  # Sum = 6 > 5
        result_false = wrapper(torch.tensor([2.0, 2.0]))  # Sum = 4 < 5
        assert isinstance(result_true, bool)
        assert isinstance(result_false, bool)
        assert result_true is True
        assert result_false is False

    def test_branching_model_non_boolean_condition(self):
        """Test that non-boolean, non-tensor condition results are properly converted to
        booleans."""
        model = BranchingModel()

        # Create a custom class with __bool__ method
        class CustomBooleanable:
            def __init__(self, value):
                self.value = value

            def __bool__(self):
                return self.value > 0

        # Add a branch with a condition that returns our custom object
        def custom_condition(x):
            # Return an object that's neither a tensor nor a boolean
            return CustomBooleanable(x.sum().item())

        model.add_branch("custom_branch", custom_condition, DummyModel())
        wrapper, _ = model.get_branch("custom_branch")

        # Test with values that should convert to True and False
        result_true = wrapper(torch.tensor([1.0, 2.0]))  # Sum = 3 > 0
        result_false = wrapper(torch.tensor([-2.0, -2.0]))  # Sum = -4 < 0

        # Verify the results are converted to Python booleans
        assert isinstance(result_true, bool)
        assert isinstance(result_false, bool)
        assert result_true is True
        assert result_false is False

        # Test the forward method with the custom condition
        output_true, branch = model(torch.tensor([1.0, 2.0]), return_branch=True)
        assert branch == "custom_branch"

        # Set a default branch for the False case
        default_model = AnotherDummyModel()
        model.set_default_branch(default_model)

        output_false, branch = model(torch.tensor([-2.0, -2.0]), return_branch=True)
        assert branch == "default"


# IdentityModel tests


def test_identity_model():
    """Test identity model passes input unchanged."""
    model = IdentityModel()
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_tensor)

    # Output should be identical to input
    assert torch.allclose(output, input_tensor)

    # Test with different input types
    string_input = "test string"
    assert model(string_input) == string_input

    dict_input = {"key": "value"}
    assert model(dict_input) == dict_input


# Model composition tests


def test_model_composition():
    """Test composition of different generic models."""
    # Create components
    seq_model = SequentialModel([SimpleLayer(add_value=1.0), ScaleLayer(scale_factor=2.0)])

    parallel_model = ParallelModel(branches=[lambda x: x + 3.0, lambda x: x * 0.5], aggregator=lambda outputs: sum(outputs))

    # Compose models: sequential followed by parallel
    composed_model = SequentialModel([seq_model, parallel_model])

    # Apply composed model
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = composed_model(input_tensor)

    # Expected: Let's break it down
    # 1. Sequential: (input + 1) * 2
    intermediate = (input_tensor + 1.0) * 2.0
    # 2. Parallel: (intermediate + 3) + (intermediate * 0.5)
    expected = (intermediate + 3.0) + (intermediate * 0.5)

    assert torch.allclose(output, expected)

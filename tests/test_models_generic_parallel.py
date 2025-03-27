import pytest
from concurrent.futures import ThreadPoolExecutor
from kaira.models.generic.parallel import ParallelModel

def test_parallel_model_initialization():
    model = ParallelModel(max_workers=4)
    assert model.max_workers == 4
    assert model.steps == []

def test_parallel_model_add_step():
    model = ParallelModel()
    step = lambda x: x + 1
    model.add_step(step, name="increment")
    assert len(model.steps) == 1
    assert model.steps[0] == ("increment", step)

def test_parallel_model_remove_step():
    model = ParallelModel()
    step1 = lambda x: x + 1
    step2 = lambda x: x * 2
    model.add_step(step1, name="increment").add_step(step2, name="double")
    model.remove_step(0)
    assert len(model.steps) == 1
    assert model.steps[0] == ("double", step2)

def test_parallel_model_forward():
    model = ParallelModel(max_workers=2)
    step1 = lambda x: x + 1
    step2 = lambda x: x * 2
    model.add_step(step1, name="increment").add_step(step2, name="double")
    input_data = 3
    output = model.forward(input_data)
    assert output == {"increment": 4, "double": 6}

def test_parallel_model_forward_with_exception():
    model = ParallelModel(max_workers=2)
    step1 = lambda x: x + 1
    step2 = lambda x: x / 0  # This will raise an exception
    model.add_step(step1, name="increment").add_step(step2, name="error")
    input_data = 3
    output = model.forward(input_data)
    assert output["increment"] == 4
    assert "Error" in output["error"]

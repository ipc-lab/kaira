import pytest
import torch
from kaira.models.generic.lambda_model import LambdaModel

def test_lambda_model_forward():
    model = LambdaModel(lambda x: x * 2)
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_tensor)
    expected_output = input_tensor * 2
    assert torch.allclose(output, expected_output)

def test_lambda_model_with_args():
    def add_and_scale(x, add_value, scale_factor):
        return (x + add_value) * scale_factor

    model = LambdaModel(add_and_scale)
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_tensor, 5.0, 2.0)
    expected_output = (input_tensor + 5.0) * 2.0
    assert torch.allclose(output, expected_output)

def test_lambda_model_repr():
    model = LambdaModel(lambda x: x * 2, name="DoubleModel")
    expected_repr = "LambdaModel(name=DoubleModel, func=<lambda>)"
    assert repr(model) == expected_repr

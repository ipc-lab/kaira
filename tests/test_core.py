# tests/test_core.py
import pytest
import torch
from kaira.core import BaseChannel, BaseConstraint, BaseMetric, BaseModel, BasePipeline
from torch import nn

class DummyModule(nn.Module):
    def forward(self, x):
        return x

def test_base_channel_abstract_methods():
    """Test that BaseChannel is an abstract class and has abstract methods."""
    with pytest.raises(TypeError):
        BaseChannel()  # Cannot instantiate an abstract class

    class ConcreteChannel(BaseChannel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    channel = ConcreteChannel()
    x = torch.randn(1, 3, 32, 32)
    output = channel(x)
    assert torch.equal(output, x)

def test_base_constraint_abstract_methods():
    """Test that BaseConstraint is an abstract class and has abstract methods."""
    with pytest.raises(TypeError):
        BaseConstraint()  # Cannot instantiate an abstract class

    class ConcreteConstraint(BaseConstraint):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    constraint = ConcreteConstraint()
    x = torch.randn(1, 3, 32, 32)
    output = constraint(x)
    assert torch.equal(output, x)

def test_base_metric_abstract_methods():
    """Test that BaseMetric is an abstract class and has abstract methods."""
    with pytest.raises(TypeError):
        BaseMetric()  # Cannot instantiate an abstract class

    class ConcreteMetric(BaseMetric):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    metric = ConcreteMetric()
    x = torch.randn(1, 3, 32, 32)
    output = metric(x)
    assert torch.equal(output, x)

def test_base_model_abstract_methods():
    """Test that BaseModel is an abstract class and has abstract methods."""
    with pytest.raises(TypeError):
        BaseModel()  # Cannot instantiate an abstract class

    class ConcreteModel(BaseModel):
        @property
        def bandwidth_ratio(self) -> float:
            return 1.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = ConcreteModel()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert torch.equal(output, x)
    assert model.bandwidth_ratio == 1.0

def test_base_pipeline_abstract_methods():
    """Test that BasePipeline is an abstract class and has abstract methods."""
    with pytest.raises(TypeError):
        BasePipeline()  # Cannot instantiate an abstract class

    class ConcretePipeline(BasePipeline):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    pipeline = ConcretePipeline()
    x = torch.randn(1, 3, 32, 32)
    output = pipeline(x)
    assert torch.equal(output, x)

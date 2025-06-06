"""Tests for core modules."""

import pytest
import torch
from torch import nn

from kaira.channels.base import BaseChannel
from kaira.constraints.base import BaseConstraint
from kaira.metrics.base import BaseMetric
from kaira.models.base import BaseModel


class DummyModule(nn.Module):
    """Dummy module for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward."""
        return x


def test_base_channel_abstract_methods():
    """Tests that BaseChannel raises NotImplementedError for abstract methods."""
    with pytest.raises(TypeError):
        BaseChannel()


class ConcreteChannel(BaseChannel):
    """Concrete channel for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward."""
        return x


def test_base_constraint_abstract_methods():
    """Tests that BaseConstraint raises NotImplementedError for abstract methods."""
    with pytest.raises(TypeError):
        BaseConstraint()


class ConcreteConstraint(BaseConstraint):
    """Concrete constraint for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward."""
        return x


def test_base_metric_abstract_methods():
    """Tests that BaseMetric raises NotImplementedError for abstract methods."""
    with pytest.raises(TypeError):
        BaseMetric()


class ConcreteMetric(BaseMetric):
    """Concrete metric for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward."""
        return x


def test_base_model_abstract_methods():
    """Tests that BaseModel raises NotImplementedError for abstract methods."""
    with pytest.raises(TypeError):
        BaseModel()


class ConcreteModel(BaseModel):
    """Concrete model for testing."""

    def bandwidth_ratio(self) -> float:
        """Returns 1.0."""
        return 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward."""
        return x

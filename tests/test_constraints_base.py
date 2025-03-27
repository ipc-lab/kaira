import torch

from kaira.constraints import BaseConstraint


class MockConstraint(BaseConstraint):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


def test_base_constraint_forward():
    constraint = MockConstraint()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    assert torch.allclose(y, x * 2)


def test_get_dimensions_exclude_batch():
    x = torch.randn(32, 4, 128)
    dims = BaseConstraint.get_dimensions(x)
    assert dims == (1, 2)


def test_get_dimensions_include_batch():
    x = torch.randn(32, 4, 128)
    dims = BaseConstraint.get_dimensions(x, exclude_batch=False)
    assert dims == (0, 1, 2)

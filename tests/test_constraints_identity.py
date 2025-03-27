import pytest
import torch
from kaira.constraints import IdentityConstraint

def test_identity_constraint():
    constraint = IdentityConstraint()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    assert torch.allclose(y, x)

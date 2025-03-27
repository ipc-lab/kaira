import pytest
import torch
from kaira.constraints import TotalPowerConstraint, AveragePowerConstraint, PAPRConstraint

def test_total_power_constraint():
    constraint = TotalPowerConstraint(total_power=1.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    assert torch.isclose(torch.sum(torch.abs(y) ** 2), torch.tensor(1.0), rtol=1e-4, atol=1e-4)

def test_average_power_constraint():
    constraint = AveragePowerConstraint(average_power=0.1)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    assert torch.isclose(torch.mean(torch.abs(y) ** 2), torch.tensor(0.1), rtol=1e-4, atol=1e-4)

def test_papr_constraint():
    constraint = PAPRConstraint(max_papr=3.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    avg_power = torch.mean(torch.abs(y) ** 2)
    peak_power = torch.max(torch.abs(y) ** 2)
    papr = peak_power / avg_power
    assert papr <= 3.0

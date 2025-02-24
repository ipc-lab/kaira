# tests/test_models/test_components.py
import pytest
import torch
from kaira.models.components import AFModule
from kaira.constraints import (
    TotalPowerConstraint, 
    AveragePowerConstraint,
    ComplexTotalPowerConstraint,
    ComplexAveragePowerConstraint
)

def test_afmodule_initialization():
    """Test AFModule initialization with valid parameters."""
    N = 64
    csi_length = 1
    module = AFModule(N=N, csi_length=csi_length)
    
    assert module.c_in == N
    assert isinstance(module.layers, torch.nn.Sequential)

def test_afmodule_forward():
    """Test AFModule forward pass."""
    N = 64
    csi_length = 1
    module = AFModule(N=N, csi_length=csi_length)
    
    # Create test inputs
    x = torch.randn(4, N, 32, 32)
    side_info = torch.randn(4, csi_length)
    
    # Test forward pass
    output = module((x, side_info))
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check output values are in valid range (due to sigmoid in last layer)
    assert torch.all(output >= 0) and torch.all(output <= x)

@pytest.mark.parametrize("N,csi_length", [
    (32, 1),
    (64, 2),
    (128, 4)
])
def test_afmodule_different_sizes(N, csi_length):
    """Test AFModule with different sizes for N and CSI length."""
    module = AFModule(N=N, csi_length=csi_length)
    x = torch.randn(4, N, 16, 16)
    side_info = torch.randn(4, csi_length)
    
    output = module((x, side_info))
    assert output.shape == x.shape

@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_total_power_constraint(random_tensor, power):
    """Test total power constraint with different power levels."""
    constraint = TotalPowerConstraint(total_power=power)
    output = constraint(random_tensor)
    
    # Check shape preservation
    assert output.shape == random_tensor.shape
    
    # Check total power constraint is satisfied
    total_power = torch.sum(output ** 2)
    assert torch.isclose(total_power, 
                        torch.tensor(power), 
                        rtol=1e-5)

@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_average_power_constraint(random_tensor, power):
    """Test average power constraint with different power levels."""
    constraint = AveragePowerConstraint(average_power=power)
    output = constraint(random_tensor)
    
    # Check average power constraint is satisfied
    avg_power = torch.mean(output ** 2)
    assert torch.isclose(avg_power, 
                        torch.tensor(power), 
                        rtol=1e-5)

def test_complex_constraints():
    """Test complex-valued power constraints."""
    x = torch.randn(4, 2, 32, 32)  # Complex-valued input
    power = 1.0
    
    # Test complex total power constraint
    total_constraint = ComplexTotalPowerConstraint(total_power=power)
    total_output = total_constraint(x)
    assert torch.isclose(torch.sum(total_output ** 2),
                        torch.tensor(power),
                        rtol=1e-5)
    
    # Test complex average power constraint
    avg_constraint = ComplexAveragePowerConstraint(average_power=power)
    avg_output = avg_constraint(x)
    assert torch.isclose(torch.mean(avg_output ** 2),
                        torch.tensor(power),
                        rtol=1e-5)
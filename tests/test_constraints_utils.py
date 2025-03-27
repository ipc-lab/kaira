import pytest
import torch
from kaira.constraints.utils import create_ofdm_constraints, create_mimo_constraints, combine_constraints, verify_constraint, apply_constraint_chain, measure_signal_properties

def test_create_ofdm_constraints():
    constraints = create_ofdm_constraints(total_power=1.0, max_papr=4.0)
    assert isinstance(constraints, CompositeConstraint)

def test_create_mimo_constraints():
    constraints = create_mimo_constraints(num_antennas=4, uniform_power=0.25, max_papr=4.0)
    assert isinstance(constraints, CompositeConstraint)

def test_combine_constraints():
    power_constraint = TotalPowerConstraint(1.0)
    papr_constraint = PAPRConstraint(4.0)
    combined = combine_constraints([power_constraint, papr_constraint])
    assert isinstance(combined, CompositeConstraint)

def test_verify_constraint():
    power_constraint = TotalPowerConstraint(1.0)
    input_signal = torch.randn(8, 64)
    result = verify_constraint(power_constraint, input_signal, 'power', 1.0)
    assert result['success']

def test_apply_constraint_chain():
    constraints = [
        TotalPowerConstraint(1.0),
        PAPRConstraint(4.0)
    ]
    input_signal = torch.randn(8, 64)
    output = apply_constraint_chain(constraints, input_signal, verbose=True)
    assert output.shape == input_signal.shape

def test_measure_signal_properties():
    signal = torch.randn(1, 64)
    props = measure_signal_properties(signal)
    assert 'mean_power' in props
    assert 'papr' in props

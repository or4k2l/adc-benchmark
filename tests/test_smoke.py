"""
Smoke tests for ADC benchmark suite
Quick validation for CI pipeline
"""
import pytest
import numpy as np

# Tests use direct import from src directory
from adc_benchmark import (
    QuantumCircuit,
    make_mitigation,
    run_state_preparation,
)


def test_quantum_circuit_initialization():
    """Test that QuantumCircuit initializes correctly"""
    circ = QuantumCircuit(N=4, T=50, seed=42)
    assert circ.N == 4
    assert circ.T == 50
    assert circ.seed == 42
    assert 'sx' in circ.ops
    assert 'sy' in circ.ops
    assert 'sz' in circ.ops
    assert len(circ.ops['sx']) == 4


def test_hamiltonian_generation():
    """Test Hamiltonian generation"""
    circ = QuantumCircuit(N=2, T=50, seed=42)
    H = circ.H(t=0, drive=False)
    assert H is not None
    # Check it's a valid quantum object
    assert hasattr(H, 'dims')
    assert H.dims == [[2, 2], [2, 2]]


def test_dissipators():
    """Test dissipator generation"""
    circ = QuantumCircuit(N=2, T=50, seed=42)
    c_ops = circ.diss(γ_relax=1e-5)
    assert len(c_ops) > 0
    # Check they are valid quantum objects
    for c in c_ops:
        assert hasattr(c, 'dims')


def test_mitigation_strategies():
    """Test all mitigation strategies can be created"""
    circ = QuantumCircuit(N=2, T=50, seed=42)
    
    strategies = [
        {'name': 'baseline'},
        {'name': 'adc', 'γm': 1e-5, 'γc': 0.01},
        {'name': 'dd'},
        {'name': 'hybrid', 'γm': 1e-5, 'γc': 0.01}
    ]
    
    for config in strategies:
        mitigation_func, gates = make_mitigation(circ, config)
        assert callable(mitigation_func)
        assert isinstance(gates, int)
        assert gates >= 0


def test_quick_benchmark():
    """Run a very quick benchmark to test end-to-end"""
    config = {'name': 'baseline'}
    result = run_state_preparation(
        T=50, 
        config=config, 
        cycles=2,  # Very short
        num_seeds=2  # Few seeds
    )
    
    assert 'median' in result
    assert 'ci_low' in result
    assert 'ci_high' in result
    assert 'gates' in result
    assert 0 <= result['median'] <= 1
    assert 0 <= result['ci_low'] <= 1
    assert 0 <= result['ci_high'] <= 1
    # Verify median is within CI bounds
    assert result['ci_low'] <= result['median'] <= result['ci_high'], \
        f"Median {result['median']} should be within CI [{result['ci_low']}, {result['ci_high']}]"


def test_import_main():
    """Test that main function can be imported"""
    from adc_benchmark import main
    assert callable(main)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

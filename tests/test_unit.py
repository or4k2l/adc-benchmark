"""
Unit tests for ADC benchmark individual components
"""
import pytest
import numpy as np
import qutip as qt
from src.adc_benchmark import (
    QuantumCircuit,
    depolarize,
    make_mitigation,
    run_state_preparation,
)


# ---------------------------------------------------------------------------
# depolarize()
# ---------------------------------------------------------------------------

def _make_dm(N=2):
    """Helper: return a simple N-qubit pure state density matrix."""
    ket = qt.tensor([qt.basis(2, 0) for _ in range(N)])
    return qt.ket2dm(ket)


def test_depolarize_p0_unchanged():
    """With p=0 the state must be returned unchanged."""
    ρ = _make_dm(N=2)
    out = depolarize(ρ, p=0)
    assert out is ρ


def test_depolarize_p1_maximally_mixed():
    """With p=1 the output must be the maximally mixed state."""
    N = 2
    ρ = _make_dm(N=N)
    out = depolarize(ρ, p=1)
    dim = 2 ** N
    expected = qt.tensor([qt.qeye(2) for _ in range(N)]) / dim
    assert (out - expected).norm() < 1e-10


# ---------------------------------------------------------------------------
# QuantumCircuit.__init__
# ---------------------------------------------------------------------------

def test_quantumcircuit_noise_scales_with_temperature():
    """thermal_base and gamma_phi_base should scale linearly with T/50."""
    circ_50 = QuantumCircuit(N=2, T=50, seed=0)
    circ_100 = QuantumCircuit(N=2, T=100, seed=0)

    assert circ_100.thermal_base == pytest.approx(2 * circ_50.thermal_base)
    assert circ_100.gamma_phi_base == pytest.approx(2 * circ_50.gamma_phi_base)
    # crosstalk_base is temperature-independent
    assert circ_100.crosstalk_base == pytest.approx(circ_50.crosstalk_base)


# ---------------------------------------------------------------------------
# QuantumCircuit.diss()
# ---------------------------------------------------------------------------

def test_diss_low_temperature_nth_zero():
    """At very low T the thermal occupation nth → 0 and no excitation operators added."""
    # T=1 mK → arg = 240/1 = 240 → nth ≈ 0
    circ = QuantumCircuit(N=2, T=1, seed=0)
    c_ops = circ.diss(1e-4)
    # Without nth>1e-6 excitation operators, we get N relaxation + N dephasing = 2*N
    assert len(c_ops) == 2 * circ.N


def test_diss_operator_count_with_nth():
    """At moderate T, nth > 0 → both relaxation and excitation operators are present."""
    # T=50 mK → arg = 240/50 = 4.8 → nth ≈ 0.0083 > 1e-6
    circ = QuantumCircuit(N=2, T=50, seed=0)
    c_ops = circ.diss(1e-4)
    # N relaxation + N excitation + N dephasing = 3*N
    assert len(c_ops) == 3 * circ.N


# ---------------------------------------------------------------------------
# make_mitigation()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy,expected_gates", [
    ("baseline", 0),
    ("adc", 0),
    ("dd", 8 * 2),    # 8 * N with N=2
    ("hybrid", 4 * 2),  # 4 * N with N=2
])
def test_make_mitigation_returns_callable_and_gates(strategy, expected_gates):
    """Each strategy returns a callable and the expected gate overhead."""
    N = 2
    circ = QuantumCircuit(N=N, T=50, seed=0)
    config = {"name": strategy, "γm": 1e-5, "γc": 0.05}
    func, gates = make_mitigation(circ, config, seed_offset=0)

    assert callable(func)
    assert gates == expected_gates


def test_make_mitigation_does_not_mutate_seed():
    """make_mitigation must not change circ.seed."""
    circ = QuantumCircuit(N=2, T=50, seed=99)
    original_seed = circ.seed
    make_mitigation(circ, {"name": "baseline"}, seed_offset=3)
    assert circ.seed == original_seed


# ---------------------------------------------------------------------------
# run_state_preparation()
# ---------------------------------------------------------------------------

def test_run_state_preparation_raises_without_config():
    """Calling without config must raise ValueError."""
    with pytest.raises(ValueError, match="config must be provided"):
        run_state_preparation(T=50, config=None, cycles=1, num_seeds=1)


def test_run_state_preparation_returns_expected_keys():
    """run_state_preparation returns a dict with the required keys."""
    config = {"name": "baseline", "γm": 1e-5}
    result = run_state_preparation(T=50, config=config, cycles=1, num_seeds=2)

    assert set(result.keys()) == {"median", "ci_low", "ci_high", "gates", "median_purity", "runtime_seconds"}


def test_run_state_preparation_ci_ordering():
    """ci_low ≤ median ≤ ci_high must always hold."""
    config = {"name": "baseline", "γm": 1e-5}
    result = run_state_preparation(T=50, config=config, cycles=1, num_seeds=2)

    assert result["ci_low"] <= result["median"] <= result["ci_high"]

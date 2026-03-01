"""
Plausibility tests for ADC benchmark statistical correctness
"""
import pytest
from src.adc_benchmark import run_state_preparation


def test_run_state_preparation_plausibility_baseline():
    """
    Validate statistical correctness of run_state_preparation output.

    Runs a short baseline benchmark and asserts:
    - median fidelity is in (0, 1]
    - CI bounds are ordered correctly (ci_low ≤ median ≤ ci_high)
    - CI width is meaningful (< 1.0)
    - gate overhead is 0 for baseline
    """
    result = run_state_preparation(
        T=50,
        config={"name": "baseline"},
        cycles=2,
        num_seeds=3,
    )

    assert result["median"] > 0.0, "median fidelity must be positive"
    assert result["median"] <= 1.0, "median fidelity must not exceed 1.0"
    assert result["ci_low"] <= result["median"], "ci_low must be ≤ median"
    assert result["median"] <= result["ci_high"], "median must be ≤ ci_high"
    assert result["ci_high"] - result["ci_low"] < 1.0, "CI width must be < 1.0"
    assert result["gates"] == 0, "baseline strategy must have 0 gate overhead"

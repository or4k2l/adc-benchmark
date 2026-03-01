# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-01

### Added
- Complete ADC benchmark implementation with Baseline, DD (XY8), ADC, and Hybrid strategies
- Temperature-dependent optimization via grid search over γ_compute
- Statistical validation with bootstrap confidence intervals (95% CI)
- `temperature_sweep()` function comparing all strategies across temperatures
- `plot_temperature_sweep()` for publication-quality output plots
- CLI interface with `--quick` and `--output` flags
- GitHub Actions CI workflow running the smoke test in a conda environment
- End-to-end smoke test (`tests/test_smoke.py`) verifying quick-mode execution and plot output
- Unit tests (`tests/test_unit.py`) covering depolarize, QuantumCircuit, make_mitigation, and run_state_preparation
- Fixed README run instructions with correct `python -m src` commands
- Fixed 13 code quality and correctness issues across the benchmark suite including thermal overflow protection, bootstrap median fix, config validation, and operator count correctness

### Fixed
- Thermal occupation overflow for very low temperatures (arg clamped to 700)
- Bootstrap now samples the median (not mean) for correct statistical interpretation
- `run_state_preparation` raises `ValueError` when config is missing
- Dephasing operator count corrected for temperature-dependent dissipator

# ADC-Benchmark

![CI](https://github.com/or4k2l/adc-benchmark/actions/workflows/ci.yml/badge.svg)

Adaptive Dissipation Control — Comprehensive benchmark suite for superconducting qubits.

<img width="2084" height="1182" alt="adc_temperature_sweep" src="https://github.com/user-attachments/assets/ec1c81e2-e15c-4fcd-ac8f-c12c56504005" />

## Features
- Baseline, Dynamical Decoupling (XY8), ADC, Hybrid
- Temperature-dependent optimization
- Bootstrap confidence intervals
- Quick/Full modes for CI vs. full experiments

## Requirements
- Python 3.10+
- Recommended: conda (qutip is easiest to install from conda-forge)

## Quick install (recommended)
```bash
# create env (recommended)
conda create -n adcbench -c conda-forge python=3.10 qutip matplotlib numpy
conda activate adcbench
```

Or with pip (may fail or be slow for qutip):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
- Quick smoke run (fast, runs 50 mK quick mode, writes `adc_temperature_sweep.png`):
```bash
python -m src --quick
```
Or run the script directly:
```bash
python src/adc_benchmark.py --quick
```
- Full benchmark:
```bash
python -m src
```
Or:
```bash
python src/adc_benchmark.py
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--quick` | Run abbreviated benchmark (50 mK only, ~1-2 min) |
| `--output PATH` | Output path for the plot (default: `adc_temperature_sweep.png`) |
| `--save-results` | Save results as both JSON **and** CSV to `--results-dir` |
| `--results-dir DIR` | Directory for saved results (default: `results`) |
| `--json` | Save results as JSON only |
| `--csv` | Save results as CSV only |

Example — save JSON after a quick run:
```bash
python -m src --quick --json
```

## Testing & CI
- A lightweight smoke test is provided in `tests/`. Run it locally with:
```bash
pytest -q tests/test_smoke.py
```
- GitHub Actions runs the smoke test in a small conda environment.

## License
- MIT (see LICENSE file)

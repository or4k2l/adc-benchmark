# ADC-Benchmark

Adaptive Dissipation Control — Comprehensive benchmark suite for superconducting qubits.

Features
- Baseline, Dynamical Decoupling (XY8), ADC, Hybrid
- Temperature-dependent optimization
- Bootstrap confidence intervals
- Quick/Full modes for CI vs. full experiments

Requirements
- Python 3.10+
- Recommended: conda (qutip is easiest to install from conda-forge)

Quick install (recommended)
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

Run
- Quick smoke run (fast, used by CI):
```bash
python -m adc_benchmark --quick
```
- Full benchmark:
```bash
python -m adc_benchmark
```

Testing & CI
- A lightweight smoke test is provided in `tests/`. GitHub Actions runs the smoke test in a small conda environment.

Files to pay attention to
- `adc_benchmark.py` (main script) — move top-level prints under `main()` and add `--quick` flag.
- `requirements.txt`, `LICENSE`, `.github/workflows/ci.yml`

License
- MIT (see LICENSE file)

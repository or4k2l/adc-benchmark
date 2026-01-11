# Adaptive Dissipation Control — Benchmark Suite

This repository contains a benchmark suite to compare error mitigation strategies
for superconducting qubits: Baseline, Dynamical Decoupling (DD), Adaptive Dissipation Control (ADC) and Hybrid.

## Requirements

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `qutip` can be heavy to build on some systems — prefer conda on Linux/macOS:
`conda install -c conda-forge qutip matplotlib numpy`

## Usage

Run the benchmark (long; full run ≈ 25 min on a modern desktop):

```bash
python3 src/adc_benchmark.py
```

A figure `adc_temperature_sweep.png` will be saved to the repository.

## Files

- `src/adc_benchmark.py` — main benchmark script
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores build and data files
- `LICENSE` — MIT license

## License

MIT — see LICENSE file.

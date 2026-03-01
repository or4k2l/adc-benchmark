"""
pytest configuration: ensure the project root is on sys.path so that
`from src.adc_benchmark import ...` works regardless of the directory
pytest is invoked from (e.g. Google Colab, Jupyter, local without install).
"""
import sys
from pathlib import Path

# Insert project root (parent of this file's directory) at the front of sys.path.
# This makes `src.*` importable without requiring `pip install -e .`.
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Smoke tests for ADC benchmark suite
Quick validation for CI pipeline
"""
import os
import subprocess
import sys
from pathlib import Path

# Project root is two levels up from this file (tests/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Constants
SMOKE_TEST_TIMEOUT = 180  # seconds - timeout for quick mode execution
MODULE_NAME = "src"  # The main module name to run


def test_quick_mode_end_to_end():
    """
    End-to-end smoke test: runs the benchmark in quick mode and verifies output.

    This test:
    - Runs `python -m src --quick` using subprocess
    - Sets cwd and PYTHONPATH to the project root so `src` is always findable
      regardless of the directory pytest is invoked from (e.g. Google Colab)
    - Uses a 180 second timeout to prevent CI hanging
    - Verifies the process exits with code 0
    - Verifies that adc_temperature_sweep.png is created with non-zero size
    - Cleans up the generated file
    """
    output_file = PROJECT_ROOT / "adc_temperature_sweep.png"

    # Clean up any existing file from previous runs
    if output_file.exists():
        output_file.unlink()

    # Run the benchmark in quick mode from the project root.
    # Setting cwd=PROJECT_ROOT ensures `python -m src` can find the src package.
    # Setting PYTHONPATH redundantly covers edge cases where cwd alone is not enough.
    result = subprocess.run(
        [sys.executable, "-m", MODULE_NAME, "--quick"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        timeout=SMOKE_TEST_TIMEOUT,
        capture_output=True,
        text=True,
    )

    # Verify the process completed successfully
    assert result.returncode == 0, (
        f"Command failed with exit code {result.returncode}.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )

    # Verify the output file was created
    assert output_file.exists(), (
        f"Expected output file '{output_file}' was not created.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )

    # Verify the file has non-zero size
    file_size = output_file.stat().st_size
    assert file_size > 0, (
        f"Output file '{output_file}' exists but is empty (size: {file_size} bytes)"
    )

    # Clean up the generated file
    output_file.unlink()

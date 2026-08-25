"""
Run every example script, to make sure the documentation does not rot.

The examples are the most visible part of the project, so a change that breaks
one should fail the test suite rather than being found by the next reader.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(path for path in EXAMPLES.rglob("*.py") if path.name != "_common.py")


@pytest.mark.skipif(not SCRIPTS, reason="the examples directory is not present")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_example_runs(script, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{script.name} failed:\n{result.stdout}\n{result.stderr}"
    # Every example prints something worth reading.
    assert result.stdout.strip()


def test_every_example_is_listed_in_the_readme():
    readme = (EXAMPLES / "README.md").read_text()
    for script in SCRIPTS:
        relative = script.relative_to(EXAMPLES).as_posix()
        assert relative in readme, f"{relative} is missing from examples/README.md"

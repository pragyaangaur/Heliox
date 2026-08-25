"""Shared setup for the example scripts."""

import sys
from pathlib import Path

import matplotlib

# The examples are meant to run unattended, including in CI, so they never open
# a window; each one saves a figure instead.
matplotlib.use("Agg")

#: The root of the repository.
ROOT = Path(__file__).resolve().parent.parent

# Let the examples run straight from a checkout, without heliox being installed.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def figure_path(name):
    """Return the path to write a figure to, creating the directory."""
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    return directory / name


def save(figure, name):
    """Save a figure and report where it went."""
    path = figure_path(name)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")
    return path

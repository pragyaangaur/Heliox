"""Check a built Heliox wheel in a clean virtual environment.

The release workflow runs this after every wheel build. It creates a new
virtual environment, installs the wheel with binary packages only (so no Rust
toolchain or source build can be involved), and then runs the checks from a
new temporary directory, so that the source tree is never imported.

usage: python smoke_test.py path/to/heliox-X.Y.Z-tag.whl
"""

import os
import pathlib
import re
import subprocess
import sys
import tempfile
import venv

# Run inside the clean environment. It checks the version, that the package
# comes from the environment rather than the source tree, that every compiled
# submodule imports, and that a sample map can be loaded and read.
CHECKS = r"""
import importlib
import os
import sys
import types

expected, env_dir = sys.argv[1], sys.argv[2]

import heliox

assert heliox.__version__ == expected, (heliox.__version__, expected)
installed = os.path.realpath(heliox.__file__)
assert installed.startswith(os.path.realpath(env_dir)), installed
print("heliox", heliox.__version__, "from", installed)

import heliox._core as core

submodules = sorted(name for name, value in vars(core).items() if isinstance(value, types.ModuleType))
assert submodules, "heliox._core has no submodules"
for name in submodules:
    importlib.import_module(f"heliox._core.{name}")
print("compiled submodules:", ", ".join(submodules))

import heliox.map
from heliox.data.sample import AIA_171_IMAGE

aia = heliox.map.Map(AIA_171_IMAGE)
assert aia.data.shape == (512, 512), aia.data.shape
print("sample map:", type(aia).__name__, aia.data.shape)
"""


def run(command, **kwargs):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    wheel = pathlib.Path(sys.argv[1]).resolve()
    match = re.match(r"heliox-([^-]+)-", wheel.name)
    if match is None:
        sys.exit(f"not a heliox wheel: {wheel.name}")
    expected_version = match.group(1)

    work = pathlib.Path(tempfile.mkdtemp(prefix="heliox-smoke-"))
    env_dir = work / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--only-binary",
            ":all:",
            str(wheel),
            "pytest",
        ]
    )

    # Every check runs from an empty directory outside the source tree.
    checks_dir = work / "checks"
    checks_dir.mkdir()
    run([str(python), "-c", CHECKS, expected_version, str(env_dir)], cwd=checks_dir)
    run(
        [
            str(python),
            "-m",
            "pytest",
            "--pyargs",
            "heliox.tests.test_core",
            "heliox.sun",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=checks_dir,
    )
    print(f"smoke test passed for {wheel.name}")


if __name__ == "__main__":
    main()

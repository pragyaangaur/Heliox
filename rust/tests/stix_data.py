"""Real STIX visibilities for the MEM tests.

The data are the two L1 files of the 2021-09-23 flare from stixpy's imaging demo. Point the
environment variable ``HELIOX_STIX_DATA`` at the directory that holds them (they are downloaded
there when missing) to run the tests that need them. The set-up of the demo is in
``profiling/stix/common.py`` at the repository root.
"""

import os
import sys
from pathlib import Path

import pytest

PROFILING = Path(__file__).resolve().parents[2] / "profiling" / "stix"


def load_pipeline():
    data_dir = os.environ.get("HELIOX_STIX_DATA")
    if not data_dir:
        pytest.skip("HELIOX_STIX_DATA is not set")
    pytest.importorskip("stixpy")
    pytest.importorskip("xrayvision")
    sys.path.insert(0, str(PROFILING))
    import common

    return common.Pipeline(data_dir).prepare()


def run_mem(pipeline, size, mem_function, maxiter=1000, pixel=2.0):
    """The MEM stage of the demo with the given ``mem`` function, so original and Rust share inputs."""
    import astropy.units as u
    import numpy as np
    from xrayvision.mem import resistant_mean

    vis = pipeline.vis10_3
    snr, _ = resistant_mean((np.abs(vis.visibilities) / vis.amplitude_uncertainty).flatten(), 3)
    lam = 2 / (snr**2 + 90)
    return mem_function(
        vis,
        shape=[size, size] * u.pixel,
        pixel_size=[pixel, pixel] * u.arcsec / u.pixel,
        percent_lambda=lam,
        maxiter=maxiter,
    )

"""Shared fixtures. The real velocity field comes from the sunkit-image test data."""

import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _load_real():
    pytest.importorskip("sunkit_image.asda")
    pytest.importorskip("skimage")
    import sunkit_image
    from astropy.io import fits

    path = Path(sunkit_image.__file__).parent / "data" / "test" / "asda_vxvy.fits"
    with fits.open(path) as hdus:
        data = {h.name.lower(): np.array(h.data) for h in hdus[1:]}
    return data["vx"], data["vy"], data["data"]


@pytest.fixture(scope="session")
def real_field():
    """The real 512 x 1024 float32 (big endian) velocity field and its image."""
    return _load_real()


@pytest.fixture(scope="session")
def real_gamma_original(real_field):
    """The original function on the full real field, run once for the whole session (about 5 s)."""
    asda = pytest.importorskip("sunkit_image.asda")
    vx, vy, _ = real_field
    return asda.calculate_gamma_values(vx, vy)


def assert_same(expected, actual):
    """Strict comparison.

    The Rust kernel repeats the NumPy operations in the same order, including NumPy's pairwise
    summation, so the results are expected to agree bit for bit. That is stronger than the
    rtol 1e-12 asked for float64, and it is also what float32 inputs need, because the original
    keeps window values, their mean and the vector norm in float32 and the rest in float64.
    Anything looser would hide an operation that is not mirrored.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    assert expected.shape == actual.shape
    assert expected.dtype.newbyteorder("=") == actual.dtype.newbyteorder("=")
    np.testing.assert_array_equal(actual, expected)

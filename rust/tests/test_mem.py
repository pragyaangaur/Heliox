"""Direct comparison of heliox_rs.mem with xrayvision.mem (0.2.1).

Everything is compared for exact equality. The Rust kernel repeats the original operations in the
original order, including NumPy's complex division and Rust's `ln`, which gives the same bits as
`np.log` on this machine (see `test_ln_matches_numpy_bit_for_bit`). A last bit difference would
mean that an operation is not mirrored, so no tolerance is used anywhere.
"""

import inspect

import astropy.units as u
import numpy as np
import pytest
from stix_data import load_pipeline, run_mem

xm = pytest.importorskip("xrayvision.mem")

from heliox_rs import _core  # noqa: E402
from heliox_rs import mem as rs  # noqa: E402

COUNT = u.ct
RATE = u.ct / (u.keV * u.s * u.cm**2)


def assert_identical(expected, actual):
    """Same unit, dtype, shape and values (NaN equal to NaN). Zero signs are not compared."""
    assert isinstance(actual, u.Quantity)
    assert actual.unit == expected.unit
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.value, expected.value)


def random_arguments(seed, size, complex_input):
    rng = np.random.default_rng(seed)
    magnitude = 10.0 ** rng.uniform(-4, 3, size)
    sign = np.where(rng.random(size) < 0.3, -1.0, 1.0)
    y = magnitude * sign
    y[rng.random(size) < 0.05] = 0.0
    m = 10.0 ** rng.uniform(-2, 2)
    # Some y far below m and some far above it.
    y[rng.random(size) < 0.1] = m * 1e-8
    if complex_input:
        imag = np.where(rng.random(size) < 0.5, 0.0, -0.0)
        y = y + 1j * imag
        assert y.dtype == np.complex128
    lamba = 10.0 ** rng.uniform(-6, 2) * u.s
    Lip = 10.0 ** rng.uniform(-3, 3) * u.s / COUNT
    return y * COUNT, m * COUNT, lamba, Lip


# ---------------------------------------------------------------- the logarithm


def test_ln_matches_numpy_bit_for_bit():
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [
            10.0 ** rng.uniform(-300, 300, 1_000_000),
            10.0 ** rng.uniform(-3, 3, 1_000_000),
            1.0 + rng.uniform(-1e-3, 1e-3, 500_000),
            np.array([1e-24, 0.5, 1.0, 2.0, np.e, 1e-300, 5e-324, np.inf, 0.0, -1.0, np.nan]),
        ]
    )
    with np.errstate(all="ignore"):
        expected = np.log(values)
    np.testing.assert_array_equal(_core.ln_f64(values), expected)


def test_real_part_of_complex_log_is_the_real_log():
    rng = np.random.default_rng(1)
    values = 10.0 ** rng.uniform(-30, 30, 500_000)
    np.testing.assert_array_equal(np.log(values.astype(complex)).real, np.log(values))


def test_complex_division_by_a_real_scalar_is_a_reciprocal_product():
    # This is why the kernel has a reciprocal_divide mode. It is not a rounding accident, it is the
    # algorithm NumPy uses for a complex divisor with zero imaginary part.
    rng = np.random.default_rng(2)
    c = 10.0 ** rng.uniform(-30, 30, 500_000)
    m = 15.80862327479201
    np.testing.assert_array_equal((c.astype(complex) / m).real, c * (1.0 / m))
    assert np.mean(c * (1.0 / m) != c / m) > 0.1


# ---------------------------------------------------------------- proximal_entropy, random input


SIZES = [1, 2, 3, 17, 1000, 2047, 2048, 2049, 3073, 4225, 16641]


@pytest.mark.parametrize("complex_input", [False, True], ids=["float64", "complex128"])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("size", SIZES)
def test_random_input_matches_original(size, seed, complex_input):
    y, m, lamba, Lip = random_arguments(seed, size, complex_input)
    expected = xm._proximal_entropy(y, m, lamba, Lip)
    assert_identical(expected, rs.proximal_entropy(y, m, lamba, Lip))


@pytest.mark.parametrize("complex_input", [False, True], ids=["float64", "complex128"])
def test_largest_size_matches_original(complex_input):
    y, m, lamba, Lip = random_arguments(5, 66049, complex_input)
    assert_identical(xm._proximal_entropy(y, m, lamba, Lip), rs.proximal_entropy(y, m, lamba, Lip))


@pytest.mark.parametrize("tol", [1e-3, 1e-6, 1e-10, 1e-12])
def test_other_tolerances(tol):
    y, m, lamba, Lip = random_arguments(3, 500, True)
    assert_identical(xm._proximal_entropy(y, m, lamba, Lip, tol), rs.proximal_entropy(y, m, lamba, Lip, tol))


@pytest.mark.parametrize("complex_input", [False, True], ids=["float64", "complex128"])
def test_f_c_exactly_zero(complex_input):
    # With lamba = 0 the function is c - y. For m = 4 and y = 2 the first midpoint is exactly 2,
    # so f_c == 0 sets both ends at once.
    y = np.array([2.0, 2.0, 1.0, 3.0, 0.5])
    if complex_input:
        y = y + 0j
    args = (y * COUNT, 4.0 * COUNT, 0.0 * u.s, 1.0 * u.s / COUNT)
    expected = xm._proximal_entropy(*args)
    assert expected.value.real[0] == 2.0
    assert_identical(expected, rs.proximal_entropy(*args))


def test_y_below_m_and_zeros():
    y = np.array([0.0, -1.0, 1e-30, 1e-8, 5.0, 100.0]) * COUNT
    args = (y, 3.0 * COUNT, 0.2 * u.s, 0.5 * u.s / COUNT)
    assert_identical(xm._proximal_entropy(*args), rs.proximal_entropy(*args))
    args_c = (y.astype(complex), *args[1:])
    assert_identical(xm._proximal_entropy(*args_c), rs.proximal_entropy(*args_c))


def test_m_array_of_the_shape_of_y():
    rng = np.random.default_rng(4)
    y = rng.normal(size=300) * 10 * COUNT
    m = (0.1 + rng.random(300) * 20) * COUNT
    args = (y, m, 0.3 * u.s, 0.7 * u.s / COUNT)
    assert_identical(xm._proximal_entropy(*args), rs.proximal_entropy(*args))


def test_two_dimensional_and_zero_dimensional_shapes():
    rng = np.random.default_rng(6)
    y = (rng.normal(size=(40, 25)) * 5 + 1j * 0.0) * COUNT
    args = (y, 2.0 * COUNT, 0.3 * u.s, 0.7 * u.s / COUNT)
    assert_identical(xm._proximal_entropy(*args), rs.proximal_entropy(*args))
    y0 = (3.5 + 0j) * COUNT
    args = (y0, 2.0 * COUNT, 0.3 * u.s, 0.7 * u.s / COUNT)
    assert_identical(xm._proximal_entropy(*args), rs.proximal_entropy(*args))


def test_non_contiguous_input():
    rng = np.random.default_rng(7)
    y = ((rng.normal(size=(60, 60)) * 4 + 0j) * COUNT)[::2, ::3]
    assert not y.flags["C_CONTIGUOUS"]
    args = (y, 2.0 * COUNT, 0.3 * u.s, 0.7 * u.s / COUNT)
    assert_identical(xm._proximal_entropy(*args), rs.proximal_entropy(*args))


def test_nan_m_stops_at_once_like_the_original():
    y = np.array([1.0, 2.0, 3.0]) * COUNT
    args = (y, np.nan * COUNT, 0.3 * u.s, 0.7 * u.s / COUNT)
    with np.errstate(all="ignore"):
        expected = xm._proximal_entropy(*args)
    assert_identical(expected, rs.proximal_entropy(*args))
    assert np.isnan(expected.value).all()


def test_nan_in_y_raises_where_the_original_never_ends():
    y = np.array([1.0, np.nan, 3.0]) * COUNT
    with pytest.raises(RuntimeError, match="did not converge"):
        rs.proximal_entropy(y, 2.0 * COUNT, 0.3 * u.s, 0.7 * u.s / COUNT)


def test_bracket_that_cannot_shrink_to_tol_raises():
    # Above about 8e5 one step of the float64 grid is wider than the default tol of 1e-10, so the
    # bracket stops shrinking and the original loops for ever. The Rust version raises instead.
    y = np.array([1.0e7, 3.0]) * COUNT
    with pytest.raises(RuntimeError, match="did not converge"):
        rs.proximal_entropy(y, 2.0 * COUNT, 0.3 * u.s, 0.7 * u.s / COUNT)


def test_result_is_a_new_array():
    y, m, lamba, Lip = random_arguments(8, 100, True)
    before = y.value.copy()
    out = rs.proximal_entropy(y, m, lamba, Lip)
    out.value[:] = 0
    np.testing.assert_array_equal(y.value, before)


# ---------------------------------------------------------------- inputs sent to the original


def test_inputs_outside_the_kernel_use_the_original():
    rng = np.random.default_rng(9)
    y = rng.normal(size=200) * 10 * COUNT
    m, lamba, Lip = 3.0 * COUNT, 0.3 * u.s, 0.7 * u.s / COUNT
    cases = {
        "complex with imaginary part": (y.astype(complex) + 1j * 0.5 * COUNT, m, lamba, Lip),
        "float32": (y.astype(np.float32), m, lamba, Lip),
        "m in another unit": (y, 3.0e-3 * u.kct, lamba, Lip),
        "lamba / Lip needs a scale factor": (y, m, 300.0 * u.ms, Lip),
    }
    # A loose tol keeps the float32 case finite: its grid step is far wider than the default 1e-10.
    for args in cases.values():
        with np.errstate(all="ignore"):
            expected = xm._proximal_entropy(*args, 1e-4)
        got = rs.proximal_entropy(*args, 1e-4)
        assert_identical(expected, got)


def test_unit_of_result_follows_y():
    y, m, lamba, Lip = random_arguments(10, 50, True)
    assert rs.proximal_entropy(y, m, lamba, Lip).unit == COUNT


# ---------------------------------------------------------------- real MEM arguments


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


@pytest.fixture(scope="module")
def captured_calls(pipeline):
    """Arguments of real `_proximal_entropy` calls: every 40th call of a 65 x 65 MEM run."""
    calls = []
    seen = {"n": 0}
    original = xm._proximal_entropy

    def recorder(y, m, lamba, Lip, tol=10**-10):
        seen["n"] += 1
        result = original(y, m, lamba, Lip, tol)
        if seen["n"] % 40 == 1:
            calls.append((y.copy(), m.copy(), lamba.copy(), Lip.copy(), tol, result.copy()))
        return result

    xm._proximal_entropy = recorder
    try:
        run_mem(pipeline, 65, xm.mem)
    finally:
        xm._proximal_entropy = original
    assert xm._proximal_entropy is original
    assert len(calls) >= 8
    return calls


def test_captured_real_calls(captured_calls):
    for y, m, lamba, Lip, tol, expected in captured_calls:
        assert y.dtype == np.complex128 and y.shape == (65, 65)
        assert_identical(expected, rs.proximal_entropy(y, m, lamba, Lip, tol))


def test_captured_real_calls_with_float_part_only(captured_calls):
    # The same arguments as float64, which takes the real division path.
    for y, m, lamba, Lip, tol, _ in captured_calls[:4]:
        yr = y.real.copy()
        assert_identical(xm._proximal_entropy(yr, m, lamba, Lip, tol), rs.proximal_entropy(yr, m, lamba, Lip, tol))


# ---------------------------------------------------------------- full images


@pytest.mark.parametrize("size", [65, 129])
def test_full_mem_image_matches_original(pipeline, size):
    expected = run_mem(pipeline, size, xm.mem)
    actual = run_mem(pipeline, size, rs.mem)
    assert xm._proximal_entropy is not rs.proximal_entropy
    assert actual.data.dtype == expected.data.dtype
    assert actual.data.shape == (size, size)
    assert actual.unit == expected.unit
    np.testing.assert_array_equal(actual.data, expected.data)
    assert dict(actual.meta) == dict(expected.meta)


def test_full_mem_image_without_map(pipeline):
    import astropy.units as u

    def bare(function):
        return lambda *a, **k: function(*a, map=False, **k)

    expected = run_mem(pipeline, 65, bare(xm.mem))
    actual = run_mem(pipeline, 65, bare(rs.mem))
    assert isinstance(actual, u.Quantity) == isinstance(expected, u.Quantity)
    np.testing.assert_array_equal(np.asarray(getattr(actual, "value", actual)), np.asarray(getattr(expected, "value", expected)))


# ---------------------------------------------------------------- the drop-in mem


def test_mem_signature_matches_original():
    ours = inspect.signature(rs.mem).parameters
    theirs = inspect.signature(xm.mem).parameters
    assert list(ours) == list(theirs)
    for name in theirs:
        assert ours[name].kind == theirs[name].kind, name
        default = theirs[name].default
        if isinstance(default, u.Quantity):
            assert ours[name].default == default
        else:
            assert ours[name].default == default or ours[name].default is default, name


def test_proximal_entropy_signature_matches_original():
    ours = inspect.signature(rs.proximal_entropy)
    theirs = inspect.signature(xm._proximal_entropy)
    assert [(p.name, p.kind, p.default) for p in ours.parameters.values()] == [
        (p.name, p.kind, p.default) for p in theirs.parameters.values()
    ]


def test_original_is_restored_after_mem():
    before = xm._proximal_entropy
    assert before is not rs.proximal_entropy
    seen = {}
    real_mem = xm.mem

    def spy(*args, **kwargs):
        seen["inside"] = xm._proximal_entropy
        return "done"

    xm.mem = spy
    try:
        assert rs.mem(None, None, None) == "done"
    finally:
        xm.mem = real_mem
    assert seen["inside"] is rs.proximal_entropy
    assert xm._proximal_entropy is before


def test_original_is_restored_after_an_exception():
    before = xm._proximal_entropy
    real_mem = xm.mem

    def boom(*args, **kwargs):
        raise ZeroDivisionError("boom")

    xm.mem = boom
    try:
        with pytest.raises(ZeroDivisionError):
            rs.mem(None, None, None)
    finally:
        xm.mem = real_mem
    assert xm._proximal_entropy is before


def test_mem_passes_arguments_through():
    seen = {}
    real_mem = xm.mem

    def spy(vis, shape, pixel_size, **kwargs):
        seen.update(vis=vis, shape=shape, pixel_size=pixel_size, **kwargs)

    xm.mem = spy
    try:
        rs.mem("v", "s", "p", percent_lambda=1 * u.percent, maxiter=7, tol=1e-5, map=False, total_flux=3 * COUNT)
    finally:
        xm.mem = real_mem
    assert seen == {
        "vis": "v",
        "shape": "s",
        "pixel_size": "p",
        "percent_lambda": 1 * u.percent,
        "maxiter": 7,
        "tol": 1e-5,
        "map": False,
        "total_flux": 3 * COUNT,
    }


def test_mem_validates_units_like_the_original():
    # The original is wrapped in astropy's quantity_input, and so is the call it makes.
    with pytest.raises(u.UnitsError):
        rs.mem(None, [65, 65] * u.arcsec, [2, 2] * u.arcsec / u.pixel)

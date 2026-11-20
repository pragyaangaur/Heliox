"""Direct comparison of heliox_rs.asda with sunkit_image.asda (0.7.0)."""

import numpy as np
import pytest
from conftest import assert_same
from sunkit_image import asda

from heliox_rs.asda import calculate_gamma_values as rs_gamma


def random_field(seed, shape, dtype):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(dtype), rng.normal(size=shape).astype(dtype)


# ---------------------------------------------------------------- real data


def test_real_field_full(real_field, real_gamma_original):
    vx, vy, _ = real_field
    out = rs_gamma(vx, vy)
    assert out.shape == (512, 1024, 2)
    assert_same(real_gamma_original, out)


@pytest.mark.parametrize("r", [1, 2, 5])
def test_real_field_crop_other_radii(real_field, r):
    vx, vy, _ = real_field
    cx, cy = vx[100:228, 300:460], vy[100:228, 300:460]
    assert_same(asda.calculate_gamma_values(cx, cy, r=r), rs_gamma(cx, cy, r=r))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("factor", [1, 2])
def test_real_field_crop_dtypes_and_factor(real_field, dtype, factor):
    vx, vy, _ = real_field
    cx, cy = vx[200:260, 500:560].astype(dtype), vy[200:260, 500:560].astype(dtype)
    assert_same(asda.calculate_gamma_values(cx, cy, factor=factor), rs_gamma(cx, cy, factor=factor))


# ---------------------------------------------------------------- random fields


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("shape", [(30, 30), (17, 41), (41, 17), (25, 26), (63, 20)])
@pytest.mark.parametrize("r", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_fields(seed, shape, r, dtype):
    vx, vy = random_field(seed, shape, dtype)
    expected = asda.calculate_gamma_values(vx, vy, r=r)
    assert expected.dtype == dtype
    assert_same(expected, rs_gamma(vx, vy, r=r))


@pytest.mark.parametrize("seed", [3, 4])
@pytest.mark.parametrize("shape", [(24, 24), (19, 31), (31, 19)])
@pytest.mark.parametrize("r", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_fields_factor_two(seed, shape, r, dtype):
    vx, vy = random_field(seed, shape, dtype)
    expected = asda.calculate_gamma_values(vx, vy, factor=2, r=r)
    out = rs_gamma(vx, vy, factor=2, r=r)
    assert out.shape == (2 * shape[0], 2 * shape[1], 2)
    assert_same(expected, out)


def test_factor_three():
    vx, vy = random_field(11, (22, 18), np.float64)
    assert_same(asda.calculate_gamma_values(vx, vy, factor=3, r=2), rs_gamma(vx, vy, factor=3, r=2))


def test_factor_greater_than_one_only_fills_the_unmagnified_block():
    # A quirk of the original that is kept: the loop bounds come from the unmagnified shape.
    vx, vy = random_field(5, (30, 30), np.float64)
    out = rs_gamma(vx, vy, factor=2, r=3)
    assert np.all(out[27:] == 0)
    assert np.all(out[:, 27:] == 0)
    assert np.any(out[3:27, 3:27] != 0)


def test_zero_and_nonpositive_factor_behave_like_one():
    vx, vy = random_field(6, (20, 20), np.float64)
    for factor in (0, -2):
        assert_same(asda.calculate_gamma_values(vx, vy, factor=factor), rs_gamma(vx, vy, factor=factor))


def test_r_zero():
    vx, vy = random_field(7, (9, 9), np.float64)
    assert_same(asda.calculate_gamma_values(vx, vy, r=0), rs_gamma(vx, vy, r=0))


# ---------------------------------------------------------------- tiny shapes


@pytest.mark.parametrize("r", [1, 2, 3, 5])
@pytest.mark.parametrize("extra", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 3)])
def test_shapes_at_and_just_above_the_minimum(r, extra):
    shape = (2 * r + 1 + extra[0], 2 * r + 1 + extra[1])
    vx, vy = random_field(8, shape, np.float64)
    expected = asda.calculate_gamma_values(vx, vy, r=r)
    assert np.count_nonzero(expected) > 0
    assert_same(expected, rs_gamma(vx, vy, r=r))


@pytest.mark.parametrize("shape", [(6, 20), (20, 6), (7, 6), (1, 1), (0, 5)])
def test_too_small_raises_index_error_like_the_original(shape):
    vx, vy = random_field(9, shape, np.float64)
    with pytest.raises(IndexError):
        asda.calculate_gamma_values(vx, vy, r=3)
    with pytest.raises(IndexError):
        rs_gamma(vx, vy, r=3)


# ---------------------------------------------------------------- NaN and inf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fraction", [0.001, 0.05, 0.5])
def test_scattered_nans(dtype, fraction):
    vx, vy = random_field(10, (40, 36), dtype)
    rng = np.random.default_rng(99)
    vx[rng.random(vx.shape) < fraction] = np.nan
    vy[rng.random(vy.shape) < fraction] = np.nan
    assert_same(asda.calculate_gamma_values(vx, vy), rs_gamma(vx, vy))


def test_nan_blocks_and_whole_nan_rows():
    vx, vy = random_field(12, (30, 30), np.float64)
    vx[10:14, 10:14] = np.nan
    vy[20, :] = np.nan
    vx[:, 3] = np.nan
    expected = asda.calculate_gamma_values(vx, vy, r=2)
    out = rs_gamma(vx, vy, r=2)
    assert_same(expected, out)
    assert np.isfinite(out).all()
    # gamma2 of a window that contains a NaN is exactly zero.
    assert np.all(out[11:13, 11:13, 1] == 0)


def test_nan_in_one_component_only_and_all_nan():
    vx, vy = random_field(13, (15, 15), np.float32)
    vy[:] = np.nan
    assert_same(asda.calculate_gamma_values(vx, vy), rs_gamma(vx, vy))
    vx[:] = np.nan
    assert_same(asda.calculate_gamma_values(vx, vy), rs_gamma(vx, vy))


def test_infinities():
    vx, vy = random_field(14, (20, 20), np.float64)
    vx[5, 5] = np.inf
    vy[12, 9] = -np.inf
    assert_same(asda.calculate_gamma_values(vx, vy), rs_gamma(vx, vy))


def test_zero_field_and_constant_field():
    z = np.zeros((14, 14))
    assert_same(asda.calculate_gamma_values(z, z), rs_gamma(z, z))
    c = np.full((14, 14), 2.5)
    assert_same(asda.calculate_gamma_values(c, c), rs_gamma(c, c))


# ---------------------------------------------------------------- layout, dtype and validation


def test_non_contiguous_inputs():
    vx, vy = random_field(15, (50, 60), np.float64)
    cases = [
        (vx[::2, ::3], vy[::2, ::3]),
        (vx.T, vy.T),
        (np.asfortranarray(vx), np.asfortranarray(vy)),
        (vx[5:40, 7:50], vy[5:40, 7:50]),
        (vx[::-1, ::-1], vy[::-1, ::-1]),
    ]
    for a, b in cases:
        assert_same(asda.calculate_gamma_values(a, b), rs_gamma(a, b))


def test_inputs_are_not_modified_and_readonly_is_fine():
    vx, vy = random_field(16, (20, 20), np.float64)
    vx.setflags(write=False)
    vy.setflags(write=False)
    keep = vx.copy(), vy.copy()
    rs_gamma(vx, vy)
    np.testing.assert_array_equal(vx, keep[0])
    np.testing.assert_array_equal(vy, keep[1])


def test_big_endian_float32_input_gives_native_output_like_the_original(real_field):
    vx, vy, _ = real_field
    cx, cy = vx[:40, :40], vy[:40, :40]
    assert cx.dtype.byteorder == ">"
    expected = asda.calculate_gamma_values(cx, cy)
    out = rs_gamma(cx, cy)
    assert expected.dtype == out.dtype == np.dtype(np.float32)
    assert_same(expected, out)


def test_mixed_float32_float64():
    vx, vy = random_field(17, (20, 20), np.float64)
    vx = vx.astype(np.float32)
    expected = asda.calculate_gamma_values(vx, vy)
    out = rs_gamma(vx, vy)
    assert expected.dtype == out.dtype == np.float64
    assert_same(expected, out)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_integer_inputs(dtype):
    # Gamma is bounded by 48/49 in magnitude for r = 3 because the centre point never contributes,
    # so truncating it into an integer output always gives 0 and the integer output is zero for
    # both versions by construction. The comparison with the original is therefore backed by a
    # second check that the float64 gammas of the same values are far from zero, which shows the
    # integer inputs really are processed and not just ignored.
    y, x = np.mgrid[0:40, 0:40].astype(np.float64)
    r2 = (x - 20) ** 2 + (y - 19) ** 2 + 1e-9
    swirl = (1 - np.exp(-r2 / 30)) / r2
    rng = np.random.default_rng(18)
    vx = np.rint(-(y - 19) * swirl * 100000 + rng.normal(size=r2.shape) * 50).astype(dtype)
    vy = np.rint((x - 20) * swirl * 100000 + rng.normal(size=r2.shape) * 50).astype(dtype)
    assert np.abs(vx).max() > 1000
    expected = asda.calculate_gamma_values(vx, vy)
    out = rs_gamma(vx, vy)
    assert expected.dtype == out.dtype == dtype
    assert_same(expected, out)
    as_float = asda.calculate_gamma_values(vx.astype(np.float64), vy.astype(np.float64))
    assert np.abs(as_float).max() > 0.5
    assert_same(np.trunc(as_float).astype(dtype), out)
    assert_same(as_float, rs_gamma(vx.astype(np.float64), vy.astype(np.float64)))


def test_output_is_new_and_writable():
    vx, vy = random_field(19, (12, 12), np.float64)
    out = rs_gamma(vx, vy)
    assert out.flags.writeable
    out[...] = 1.0
    assert not np.shares_memory(out, vx)


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"r": 2.0}, TypeError),
        ({"r": np.int64(2)}, TypeError),
        ({"r": "3"}, TypeError),
        ({"factor": 1.0}, TypeError),
        ({"factor": np.int32(2)}, TypeError),
        ({"factor": None}, TypeError),
    ],
)
def test_argument_validation_matches(kwargs, exc):
    vx, vy = random_field(20, (20, 20), np.float64)
    with pytest.raises(exc) as orig:
        asda.calculate_gamma_values(vx, vy, **kwargs)
    with pytest.raises(exc) as ours:
        rs_gamma(vx, vy, **kwargs)
    assert str(orig.value) == str(ours.value)


def test_shape_mismatch_matches():
    vx, _ = random_field(21, (20, 20), np.float64)
    _, vy = random_field(21, (20, 21), np.float64)
    with pytest.raises(ValueError, match="do not match") as orig:
        asda.calculate_gamma_values(vx, vy)
    with pytest.raises(ValueError, match="do not match") as ours:
        rs_gamma(vx, vy)
    assert str(orig.value) == str(ours.value)


def test_positional_arguments_and_defaults_match():
    vx, vy = random_field(22, (20, 20), np.float64)
    assert_same(asda.calculate_gamma_values(vx, vy, 1, 2), rs_gamma(vx, vy, 1, 2))


def test_rust_only_rejections():
    vx, vy = random_field(23, (20, 20), np.float64)
    with pytest.raises(ValueError, match="negative"):
        rs_gamma(vx, vy, r=-1)
    with pytest.raises(ValueError, match="2d"):
        rs_gamma(vx[0], vy[0])
    with pytest.raises(TypeError, match="Unsupported dtype"):
        rs_gamma(vx.astype(np.float16), vy.astype(np.float16))
    with pytest.raises(TypeError, match="Unsupported dtype"):
        rs_gamma(vx.astype(complex), vy.astype(complex))


# ---------------------------------------------------------------- downstream pipeline


def _assert_edges_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert len(a[key]) == len(b[key]), key
        for u, v in zip(a[key], b[key]):
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v))


def test_pipeline_on_real_field(real_field, real_gamma_original):
    vx, vy, image = real_field
    gamma = rs_gamma(vx, vy)
    expected = asda.get_vortex_edges(real_gamma_original)
    out = asda.get_vortex_edges(gamma)
    assert len(expected["center"]) > 0, "the real field is expected to contain vortices"
    _assert_edges_equal(expected, out)
    # Vortex count and centres, the quantities ASDA is used for.
    assert len(out["center"]) == len(expected["center"])
    np.testing.assert_array_equal(np.array(out["center"]), np.array(expected["center"]))
    props_expected = asda.get_vortex_properties(vx, vy, expected, image=image)
    props_out = asda.get_vortex_properties(vx, vy, out, image=image)
    for a, b in zip(props_expected, props_out):
        np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def test_pipeline_with_factor_two_on_a_crop(real_field):
    vx, vy, _ = real_field
    # This crop is one where the original get_vortex_edges runs (it fails on ragged contour lists for
    # many other crops, which is an upstream quirk that does not involve the gamma values) and finds a vortex.
    cx, cy = vx[300:364, 100:164].astype(np.float64), vy[300:364, 100:164].astype(np.float64)
    expected = asda.get_vortex_edges(asda.calculate_gamma_values(cx, cy, factor=2), factor=2)
    out = asda.get_vortex_edges(rs_gamma(cx, cy, factor=2), factor=2)
    assert len(out["center"]) == 1
    _assert_edges_equal(expected, out)


def test_pipeline_on_a_synthetic_vortex():
    # An analytic Lamb-Oseen style vortex plus a weak background, so that a vortex is certainly found.
    y, x = np.mgrid[0:64, 0:64].astype(np.float64)
    r2 = (x - 30) ** 2 + (y - 34) ** 2 + 1e-9
    swirl = (1 - np.exp(-r2 / 40)) / r2
    vx = -(y - 34) * swirl * 50
    vy = (x - 30) * swirl * 50
    expected = asda.get_vortex_edges(asda.calculate_gamma_values(vx, vy))
    out = asda.get_vortex_edges(rs_gamma(vx, vy))
    assert len(out["center"]) >= 1
    _assert_edges_equal(expected, out)

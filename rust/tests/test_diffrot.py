"""Direct comparison of heliox_rs.diffrot with sunpy.physics.differential_rotation (sunpy 8.0.1).

Unlike the ASDA and MEM kernels this one is not bit exact. The original goes through wcslib and
astropy representations and the Rust kernel works on unit vectors, so the source pixel coordinates
agree to rounding and the tests use tolerances. Where a tolerance is not at the level of double
precision, the reason is the original: its law of cosines for the distance to the solar surface
cancels two numbers of about 1e23 m^2 to get one of about 1e18 m^2, which leaves a rounding error
of a few millimetres in the distance (see `test_kernel_matches_exact_arithmetic`, which compares
both implementations with 50 digit arithmetic). That error is about 1e-9 pixel at the disc centre
and grows to about 1e-7 pixel within a pixel of the limb. The tests assert identical NaN masks on
the pixel grids they use. That is not a guarantee for every point: within about 1e-8 pixel of the
limb the finite and NaN boundaries of sunpy and the kernel differ by up to about 5e-9 pixel, again
because of the cancellation in sunpy.

The sunpy sample AIA 171 image is the real data. Point ``SUNPY_SAMPLEDIR`` at a directory that
holds it (sunpy downloads it there when it is missing, which needs network access). Without it the
tests are skipped.
"""

import inspect
import threading
import types
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

pytest.importorskip("skimage")
sunpy_map = pytest.importorskip("sunpy.map")
dr = pytest.importorskip("sunpy.physics.differential_rotation")
from sunpy.coordinates import HeliographicStonyhurst, get_earth  # noqa: E402
from sunpy.sun.models import differential_rotation  # noqa: E402

from heliox_rs import _core  # noqa: E402
from heliox_rs import diffrot as rs  # noqa: E402

# Tolerances, each two to three times the largest value measured over the cases below (the 99th
# percentile reaches 5.6e-8 for the 14 day rotation, which leaves only a few thousand pixels, all
# near the limb of the map observer, and is under 2e-8 for the others; the largest difference is
# 7.6e-7). The causes are explained in the module documentation.
COORD_P99 = 1e-7  # pixels, 99th percentile of the source coordinate difference
COORD_MAX = 2e-6  # pixels, largest difference, reached within a pixel of the limb
FLOAT64_RTOL = 4 * np.finfo(np.float64).eps  # image, float64 data, on top of the gradient term
FLOAT32_RTOL = 4 * np.finfo(np.float32).eps  # image, float32 data (a few units in the last place)


@pytest.fixture(scope="session")
def aia():
    """The real 1024 x 1024 AIA 171 sample image."""
    try:
        import sunpy.data.sample as sample

        path = sample.AIA_171_IMAGE
    except Exception as exc:  # no data and no network
        pytest.skip(f"sunpy sample data is not available: {exc}")
    return sunpy_map.Map(path)


@pytest.fixture
def rust_calls(monkeypatch):
    """The arguments of every call that reaches the Rust kernel."""
    calls = []
    real = _core.warp_sun_coordinates_f64

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(_core, "warp_sun_coordinates_f64", counting)
    return calls


def crop(smap, lo, hi):
    return smap.submap(lo * u.pix, top_right=hi * u.pix)


def grid(n):
    """The pixel grid `skimage.transform.warp` passes to the inverse map (x first)."""
    return np.indices((n, n), dtype=float).reshape(2, -1).T


def stereo_like(smap):
    return SkyCoord(60 * u.deg, 4 * u.deg, 0.97 * u.AU, frame=HeliographicStonyhurst, obstime=smap.date + 2 * u.day)


def new_observer(smap, kwargs):
    return dr._get_new_observer(smap.date, kwargs.get("observer"), kwargs.get("time"))


# name -> (builder of the map, rotation arguments, model arguments)
SCENARIOS = {
    "time +1h": (lambda m: m, lambda m: dict(time=1 * u.hr), {}),
    "time +6h": (lambda m: m, lambda m: dict(time=6 * u.hr), {}),
    "time -2d": (lambda m: m, lambda m: dict(time=-2 * u.day), {}),
    "time +14d": (lambda m: m, lambda m: dict(time=14 * u.day), {}),
    "earth +3d": (lambda m: m, lambda m: dict(observer=get_earth(m.date + 3 * u.day)), {}),
    "stereo like": (lambda m: m, lambda m: dict(observer=stereo_like(m)), {}),
    "snodgrass": (lambda m: m, lambda m: dict(time=1 * u.day), dict(model="snodgrass")),
    "allen synodic": (lambda m: m, lambda m: dict(time=1 * u.day), dict(model="allen", frame_time="synodic")),
    "rigid": (lambda m: m, lambda m: dict(time=1 * u.day), dict(model="rigid")),
    "howard synodic": (lambda m: m, lambda m: dict(time=-3 * u.day), dict(model="howard", frame_time="synodic")),
}

IMAGE_SCENARIOS = {
    **SCENARIOS,
    "crop centre 512, +6h": (lambda m: crop(m, [256, 256], [767, 767]), lambda m: dict(time=6 * u.hr), {}),
    "crop corner 512, +1d": (lambda m: crop(m, [0, 0], [511, 511]), lambda m: dict(time=1 * u.day), {}),
    "crop limb 512, -1d": (lambda m: crop(m, [512, 300], [1023, 811]), lambda m: dict(time=-1 * u.day), {}),
    "crop off limb 512, earth": (
        lambda m: crop(m, [600, 0], [1023, 511]),
        lambda m: dict(observer=get_earth(m.date - 2 * u.day)),
        {},
    ),
    "rotated PC matrix": (
        lambda m: sunpy_map.Map(m.data, {**m.meta, "crota2": 25.0}),
        lambda m: dict(time=1 * u.day),
        {},
    ),
    "rotated PC matrix, crop": (
        lambda m: crop(sunpy_map.Map(m.data, {**m.meta, "crota2": -40.0}), [100, 200], [611, 711]),
        lambda m: dict(time=-6 * u.hr),
        dict(model="snodgrass"),
    ),
}


# ------------------------------------------------------------------ the coordinates


def compare_coordinates(expected, actual):
    """NaN masks identical on this grid (not guaranteed for points within about 1e-8 pixel of the
    limb), and the coordinate differences within the documented bounds."""
    assert expected.shape == actual.shape and actual.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    ok = ~np.isnan(expected[:, 0])
    assert ok.any()
    diff = np.abs(actual[ok] - expected[ok]).max(axis=1)
    assert np.percentile(diff, 99) < COORD_P99, np.percentile(diff, 99)
    assert diff.max() < COORD_MAX, diff.max()
    return diff


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_inverse_map_matches_sunpy_on_the_full_map(aia, rust_calls, name):
    _, rotation, model = SCENARIOS[name]
    obs = new_observer(aia, rotation(aia))
    xy = grid(1024)
    expected = dr._warp_sun_coordinates(xy, aia, obs, **model)
    actual = rs.warp_sun_coordinates(xy, aia, obs, **model)
    assert len(rust_calls) == 1
    compare_coordinates(expected, actual)


@pytest.mark.parametrize("name", ["crop centre 512, +6h", "crop corner 512, +1d", "rotated PC matrix", "rotated PC matrix, crop"])
def test_inverse_map_matches_sunpy_on_crops_and_rotated_headers(aia, rust_calls, name):
    build, rotation, model = IMAGE_SCENARIOS[name]
    smap = build(aia)
    obs = new_observer(smap, rotation(smap))
    rows, cols = smap.data.shape
    xy = np.indices((cols, rows), dtype=float).reshape(2, -1).T
    expected = dr._warp_sun_coordinates(xy, smap, obs, **model)
    actual = rs.warp_sun_coordinates(xy, smap, obs, **model)
    assert len(rust_calls) == 1
    compare_coordinates(expected, actual)


def test_inverse_map_on_random_and_off_grid_points(aia, rust_calls):
    """Points that are not on the pixel grid, outside the image and far from the disc."""
    rng = np.random.default_rng(5)
    xy = np.vstack([rng.uniform(-300, 1300, (20000, 2)), [[512.0, 512.0], [0.0, 0.0], [1023.0, 1023.0]]])
    obs = new_observer(aia, dict(time=-1 * u.day))
    expected = dr._warp_sun_coordinates(xy, aia, obs)
    actual = rs.warp_sun_coordinates(xy, aia, obs)
    assert len(rust_calls) == 1
    compare_coordinates(expected, actual)


def test_nan_masks_for_all_three_kinds_of_pixel(aia, rust_calls):
    """Off disk for the new observer, behind the limb for the map observer, and mapped."""
    obs = stereo_like(aia)
    xy = grid(1024)
    expected = dr._warp_sun_coordinates(xy, aia, obs)
    actual = rs.warp_sun_coordinates(xy, aia, obs)
    assert len(rust_calls) == 1
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    # A rotation of 60 degrees in longitude puts a large part of the disc behind the limb, so
    # the NaN pixels are not just the off-disc ones.
    on_disc_new_observer = np.isfinite(dr._warp_sun_coordinates(xy, aia, get_earth(aia.date)))
    assert 0 < np.isfinite(expected[:, 0]).sum() < on_disc_new_observer[:, 0].sum()


def test_float32_and_non_contiguous_input(aia, rust_calls):
    xy = grid(1024)[::7]
    obs = new_observer(aia, dict(time=6 * u.hr))
    expected = dr._warp_sun_coordinates(xy, aia, obs)
    for variant in (xy.astype(np.float32), np.asfortranarray(xy), xy[:, ::-1][:, ::-1]):
        actual = rs.warp_sun_coordinates(variant, aia, obs)
        expected_variant = dr._warp_sun_coordinates(variant, aia, obs)
        compare_coordinates(expected_variant, actual)
    assert len(rust_calls) == 3
    assert expected.shape == (len(xy), 2)


def test_nan_and_infinite_input_gives_nan_rows(aia, rust_calls):
    xy = grid(1024)[:5000].copy()
    xy[10] = np.nan
    xy[20, 0] = np.inf
    obs = new_observer(aia, dict(time=6 * u.hr))
    actual = rs.warp_sun_coordinates(xy, aia, obs)
    assert len(rust_calls) == 1
    assert np.isnan(actual[10]).all() and np.isnan(actual[20]).all()


def test_parallel_and_serial_give_identical_bits(aia):
    obs = new_observer(aia, dict(time=-1 * u.day))
    wcs, chain = rs._parameters(aia, obs, {})
    xy = grid(1024)
    parallel = _core.warp_sun_coordinates_f64(xy, wcs, chain, True)
    serial = _core.warp_sun_coordinates_f64(xy, wcs, chain, False)
    np.testing.assert_array_equal(parallel, serial)


def test_kernel_rejects_malformed_parameters(aia):
    obs = new_observer(aia, dict(time=1 * u.hr))
    wcs, chain = rs._parameters(aia, obs, {})
    xy = grid(8)
    with pytest.raises(ValueError):
        _core.warp_sun_coordinates_f64(xy, wcs[:-1], chain)
    with pytest.raises(ValueError):
        _core.warp_sun_coordinates_f64(xy, wcs, chain[:-1])
    with pytest.raises(ValueError):
        _core.warp_sun_coordinates_f64(np.zeros((4, 3)), wcs, chain)
    with pytest.raises(ValueError):
        _core.warp_sun_coordinates_f64(xy, [0.0] * 10, chain)


def test_kernel_matches_independent_exact_arithmetic(aia):
    """Both implementations against 50 digit arithmetic written independently of the kernel.

    The reference uses angles and inverse trigonometric functions throughout: the TAN projection
    through the native longitude and latitude (``atan2`` and ``asin``), the law of cosines in its
    original form, heliographic longitude and latitude from ``atan2`` and ``asin``, and the
    rotation by subtracting the angle from the longitude. Only the numbers that Python hands the
    kernel are shared (WCS parameters, observer distances, rate coefficients and the two matrices).
    The comparison with sunpy checks those. Disc pixels and a band of pixels just inside the limb
    are used, for two rotations.
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    rng = np.random.default_rng(3)
    for kwargs in (dict(time=-2 * u.day), dict(time=6 * u.hr)):
        obs = new_observer(aia, kwargs)
        wcs, chain = rs._parameters(aia, obs, {})
        limb_px = np.degrees(np.arcsin(chain[1] / chain[0])) / np.hypot(wcs[0], wcs[2])
        centre = np.array(wcs[4:6])
        angle = rng.uniform(0, 2 * np.pi, 100)
        radius = limb_px * rng.uniform(0.97, 0.999, 100)
        band = centre + np.c_[radius * np.cos(angle), radius * np.sin(angle)]
        xy = np.vstack([rng.uniform(100, 900, (120, 2)), band])
        expected = dr._warp_sun_coordinates(xy, aia, obs)
        actual = rs.warp_sun_coordinates(xy, aia, obs)

        cd = [mp.mpf(v) for v in wcs[:4]]
        crpix = [mp.mpf(v) for v in wcs[4:6]]
        a0, d0 = mp.mpf(wcs[6]), mp.mpf(wcs[7])
        phip = mp.atan2(mp.mpf(wcs[8]), mp.mpf(wcs[9]))
        r_new, rsun, c0, c1, c2, r_map = (mp.mpf(v) for v in chain[:6])
        mat_a = mp.matrix(3, 3)
        mat_b = mp.matrix(3, 3)
        for i in range(3):
            for j in range(3):
                mat_a[i, j] = mp.mpf(chain[6 + 3 * i + j])
                mat_b[i, j] = mp.mpf(chain[15 + 3 * i + j])
        det = cd[0] * cd[3] - cd[1] * cd[2]
        inv = [cd[3] / det, -cd[1] / det, -cd[2] / det, cd[0] / det]
        deg = mp.pi / 180

        def exact(px, py):
            # Pixel to native coordinates, then to longitude and latitude (Calabretta and Greisen).
            dx, dy = mp.mpf(px) - crpix[0], mp.mpf(py) - crpix[1]
            x, y = cd[0] * dx + cd[1] * dy, cd[2] * dx + cd[3] * dy
            phi = mp.atan2(x, -y)
            theta = mp.atan2(1 / deg, mp.hypot(x, y))
            lat = mp.asin(mp.sin(theta) * mp.sin(d0) + mp.cos(theta) * mp.cos(d0) * mp.cos(phi - phip))
            lon = a0 + mp.atan2(
                -mp.cos(theta) * mp.sin(phi - phip),
                mp.sin(theta) * mp.cos(d0) - mp.cos(theta) * mp.sin(d0) * mp.cos(phi - phip),
            )
            # Law of cosines in its original form, then heliocentric and heliographic.
            cos_alpha = mp.cos(lat) * mp.cos(lon)
            b = -2 * r_new * cos_alpha
            disc = b * b - 4 * (r_new**2 - rsun**2)
            if disc < 0:
                return None
            dist = (-b - mp.sqrt(disc)) / 2
            hpc = [dist * cos_alpha, dist * mp.cos(lat) * mp.sin(lon), dist * mp.sin(lat)]
            h = mat_a * mp.matrix([hpc[1], hpc[2], r_new - hpc[0]])
            hlon = mp.atan2(h[1], h[0])
            hlat = mp.asin(h[2] / mp.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2))
            rot = c0 + c1 * mp.sin(hlat) ** 2 + c2 * mp.sin(hlat) ** 4
            rl = hlon - rot
            rc = mp.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            m = mat_b * mp.matrix([rc * mp.cos(hlat) * mp.cos(rl), rc * mp.cos(hlat) * mp.sin(rl), rc * mp.sin(hlat)])
            if m[2] < 0:
                return None
            # Back to helioprojective angles as seen by the map observer, then the inverse WCS.
            v = [r_map - m[2], m[0], m[1]]
            tx = mp.atan2(v[1], v[0])
            ty = mp.asin(v[2] / mp.sqrt(sum(t**2 for t in v)))
            sin_t = mp.sin(ty) * mp.sin(d0) + mp.cos(ty) * mp.cos(d0) * mp.cos(tx - a0)
            theta2 = mp.asin(sin_t)
            phi2 = phip + mp.atan2(
                -mp.cos(ty) * mp.sin(tx - a0),
                mp.sin(ty) * mp.cos(d0) - mp.cos(ty) * mp.sin(d0) * mp.cos(tx - a0),
            )
            radius_native = mp.cot(theta2) / deg
            px_w, py_w = radius_native * mp.sin(phi2), -radius_native * mp.cos(phi2)
            return crpix[0] + inv[0] * px_w + inv[1] * py_w, crpix[1] + inv[2] * px_w + inv[3] * py_w

        rust_error = []
        sunpy_error = []
        for i in range(len(xy)):
            truth = exact(*xy[i])
            if truth is None or np.isnan(expected[i, 0]) or np.isnan(actual[i, 0]):
                continue
            rust_error.append(max(abs(float(truth[k] - actual[i, k])) for k in (0, 1)))
            sunpy_error.append(max(abs(float(truth[k] - expected[i, k])) for k in (0, 1)))
        assert len(rust_error) > 100, len(rust_error)
        assert max(rust_error) < 1e-10, max(rust_error)
        # sunpy's own error is larger than the Rust error.
        assert np.median(sunpy_error) > 5 * np.median(rust_error)


def test_rate_coefficients_reproduce_sunpy_for_every_model():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-90, 90, 500) * u.deg
    for model in ("howard", "snodgrass", "allen", "rigid"):
        for frame_time in ("sidereal", "synodic"):
            for seconds in (3600.0, -172800.0, 1209600.0):
                c = rs._rate_coefficients(seconds, dict(model=model, frame_time=frame_time))
                s2 = np.sin(lat.to_value(u.rad)) ** 2
                mine = np.rad2deg(c[0] + c[1] * s2 + c[2] * s2**2)
                original = differential_rotation(seconds * u.s, lat, model=model, frame_time=frame_time).deg
                # The original wraps the angle to [0, 360) degrees.
                np.testing.assert_allclose(np.mod(mine, 360.0), original, rtol=0, atol=1e-11)


def test_unknown_model_and_frame_time():
    with pytest.raises(ValueError, match="model must equal one of"):
        rs._rate_coefficients(3600.0, dict(model="nope"))


# ------------------------------------------------------------------ the whole function


def local_gradient(a):
    """Largest absolute difference between a pixel and its four neighbours, ignoring NaN."""
    g = np.zeros(a.shape)
    for axis in (0, 1):
        d = np.nan_to_num(np.abs(np.diff(a, axis=axis)), nan=0.0)
        lead = [slice(None)] * 2
        trail = [slice(None)] * 2
        lead[axis] = slice(None, -1)
        trail[axis] = slice(1, None)
        g[tuple(lead)] = np.maximum(g[tuple(lead)], d)
        g[tuple(trail)] = np.maximum(g[tuple(trail)], d)
    return g


def compare_maps(expected, actual, rtol):
    """Same type, metadata and NaN mask. The data differ by what a shift of the sampling point by
    the coordinate tolerance does (the local gradient times COORD_MAX pixels) plus a few units in
    the last place."""
    assert type(actual) is type(expected)
    assert actual.data.dtype == expected.data.dtype
    assert actual.data.shape == expected.data.shape
    assert dict(actual.meta) == dict(expected.meta)
    assert actual.date == expected.date
    assert actual.reference_date == expected.reference_date
    assert actual.observer_coordinate.obstime == expected.observer_coordinate.obstime
    a, b = np.asarray(expected.data, dtype=float), np.asarray(actual.data, dtype=float)
    np.testing.assert_array_equal(np.isnan(b), np.isnan(a))
    ok = ~np.isnan(a)
    assert ok.any()
    allowed = rtol * np.maximum(np.abs(a), 1.0) + COORD_MAX * local_gradient(a)
    excess = (np.abs(b - a) - allowed)[ok]
    assert excess.max() <= 0, f"largest excess {excess.max():.3e}"
    return np.abs(b[ok] - a[ok]).max()


@pytest.mark.parametrize("name", list(IMAGE_SCENARIOS))
def test_differential_rotate_matches_sunpy(aia, rust_calls, name):
    build, rotation, model = IMAGE_SCENARIOS[name]
    smap = build(aia)
    kwargs = rotation(smap)
    expected = dr.differential_rotate(smap, **kwargs, **model)
    actual = rs.differential_rotate(smap, **kwargs, **model)
    assert len(rust_calls) == 1
    compare_maps(expected, actual, FLOAT32_RTOL)


@pytest.mark.parametrize("kwargs", [dict(time=1 * u.hr), dict(time=-2 * u.day)])
def test_differential_rotate_matches_sunpy_for_float64_data(aia, rust_calls, kwargs):
    smap = sunpy_map.Map(aia.data.astype(np.float64), aia.meta)
    expected = dr.differential_rotate(smap, **kwargs)
    actual = rs.differential_rotate(smap, **kwargs)
    assert len(rust_calls) == 1
    assert actual.data.dtype == np.float64
    compare_maps(expected, actual, FLOAT64_RTOL)


@pytest.mark.parametrize("name", ["time +1h", "time -2d", "earth +3d", "stereo like"])
def test_float64_image_difference_is_explained_by_the_coordinate_difference(aia, name):
    """Bilinear interpolation is linear in the sampling point, so moving a sample by `d` pixels
    changes it by at most `d` times the largest difference between neighbouring source pixels in
    the cells around it. This bounds the image difference pixel by pixel with the actual coordinate
    difference, and the ratio of the two is reported as the slack."""
    from scipy.ndimage import maximum_filter

    smap = sunpy_map.Map(aia.data.astype(np.float64), aia.meta)
    _, rotation, model = SCENARIOS[name]
    kwargs = rotation(smap)
    obs = new_observer(smap, kwargs)
    rows, cols = smap.data.shape
    xy = np.indices((cols, rows), dtype=float).reshape(2, -1).T
    original = dr._warp_sun_coordinates(xy, smap, obs, **model)
    rust = rs.warp_sun_coordinates(xy, smap, obs, **model)
    expected = dr.differential_rotate(smap, **kwargs, **model).data
    actual = rs.differential_rotate(smap, **kwargs, **model).data
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))

    data = smap.data
    slope_x = maximum_filter(np.abs(np.diff(data, axis=1)), size=3, mode="nearest")
    slope_y = maximum_filter(np.abs(np.diff(data, axis=0)), size=3, mode="nearest")
    # Per output pixel, as arrays indexed [row, column].
    as_image = lambda column: column.reshape(cols, rows).T  # noqa: E731
    sx, sy = as_image(original[:, 0]), as_image(original[:, 1])
    dx = np.abs(as_image(rust[:, 0] - original[:, 0]))
    dy = np.abs(as_image(rust[:, 1] - original[:, 1]))
    ok = np.isfinite(sx) & np.isfinite(sy)
    ix = np.clip(np.floor(sx[ok]).astype(int), 0, cols - 2)
    iy = np.clip(np.floor(sy[ok]).astype(int), 0, rows - 2)
    bound = dx[ok] * slope_x[np.minimum(iy, rows - 1), ix] + dy[ok] * slope_y[iy, ix]
    difference = np.abs(actual - expected)[ok]
    rounding = FLOAT64_RTOL * np.maximum(np.abs(expected[ok]), 1.0)
    assert np.all(difference <= bound + rounding), (difference - bound - rounding).max()
    # The bound is not vacuous: where the coordinates differ by more than 1e-9 pixel the image does
    # change, so the bound is used.
    assert difference.max() > 0


def test_masked_map_matches_sunpy(aia, rust_calls):
    mask = np.zeros(aia.data.shape, bool)
    mask[100:300, 400:500] = True
    smap = sunpy_map.Map(np.ma.array(aia.data, mask=mask), aia.meta)
    expected = dr.differential_rotate(smap, time=1 * u.day)
    actual = rs.differential_rotate(smap, time=1 * u.day)
    assert len(rust_calls) == 1
    assert (actual.mask is None) == (expected.mask is None)
    if expected.mask is not None:
        np.testing.assert_array_equal(actual.mask, expected.mask)
    compare_maps(expected, actual, FLOAT32_RTOL)


def test_signature_and_documentation_match_sunpy():
    assert inspect.signature(rs.differential_rotate) == inspect.signature(dr.differential_rotate)
    assert rs.differential_rotate.__name__ == "differential_rotate"
    assert "inverse map" in rs.differential_rotate.__doc__


def test_warnings_and_errors_match_sunpy(aia):
    def run(function, **kwargs):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                function(aia, **kwargs)
                error = None
            except Exception as exc:
                error = (type(exc), str(exc))
        return [(w.category, str(w.message)) for w in record], error

    for kwargs in (
        dict(time=1 * u.hr),
        dict(time=1 * u.hr, observer=get_earth(aia.date)),
        dict(),
        dict(time=1 * u.hr, model="nope"),
        dict(time=1 * u.hr, frame_time="synodic"),
        dict(time=1 * u.hr, unexpected=1),
    ):
        expected = run(dr.differential_rotate, **kwargs)
        actual = run(rs.differential_rotate, **kwargs)
        assert actual == expected, kwargs
    assert run(rs.differential_rotate, time=1 * u.hr)[0], "the Earth observer warning has to be kept"


def test_entirely_off_disk_map_raises_like_sunpy(aia):
    off = crop(aia, [0, 0], [20, 20])
    with pytest.raises(ValueError) as expected:
        dr.differential_rotate(off, time=1 * u.hr)
    with pytest.raises(ValueError) as actual:
        rs.differential_rotate(off, time=1 * u.hr)
    assert str(actual.value) == str(expected.value)


# ------------------------------------------------------------------ fallbacks


def assert_same_map(expected, actual):
    assert dict(actual.meta) == dict(expected.meta)
    assert actual.data.dtype == expected.data.dtype
    np.testing.assert_array_equal(actual.data, expected.data)


def test_other_projection_runs_sunpys_own_code(aia, rust_calls):
    meta = {**aia.meta, "ctype1": "HPLN-ARC", "ctype2": "HPLT-ARC"}
    smap = sunpy_map.Map(aia.data, meta)
    assert rs._wcs_parameters(smap) is None
    expected = dr.differential_rotate(smap, time=1 * u.hr)
    actual = rs.differential_rotate(smap, time=1 * u.hr)
    assert rust_calls == []
    assert_same_map(expected, actual)


def test_extra_keyword_arguments_run_sunpys_own_code(aia, rust_calls):
    for kwargs in (dict(unexpected=1), dict(model="nope")):
        with pytest.raises(Exception) as expected:
            dr.differential_rotate(aia, time=1 * u.hr, **kwargs)
        with pytest.raises(Exception) as actual:
            rs.differential_rotate(aia, time=1 * u.hr, **kwargs)
        assert type(actual.value) is type(expected.value) and str(actual.value) == str(expected.value)
    assert rust_calls == []


def test_distorted_wcs_and_other_wcs_types_are_not_handled(aia):
    from astropy.wcs import Sip

    wcs = aia.wcs.deepcopy()
    assert rs._wcs_parameters(types.SimpleNamespace(wcs=wcs)) is not None
    wcs.sip = Sip(np.zeros((3, 3)), np.zeros((3, 3)), None, None, wcs.wcs.crpix)
    assert rs._wcs_parameters(types.SimpleNamespace(wcs=wcs)) is None

    wcs = aia.wcs.deepcopy()
    wcs.wcs.set_pv([(2, 1, 0.5)])
    assert rs._wcs_parameters(types.SimpleNamespace(wcs=wcs)) is None

    wcs = aia.wcs.deepcopy()
    wcs.wcs.ctype = ["HGLN-CAR", "HGLT-CAR"]
    assert rs._wcs_parameters(types.SimpleNamespace(wcs=wcs)) is None


def test_unreproducible_xy_runs_sunpys_own_code(aia, rust_calls):
    obs = new_observer(aia, dict(time=1 * u.hr))
    xy = grid(1024)[:100]
    for odd in (xy.astype(int), xy.T, xy[:, :1]):
        try:
            expected = dr._warp_sun_coordinates(odd, aia, obs)
        except Exception as exc:
            with pytest.raises(type(exc)):
                rs.warp_sun_coordinates(odd, aia, obs)
        else:
            np.testing.assert_array_equal(rs.warp_sun_coordinates(odd, aia, obs), expected)
    assert rust_calls == []


def test_failing_setup_falls_back_to_the_original(aia, rust_calls, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no matrices")

    monkeypatch.setattr(rs, "_observer_matrices", boom)
    obs = new_observer(aia, dict(time=1 * u.hr))
    xy = grid(1024)[::11]
    np.testing.assert_array_equal(rs.warp_sun_coordinates(xy, aia, obs), dr._warp_sun_coordinates(xy, aia, obs))
    assert rust_calls == []


def test_frame_mismatch_falls_back(aia, rust_calls, monkeypatch):
    monkeypatch.setattr(rs, "_frames_agree", lambda smap: False)
    obs = new_observer(aia, dict(time=1 * u.hr))
    xy = grid(1024)[::11]
    np.testing.assert_array_equal(rs.warp_sun_coordinates(xy, aia, obs), dr._warp_sun_coordinates(xy, aia, obs))
    assert rust_calls == []


# ------------------------------------------------------------------ the swap


def test_inverse_map_is_looked_up_as_a_module_global(aia, rust_calls, monkeypatch):
    """sunpy has to see the replacement during the call and the original afterwards."""
    before = dr._warp_sun_coordinates
    assert not getattr(before, "_heliox_rs", False)
    seen = []
    real = dr.differential_rotate

    def spy(*args, **kwargs):
        seen.append(dr._warp_sun_coordinates)
        return real(*args, **kwargs)

    monkeypatch.setattr(dr, "differential_rotate", spy)
    rs.differential_rotate(aia, time=1 * u.hr)
    assert seen == [rs.warp_sun_coordinates]
    assert len(rust_calls) == 1
    assert dr._warp_sun_coordinates is before
    assert rs._original() is before and rs._DEPTH == 0


def test_original_is_restored_after_an_exception():
    before = dr._warp_sun_coordinates
    real = dr.differential_rotate

    def boom(*args, **kwargs):
        raise ZeroDivisionError("boom")

    dr.differential_rotate = boom
    try:
        with pytest.raises(ZeroDivisionError):
            rs.differential_rotate(None)
    finally:
        dr.differential_rotate = real
    assert dr._warp_sun_coordinates is before and rs._DEPTH == 0


def test_overlapping_threads_leave_the_original_in_place():
    before = dr._warp_sun_coordinates
    real = dr.differential_rotate
    inside = []
    errors = []

    for round_number in range(20):
        barrier = threading.Barrier(2, timeout=10)

        def spy(*args, **kwargs):
            barrier.wait()  # both threads are inside differential_rotate at the same time
            inside.append(dr._warp_sun_coordinates is rs.warp_sun_coordinates)
            barrier.wait()

        def worker(delay):
            try:
                threading.Event().wait(delay)
                rs.differential_rotate(None)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        dr.differential_rotate = spy
        try:
            threads = [threading.Thread(target=worker, args=(d,)) for d in (0.0, 0.002 * (round_number % 3))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            dr.differential_rotate = real
        assert dr._warp_sun_coordinates is before
    assert not errors
    assert inside and all(inside)
    assert rs._DEPTH == 0


def test_threads_running_real_rotations_agree(aia):
    """Two threads rotating real maps at the same time give the single thread answers."""
    kwargs = [dict(time=1 * u.hr), dict(time=-1 * u.day)]
    smap = crop(aia, [256, 256], [767, 767])
    expected = [dr.differential_rotate(smap, **k) for k in kwargs]
    results = [None, None]
    errors = []

    def worker(i):
        try:
            results[i] = rs.differential_rotate(smap, **kwargs[i])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for e, a in zip(expected, results):
        compare_maps(e, a, FLOAT32_RTOL)
    assert rs._DEPTH == 0


def test_rust_function_is_not_mistaken_for_the_original():
    assert rs.warp_sun_coordinates._heliox_rs is True
    assert not hasattr(rs._original(), "_heliox_rs")


# ------------------------------------------------------------------ coordinate contexts


def _contexts():
    from sunpy.coordinates import propagate_with_solar_surface, transform_with_sun_center
    from sunpy.coordinates.screens import PlanarScreen, SphericalScreen

    return {
        "spherical screen": lambda m: SphericalScreen(m.observer_coordinate),
        "spherical screen, only off disk": lambda m: SphericalScreen(m.observer_coordinate, only_off_disk=True),
        "planar screen": lambda m: PlanarScreen(m.observer_coordinate),
        "propagate with solar surface": lambda m: propagate_with_solar_surface(),
        "transform with sun center": lambda m: transform_with_sun_center(),
    }


@pytest.mark.parametrize("name", list(_contexts()))
def test_sunpy_coordinate_contexts_run_the_original(aia, rust_calls, name):
    """The kernel does not model these contexts, so the result must be sunpy's own, bit for bit."""
    obs = new_observer(aia, dict(time=1 * u.day))
    xy = grid(1024)[::3]
    make = _contexts()[name]
    with make(aia):
        expected = dr._warp_sun_coordinates(xy, aia, obs)
        actual = rs.warp_sun_coordinates(xy, aia, obs)
    np.testing.assert_array_equal(actual, expected)
    assert rust_calls == []
    # Outside the context the kernel is used again.
    rs.warp_sun_coordinates(xy, aia, obs)
    assert len(rust_calls) == 1


@pytest.mark.parametrize("name", ["spherical screen", "propagate with solar surface"])
def test_whole_rotation_under_a_context_matches_sunpy(aia, rust_calls, name):
    smap = crop(aia, [256, 256], [767, 767])
    make = _contexts()[name]
    with make(smap):
        expected = dr.differential_rotate(smap, time=6 * u.hr)
        actual = rs.differential_rotate(smap, time=6 * u.hr)
    assert rust_calls == []
    assert_same_map(expected, actual)


# ------------------------------------------------------------------ swap details


def test_a_users_patch_survives_the_swap(aia, rust_calls, monkeypatch):
    """The last exit restores what was there on the first entry, and the fallbacks call it."""
    seen = []
    real = dr._warp_sun_coordinates

    def user_patch(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(dr, "_warp_sun_coordinates", user_patch)
    rs.differential_rotate(aia, time=1 * u.hr)  # supported: Rust runs, the patch is not called
    assert dr._warp_sun_coordinates is user_patch and rs._DEPTH == 0
    assert len(rust_calls) == 1 and seen == []
    meta = {**aia.meta, "ctype1": "HPLN-ARC", "ctype2": "HPLT-ARC"}
    rs.differential_rotate(sunpy_map.Map(aia.data, meta), time=1 * u.hr)  # fallback calls the patch
    assert seen == [1]
    assert dr._warp_sun_coordinates is user_patch


def test_a_rust_value_error_falls_back_to_sunpy(aia, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("refused")

    monkeypatch.setattr(_core, "warp_sun_coordinates_f64", refuse)
    obs = new_observer(aia, dict(time=1 * u.hr))
    xy = grid(1024)[::11]
    np.testing.assert_array_equal(rs.warp_sun_coordinates(xy, aia, obs), dr._warp_sun_coordinates(xy, aia, obs))

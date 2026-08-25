import numpy as np
import pytest

import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import celestial_frame_to_wcs, wcs_to_celestial_frame

import heliox.coordinates.wcs_utils  # noqa: F401  (registers the mappings)
from heliox.coordinates import (
    Heliocentric,
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
)
from heliox.coordinates.wcs_utils import _register
from heliox.sun import constants

OBSTIME = "2013-10-28T12:00:00"


def make_wcs(ctypes, *, observer=True):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctypes
    wcs.wcs.dateobs = OBSTIME
    if observer:
        earth = get_earth(OBSTIME)
        wcs.wcs.aux.hgln_obs = earth.lon.to_value(u.deg)
        wcs.wcs.aux.hglt_obs = earth.lat.to_value(u.deg)
        wcs.wcs.aux.dsun_obs = earth.radius.to_value(u.m)
    return wcs


# ---------------------------------------------------------------------------
# WCS to frame
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ctypes, expected",
    [
        (["HPLN-TAN", "HPLT-TAN"], Helioprojective),
        (["HGLN-CAR", "HGLT-CAR"], HeliographicStonyhurst),
        (["CRLN-CAR", "CRLT-CAR"], HeliographicCarrington),
        (["SOLX", "SOLY"], Heliocentric),
    ],
)
def test_ctypes_map_to_frames(ctypes, expected):
    frame = wcs_to_celestial_frame(make_wcs(ctypes))
    assert isinstance(frame, expected)


def test_obstime_is_recovered():
    frame = wcs_to_celestial_frame(make_wcs(["HPLN-TAN", "HPLT-TAN"]))
    assert frame.obstime.isot == "2013-10-28T12:00:00.000"


def test_observer_is_recovered():
    frame = wcs_to_celestial_frame(make_wcs(["HPLN-TAN", "HPLT-TAN"]))
    earth = get_earth(OBSTIME)
    assert frame.observer.radius.to_value(u.m) == pytest.approx(earth.radius.to_value(u.m))


def test_missing_observer_gives_none():
    frame = wcs_to_celestial_frame(make_wcs(["HPLN-TAN", "HPLT-TAN"], observer=False))
    assert frame.observer is None


def test_observer_from_carrington_keywords():
    wcs = make_wcs(["HPLN-TAN", "HPLT-TAN"], observer=False)
    earth = get_earth(OBSTIME)
    carrington = earth.transform_to(HeliographicCarrington(obstime=OBSTIME, observer=earth))
    wcs.wcs.aux.crln_obs = carrington.lon.to_value(u.deg)
    wcs.wcs.aux.hglt_obs = carrington.lat.to_value(u.deg)
    wcs.wcs.aux.dsun_obs = earth.radius.to_value(u.m)

    frame = wcs_to_celestial_frame(wcs)
    assert frame.observer.lon.to_value(u.deg) == pytest.approx(0, abs=0.1)


def test_rsun_is_recovered():
    wcs = make_wcs(["HPLN-TAN", "HPLT-TAN"])
    wcs.wcs.aux.rsun_ref = 7.0e8
    assert wcs_to_celestial_frame(wcs).rsun.to_value(u.m) == pytest.approx(7.0e8)


def test_rsun_defaults_to_the_constant():
    frame = wcs_to_celestial_frame(make_wcs(["HPLN-TAN", "HPLT-TAN"]))
    assert frame.rsun.to_value(u.m) == pytest.approx(constants.radius.to_value(u.m))


def test_obstime_from_mjdobs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.mjdobs = 56593.5
    assert wcs_to_celestial_frame(wcs).obstime.isot.startswith("2013-10-28")


def test_missing_obstime_gives_none():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    assert wcs_to_celestial_frame(wcs).obstime is None


def test_non_solar_wcs_falls_through_to_astropy():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    from astropy.coordinates import ICRS

    assert isinstance(wcs_to_celestial_frame(wcs), ICRS)


def test_mixed_solar_ctypes_are_not_claimed():
    from heliox.coordinates.wcs_utils import solar_wcs_frame_mapping

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["HPLN-TAN", "HGLT-CAR"]
    assert solar_wcs_frame_mapping(wcs) is None


def test_an_explicit_coordinate_frame_wins():
    from heliox.coordinates.wcs_utils import solar_wcs_frame_mapping

    class Fake:
        coordinate_frame = "already known"

    assert solar_wcs_frame_mapping(Fake()) == "already known"


# ---------------------------------------------------------------------------
# Frame to WCS
# ---------------------------------------------------------------------------
def test_helioprojective_to_wcs():
    wcs = celestial_frame_to_wcs(Helioprojective(obstime=OBSTIME))
    assert list(wcs.wcs.ctype) == ["HPLN-TAN", "HPLT-TAN"]
    assert wcs.wcs.cunit[0].to_string() == "arcsec"


def test_projection_code_is_honoured():
    wcs = celestial_frame_to_wcs(HeliographicStonyhurst(obstime=OBSTIME), projection="CAR")
    assert list(wcs.wcs.ctype) == ["HGLN-CAR", "HGLT-CAR"]
    assert wcs.wcs.cunit[0].to_string() == "deg"


def test_heliocentric_to_wcs_uses_metres():
    wcs = celestial_frame_to_wcs(Heliocentric(obstime=OBSTIME, observer="earth"))
    assert wcs.wcs.cunit[0].to_string() == "m"


def test_observer_is_written():
    earth = get_earth(OBSTIME)
    wcs = celestial_frame_to_wcs(Helioprojective(obstime=OBSTIME, observer=earth))
    assert wcs.wcs.aux.dsun_obs == pytest.approx(earth.radius.to_value(u.m))
    assert wcs.wcs.aux.hglt_obs == pytest.approx(earth.lat.to_value(u.deg))


def test_named_observer_is_not_written():
    wcs = celestial_frame_to_wcs(Helioprojective(obstime=OBSTIME, observer="earth"))
    assert wcs.wcs.aux.dsun_obs is None


def test_obstime_is_written():
    wcs = celestial_frame_to_wcs(Helioprojective(obstime=OBSTIME))
    assert wcs.wcs.dateobs.startswith("2013-10-28")


def test_unsupported_frame_is_not_claimed():
    from heliox.coordinates import HeliocentricInertial
    from heliox.coordinates.wcs_utils import solar_frame_to_wcs_mapping

    assert solar_frame_to_wcs_mapping(HeliocentricInertial(obstime=OBSTIME)) is None


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
def test_frame_survives_a_round_trip():
    earth = get_earth(OBSTIME)
    original = Helioprojective(obstime=OBSTIME, observer=earth)
    recovered = wcs_to_celestial_frame(celestial_frame_to_wcs(original))

    assert isinstance(recovered, Helioprojective)
    assert recovered.obstime.isot == original.obstime.isot
    assert recovered.observer.radius.to_value(u.m) == pytest.approx(
        earth.radius.to_value(u.m), rel=1e-9
    )


def test_registering_twice_is_harmless():
    from astropy.wcs.utils import FRAME_WCS_MAPPINGS, WCS_FRAME_MAPPINGS

    before = (len(WCS_FRAME_MAPPINGS), len(FRAME_WCS_MAPPINGS))
    _register()
    _register()
    assert (len(WCS_FRAME_MAPPINGS), len(FRAME_WCS_MAPPINGS)) == before


def test_a_real_map_header_round_trips():
    import heliox.map
    from heliox.data.sample import AIA_171_IMAGE

    aia = heliox.map.Map(AIA_171_IMAGE)
    frame = wcs_to_celestial_frame(aia.wcs)
    assert isinstance(frame, Helioprojective)
    assert frame.observer.radius.to_value(u.m) == pytest.approx(aia.dsun.to_value(u.m), rel=1e-9)
    # And the frame the map builds itself agrees with the one from its WCS.
    assert np.isclose(
        frame.observer.lat.to_value(u.deg),
        aia.coordinate_frame.observer.lat.to_value(u.deg),
    )

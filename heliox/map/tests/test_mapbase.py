import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

import heliox.map
from heliox.coordinates import Helioprojective
from heliox.data.sample import AIA_171_IMAGE
from heliox.map import GenericMap, make_fitswcs_header
from heliox.sun import constants
from heliox.util.exceptions import HelioxMetadataWarning, MapMetaValidationError
from heliox.util.metadata import MetaDict


@pytest.fixture
def aia():
    return heliox.map.Map(AIA_171_IMAGE)


@pytest.fixture
def simple_map():
    """A small map with a single bright pixel at a known place."""
    data = np.zeros((64, 64))
    data[40, 20] = 100.0
    centre = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=Helioprojective,
        obstime="2013-10-28T12:00:00",
        observer="earth",
    )
    header = make_fitswcs_header(
        data, centre, scale=[8, 8] * u.arcsec / u.pix, instrument="TEST", unit="DN"
    )
    return GenericMap(data, header)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------
def test_data_and_meta(aia):
    assert aia.data.shape == (512, 512)
    assert isinstance(aia.meta, MetaDict)
    assert aia.meta["instrume"] == "AIA"


def test_shape_and_dimensions(aia):
    assert aia.shape == (512, 512)
    assert aia.ndim == 2
    assert list(aia.dimensions.value) == [512, 512]


def test_missing_wcs_keywords_are_reported():
    with pytest.raises(MapMetaValidationError, match="CRPIX1"):
        GenericMap(
            np.zeros((4, 4)), {"cdelt1": 1, "cdelt2": 1, "crval1": 0, "crval2": 0, "crpix2": 1}
        )


def test_three_dimensional_data_is_rejected():
    header = {f"c{key}{axis}": 1 for key in ("delt", "rpix", "rval") for axis in (1, 2)}
    with pytest.raises(ValueError, match="3 dimensions"):
        GenericMap(np.zeros((2, 2, 2)), header)


def test_missing_cunit_warns_and_defaults():
    header = {"cdelt1": 1, "cdelt2": 1, "crpix1": 1, "crpix2": 1, "crval1": 0, "crval2": 0}
    with pytest.warns(HelioxMetadataWarning, match="CUNIT1"):
        smap = GenericMap(np.zeros((4, 4)), header)
    assert smap.spatial_units[0] == u.arcsec


def test_missing_date_warns():
    header = make_fitswcs_header(
        np.zeros((4, 4)),
        SkyCoord(
            0 * u.arcsec,
            0 * u.arcsec,
            frame=Helioprojective,
            obstime="2013-10-28",
            observer="earth",
        ),
    )
    del header["date-obs"]
    smap = GenericMap(np.zeros((4, 4)), header)
    with pytest.warns(HelioxMetadataWarning, match="no observation date"):
        smap.date


# ---------------------------------------------------------------------------
# Metadata properties
# ---------------------------------------------------------------------------
def test_instrument_metadata(aia):
    assert aia.instrument == "AIA"
    assert aia.observatory == "SDO"
    assert aia.detector == "AIA"
    assert aia.exposure_time == 2 * u.s
    assert aia.processing_level == 1.5


def test_wavelength_and_unit(aia):
    assert aia.wavelength == 171 * u.angstrom
    assert aia.waveunit == u.angstrom
    assert aia.unit == u.Unit("DN")


def test_quantity_carries_the_unit(aia):
    assert aia.quantity.unit == u.Unit("DN")
    assert np.array_equal(aia.quantity.value, aia.data)


def test_unit_is_none_without_bunit(simple_map):
    del simple_map.meta["bunit"]
    assert simple_map.unit is None
    assert simple_map.quantity.unit == u.dimensionless_unscaled


def test_statistics(simple_map):
    assert simple_map.max() == 100.0
    assert simple_map.min() == 0.0
    assert simple_map.mean() == pytest.approx(100 / 64**2)
    assert simple_map.std() > 0


def test_statistics_ignore_nan(simple_map):
    data = simple_map.data.copy()
    data[0, 0] = np.nan
    smap = simple_map._new_instance(data=data)
    assert smap.max() == 100.0
    assert np.isfinite(smap.mean())


def test_date_and_name(aia):
    assert aia.date.isot == "2013-10-28T12:00:00.000"
    assert "AIA" in aia.name
    assert aia.date_start == aia.date
    assert aia.date_end == aia.date + 2 * u.s


def test_nickname_is_settable(simple_map):
    simple_map.nickname = "My Telescope"
    assert simple_map.nickname == "My Telescope"


# ---------------------------------------------------------------------------
# Observer and coordinate frame
# ---------------------------------------------------------------------------
def test_observer_comes_from_the_header(aia):
    observer = aia.observer_coordinate
    assert observer.lon.to_value(u.deg) == pytest.approx(aia.meta["hgln_obs"])
    assert observer.radius.to_value(u.m) == pytest.approx(aia.meta["dsun_obs"])


def test_observer_falls_back_to_the_earth(simple_map):
    for key in ("hgln_obs", "hglt_obs", "crln_obs", "crlt_obs"):
        simple_map.meta.pop(key, None)
    assert simple_map.observer_coordinate.lon.to_value(u.deg) == pytest.approx(0, abs=1e-6)


def test_rsun_and_dsun(aia):
    assert aia.rsun_meters.to_value(u.m) == pytest.approx(constants.radius.to_value(u.m))
    assert aia.dsun.to_value(u.AU) == pytest.approx(0.9935, abs=1e-3)
    assert aia.rsun_obs.to_value(u.arcsec) == pytest.approx(965.5, abs=0.1)


def test_heliographic_angles(aia):
    assert aia.heliographic_latitude.to_value(u.deg) == pytest.approx(4.723, abs=1e-3)
    assert aia.heliographic_longitude.to_value(u.deg) == pytest.approx(0, abs=1e-6)
    assert 0 <= aia.carrington_longitude.to_value(u.deg) <= 360


def test_coordinate_frame(aia):
    frame = aia.coordinate_frame
    assert isinstance(frame, Helioprojective)
    assert frame.obstime == aia.date


def test_unknown_ctype_is_reported(simple_map):
    simple_map.meta["ctype1"] = "RA---TAN"
    with pytest.raises(MapMetaValidationError, match="coordinate frame"):
        simple_map.coordinate_frame


# ---------------------------------------------------------------------------
# The WCS
# ---------------------------------------------------------------------------
def test_wcs_matches_the_header(aia):
    wcs = aia.wcs
    assert list(wcs.wcs.ctype) == ["HPLN-TAN", "HPLT-TAN"]
    assert wcs.wcs.crpix[0] == aia.meta["crpix1"]
    assert wcs.array_shape == aia.data.shape


def test_wcs_is_cached_and_invalidated(aia):
    first = aia.wcs
    assert aia.wcs is first
    aia.meta["crpix1"] = 100.0
    assert aia.wcs is not first
    assert aia.wcs.wcs.crpix[0] == 100.0


def test_reference_pixel_is_zero_based(aia):
    assert aia.reference_pixel.value[0] == aia.meta["crpix1"] - 1


def test_scale_and_units(aia):
    assert aia.scale.axis1 == aia.scale[0]
    assert aia.scale.axis1.unit == u.arcsec / u.pix
    assert aia.spatial_units == (u.arcsec, u.arcsec)


def test_rotation_matrix_defaults_to_identity(aia):
    assert np.allclose(aia.rotation_matrix, np.identity(2))
    assert aia.rotation_angle.to_value(u.deg) == pytest.approx(0)


def test_rotation_matrix_from_pc(simple_map):
    simple_map.meta["pc1_1"] = 0.0
    simple_map.meta["pc1_2"] = -1.0
    simple_map.meta["pc2_1"] = 1.0
    simple_map.meta["pc2_2"] = 0.0
    assert simple_map.rotation_angle.to_value(u.deg) == pytest.approx(90)


def test_rotation_matrix_from_crota(simple_map):
    simple_map.meta["crota2"] = 30.0
    assert simple_map.rotation_angle.to_value(u.deg) == pytest.approx(30)


def test_rotation_matrix_from_cd(simple_map):
    scale = simple_map.meta["cdelt1"]
    simple_map.meta["cd1_1"] = 0.0
    simple_map.meta["cd1_2"] = -scale
    simple_map.meta["cd2_1"] = scale
    simple_map.meta["cd2_2"] = 0.0
    assert simple_map.rotation_angle.to_value(u.deg) == pytest.approx(90)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def test_pixel_to_world_round_trip(aia):
    coordinate = aia.pixel_to_world(100 * u.pix, 200 * u.pix)
    x, y = aia.world_to_pixel(coordinate)
    assert x.to_value(u.pix) == pytest.approx(100)
    assert y.to_value(u.pix) == pytest.approx(200)


def test_reference_pixel_maps_to_the_reference_coordinate(aia):
    x, y = aia.world_to_pixel(aia.reference_coordinate)
    assert x.to_value(u.pix) == pytest.approx(aia.reference_pixel.value[0])
    assert y.to_value(u.pix) == pytest.approx(aia.reference_pixel.value[1])


def test_centre_and_corners(aia):
    assert aia.center.Tx.to_value(u.arcsec) == pytest.approx(0, abs=1e-6)
    assert aia.bottom_left_coord.Tx < aia.top_right_coord.Tx
    assert aia.bottom_left_coord.Ty < aia.top_right_coord.Ty


def test_world_to_pixel_accepts_another_frame(aia):
    # A coordinate expressed heliographically must still find the right pixel.
    from heliox.coordinates import HeliographicStonyhurst

    on_sun = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=aia.date)
    x, y = aia.world_to_pixel(on_sun)
    assert 0 < x.to_value(u.pix) < 512
    assert 0 < y.to_value(u.pix) < 512


# ---------------------------------------------------------------------------
# Saving and equality
# ---------------------------------------------------------------------------
def test_save_and_reload(aia, tmp_path):
    path = tmp_path / "saved.fits"
    aia.save(path)
    reloaded = heliox.map.Map(str(path))
    assert np.allclose(reloaded.data, aia.data)
    assert reloaded.instrument == aia.instrument
    assert reloaded.date == aia.date


def test_save_rejects_other_formats(aia, tmp_path):
    with pytest.raises(ValueError, match="only save maps as FITS"):
        aia.save(tmp_path / "x.png", filetype="png")


def test_equality(simple_map):
    assert simple_map == simple_map._new_instance()
    assert simple_map != simple_map._new_instance(data=simple_map.data + 1)
    assert simple_map != "not a map"


def test_repr_mentions_the_essentials(aia):
    text = repr(aia)
    assert "AIA" in text
    assert "512 x 512" in text
    assert "HPLN-TAN" in text


def test_new_instance_preserves_the_class(aia):
    assert type(aia._new_instance()) is type(aia)

import numpy as np
import pytest

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from heliox.coordinates import Helioprojective, get_earth
from heliox.data._synthetic import (
    make_coronagraph_image,
    make_disc_image,
    make_hdu,
    make_header,
    make_magnetogram,
)
from heliox.sun import constants


def test_disc_image_shape_and_positivity():
    image = make_disc_image((128, 128), seed=1)
    assert image.shape == (128, 128)
    assert (image >= 0).all()
    assert np.isfinite(image).all()


def test_disc_image_is_reproducible():
    assert np.array_equal(
        make_disc_image((64, 64), seed=7), make_disc_image((64, 64), seed=7)
    )


def test_different_seeds_give_different_images():
    assert not np.array_equal(
        make_disc_image((64, 64), seed=1), make_disc_image((64, 64), seed=2)
    )


def test_euv_image_is_brighter_at_the_limb_than_at_centre():
    image = make_disc_image((256, 256), wavelength=171 * u.AA, active_regions=0, seed=1)
    centre = image[128, 128]
    # Just inside the limb, along the row through disc centre.
    radius_in_pixels = int(256 / (2 * 1.28))
    limb = image[128, 128 + radius_in_pixels - 3]
    assert limb > centre


def test_visible_image_is_darker_at_the_limb():
    image = make_disc_image(
        (256, 256), wavelength=6173 * u.AA, field_of_view=1.15, active_regions=0, seed=1
    )
    centre = image[128, 128]
    radius_in_pixels = int(256 / (2 * 1.15))
    limb = image[128, 128 + radius_in_pixels - 3]
    assert 0 < limb < centre


def test_visible_image_is_dark_off_the_disc():
    image = make_disc_image(
        (256, 256), wavelength=6173 * u.AA, field_of_view=1.15, active_regions=0, seed=1
    )
    assert image[2, 2] == 0


def test_euv_image_has_emission_off_the_disc():
    image = make_disc_image((256, 256), wavelength=171 * u.AA, active_regions=0, seed=1)
    corner = image[2, 2]
    just_above_limb = image[128, int(128 + 256 / (2 * 1.28)) + 2]
    assert just_above_limb > corner


def test_active_regions_add_brightness():
    quiet = make_disc_image((128, 128), active_regions=0, seed=3)
    active = make_disc_image((128, 128), active_regions=8, seed=3)
    assert active.max() > quiet.max()


def test_magnetogram_is_roughly_balanced():
    field = make_magnetogram((256, 256), seed=1)
    # Bipolar regions mean the signed sum is far smaller than the unsigned one.
    assert abs(field.sum()) < 0.2 * np.abs(field).sum()


def test_magnetogram_has_both_polarities():
    field = make_magnetogram((256, 256), seed=1)
    assert field.max() > 100
    assert field.min() < -100


def test_magnetogram_is_zero_off_the_disc():
    field = make_magnetogram((256, 256), seed=1)
    assert field[1, 1] == 0


def test_coronagraph_occults_the_centre():
    image = make_coronagraph_image((128, 128), seed=1)
    assert image[64, 64] == 0


def test_coronagraph_brightness_falls_with_height():
    image = make_coronagraph_image((256, 256), occulter=2.0, field_of_view=6.0, seed=1)
    row = image[128, 128:]
    # Compare an annulus just outside the occulter with one further out,
    # averaging over position angle to smooth out the streamers.
    inner = image[:, :][np.hypot(*np.indices((256, 256)) - 127.5) < 60].mean()
    outer = image[:, :][np.hypot(*np.indices((256, 256)) - 127.5) > 100].mean()
    assert inner > outer
    assert row.size == 128


def test_header_has_a_valid_wcs():
    header = make_header((256, 256))
    wcs = WCS(header)
    assert list(wcs.wcs.ctype) == ["HPLN-TAN", "HPLT-TAN"]
    assert header["CUNIT1"] == "arcsec"
    # Astropy normalises the WCS to degrees when it parses the header.
    assert wcs.wcs.cunit[0].to_string() == "deg"
    assert wcs.wcs.cdelt[0] * 3600 == pytest.approx(header["CDELT1"])


def test_header_reference_pixel_is_the_centre_of_the_array():
    header = make_header((256, 256))
    wcs = WCS(header)
    # The reference pixel is disc centre, so it must map to zero arcseconds.
    world = wcs.pixel_to_world_values(header["CRPIX1"] - 1, header["CRPIX2"] - 1)
    assert world[0] == pytest.approx(0, abs=1e-9)
    assert world[1] == pytest.approx(0, abs=1e-9)


def test_header_scale_puts_the_limb_where_expected():
    header = make_header((256, 256), field_of_view=1.28)
    limb_in_pixels = header["RSUN_OBS"] / header["CDELT1"]
    assert limb_in_pixels == pytest.approx(256 / (2 * 1.28), rel=1e-6)


def test_header_observer_matches_the_earth():
    header = make_header((64, 64), obstime="2013-10-28T12:00:00")
    earth = get_earth("2013-10-28T12:00:00")
    assert header["DSUN_OBS"] == pytest.approx(earth.radius.to_value(u.m))
    assert header["HGLT_OBS"] == pytest.approx(earth.lat.to_value(u.deg))
    assert header["RSUN_REF"] == pytest.approx(constants.radius.to_value(u.m))


def test_header_round_trips_through_the_frame_mapping():
    from astropy.wcs.utils import wcs_to_celestial_frame

    header = make_header((64, 64))
    frame = wcs_to_celestial_frame(WCS(header))
    assert isinstance(frame, Helioprojective)
    assert frame.observer.radius.to_value(u.m) == pytest.approx(header["DSUN_OBS"])


@pytest.mark.parametrize("kind", ["aia", "hmi", "continuum", "lasco"])
def test_every_hdu_kind_is_well_formed(kind):
    hdu = make_hdu(kind, (64, 64), seed=1)
    assert isinstance(hdu, fits.PrimaryHDU)
    assert hdu.data.shape == (64, 64)
    assert hdu.data.dtype == np.float32
    assert WCS(hdu.header).naxis == 2


def test_hdu_instrument_keywords():
    assert make_hdu("aia", (32, 32), seed=1).header["INSTRUME"] == "AIA"
    assert make_hdu("hmi", (32, 32), seed=1).header["INSTRUME"] == "HMI"
    assert make_hdu("lasco", (32, 32), seed=1).header["DETECTOR"] == "C2"


def test_unknown_hdu_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown sample image kind"):
        make_hdu("telescope-that-does-not-exist")

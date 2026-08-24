import numpy as np
import pytest

import astropy.units as u

from heliox.sun.models import (
    DIFFERENTIAL_ROTATION_MODELS,
    differential_rotation,
    limb_darkening,
    sunspot_number_to_flux,
)


def test_equator_rotates_faster_than_the_poles():
    equator = differential_rotation(1 * u.day, 0 * u.deg)
    pole = differential_rotation(1 * u.day, 75 * u.deg)
    assert equator > pole


def test_rotation_is_symmetric_about_the_equator():
    north = differential_rotation(1 * u.day, 30 * u.deg)
    south = differential_rotation(1 * u.day, -30 * u.deg)
    assert north == south


def test_equatorial_period_is_about_25_days():
    rate = differential_rotation(1 * u.day, 0 * u.deg)
    period = (360 * u.deg / rate) * u.day
    assert period.to_value(u.day) == pytest.approx(25.1, abs=0.5)


def test_polar_period_is_longer_than_30_days():
    rate = differential_rotation(1 * u.day, 75 * u.deg)
    period = (360 * u.deg / rate) * u.day
    assert period.to_value(u.day) > 30


def test_synodic_is_slower_than_sidereal():
    sidereal = differential_rotation(1 * u.day, 0 * u.deg, frame_time="sidereal")
    synodic = differential_rotation(1 * u.day, 0 * u.deg, frame_time="synodic")
    assert (sidereal - synodic).to_value(u.deg) == pytest.approx(0.9856, abs=1e-3)


def test_synodic_equatorial_period_is_about_27_days():
    rate = differential_rotation(1 * u.day, 0 * u.deg, frame_time="synodic")
    period = (360 * u.deg / rate).to_value(u.dimensionless_unscaled)
    assert period == pytest.approx(26.9, abs=0.6)


def test_rigid_model_has_no_latitude_dependence():
    equator = differential_rotation(1 * u.day, 0 * u.deg, model="rigid")
    pole = differential_rotation(1 * u.day, 80 * u.deg, model="rigid")
    assert equator == pole
    assert equator.to_value(u.deg) == pytest.approx(14.1844)


def test_negative_duration_rotates_backwards():
    assert differential_rotation(-1 * u.day, 0 * u.deg) < 0 * u.deg


def test_array_of_latitudes():
    result = differential_rotation(1 * u.day, [0, 30, 60] * u.deg)
    assert result.shape == (3,)
    assert np.all(np.diff(result.value) < 0)


def test_every_named_model_is_usable():
    for name in DIFFERENTIAL_ROTATION_MODELS:
        assert differential_rotation(1 * u.day, 0 * u.deg, model=name) > 0 * u.deg


def test_custom_coefficients():
    custom = [14.0, 0.0, 0.0] * u.deg / u.day
    assert differential_rotation(1 * u.day, 45 * u.deg, model=custom).to_value(u.deg) == 14.0


def test_bad_model_name():
    with pytest.raises(ValueError, match="Unknown rotation model"):
        differential_rotation(1 * u.day, 0 * u.deg, model="nonsense")


def test_wrong_number_of_coefficients():
    with pytest.raises(ValueError, match="exactly three coefficients"):
        differential_rotation(1 * u.day, 0 * u.deg, model=[14.0, 0.0] * u.deg / u.day)


def test_bad_frame_time():
    with pytest.raises(ValueError, match="sidereal.*synodic"):
        differential_rotation(1 * u.day, 0 * u.deg, frame_time="tropical")


def test_limb_darkening_is_brightest_at_disc_centre():
    assert limb_darkening(0.0) == pytest.approx(1.0)


def test_limb_darkening_decreases_outwards():
    profile = limb_darkening(np.linspace(0, 1, 50))
    assert np.all(np.diff(profile) < 0)


def test_limb_darkening_is_zero_off_the_disc():
    assert limb_darkening(1.5) == 0.0
    assert np.all(limb_darkening([1.01, 2.0, 10.0]) == 0.0)


def test_limb_darkening_stays_in_range():
    profile = limb_darkening(np.linspace(0, 1.2, 100))
    assert profile.min() >= 0.0
    assert profile.max() <= 1.0


def test_limb_darkening_is_stronger_in_the_blue():
    blue = limb_darkening(0.95, wavelength=4000 * u.AA)
    red = limb_darkening(0.95, wavelength=8000 * u.AA)
    assert blue < red


def test_limb_darkening_accepts_explicit_coefficients():
    assert limb_darkening(0.0, coefficients=(0.9, -0.2)) == pytest.approx(1.0)
    # At the very limb, mu = 0, so the intensity is 1 - u1 - u2.
    assert limb_darkening(1.0, coefficients=(0.9, -0.2)) == pytest.approx(0.3)


def test_flux_floor_at_zero_sunspots():
    assert sunspot_number_to_flux(0).to_value("sfu") == pytest.approx(63.7)


def test_flux_increases_with_sunspot_number():
    flux = sunspot_number_to_flux([0, 50, 100, 200])
    assert np.all(np.diff(flux.value) > 0)


def test_solar_maximum_flux_is_plausible():
    # At a sunspot number of 150 the F10.7 index is typically around 190 sfu.
    flux = sunspot_number_to_flux(150).to_value("sfu")
    assert 170 < flux < 210


def test_negative_sunspot_number_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        sunspot_number_to_flux(-1)

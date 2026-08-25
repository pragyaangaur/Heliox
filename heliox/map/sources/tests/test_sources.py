import numpy as np
import pytest

import astropy.units as u

import heliox.map
from heliox.data.sample import (
    AIA_171_IMAGE,
    HMI_CONTINUUM_IMAGE,
    HMI_MAGNETOGRAM,
    LASCO_C2_IMAGE,
)
from heliox.map.sources import AIAMap, EITMap, HMIMap, LASCOMap


# ---------------------------------------------------------------------------
# AIA
# ---------------------------------------------------------------------------
@pytest.fixture
def aia():
    return heliox.map.Map(AIA_171_IMAGE)


def test_aia_is_recognised(aia):
    assert isinstance(aia, AIAMap)
    assert aia.observatory == "SDO"
    assert aia.detector == "AIA"
    assert aia.nickname == "AIA 171"


def test_aia_uses_its_own_colour_table(aia):
    assert aia.plot_settings["cmap"].name == "sdoaia171"


def test_aia_norm_is_asinh(aia):
    from astropy.visualization import AsinhStretch

    assert isinstance(aia.plot_settings["norm"].stretch, AsinhStretch)


def test_aia_norm_covers_the_data(aia):
    norm = aia.plot_settings["norm"]
    assert norm.vmin >= 0
    assert norm.vmax <= aia.data.max()
    assert norm.vmin < norm.vmax


def test_aia_survives_an_unknown_passband(aia):
    aia.meta["wavelnth"] = 999
    remade = AIAMap(aia.data, aia.meta)
    # No AIA table for 999 angstroms, so the inherited grey scale is used.
    assert remade.plot_settings["cmap"] == "gray"


def test_aia_handles_a_flat_image(aia):
    flat = AIAMap(np.zeros_like(aia.data), aia.meta)
    assert flat.plot_settings["norm"] is None


def test_aia_validation():
    assert AIAMap.is_datasource_for(None, {"instrume": "AIA_3"})
    assert not AIAMap.is_datasource_for(None, {"instrume": "HMI"})
    assert not AIAMap.is_datasource_for(None, {})


def test_aia_survives_cropping(aia):
    cropped = aia.submap([0, 0] * u.pix, top_right=[63, 63] * u.pix)
    assert isinstance(cropped, AIAMap)
    assert cropped.nickname == "AIA 171"


# ---------------------------------------------------------------------------
# HMI
# ---------------------------------------------------------------------------
@pytest.fixture
def magnetogram():
    return heliox.map.Map(HMI_MAGNETOGRAM)


def test_hmi_magnetogram_is_recognised(magnetogram):
    assert isinstance(magnetogram, HMIMap)
    assert magnetogram.observatory == "SDO"
    assert magnetogram.measurement == "magnetogram"
    assert magnetogram.is_magnetogram


def test_magnetogram_scale_is_symmetric(magnetogram):
    norm = magnetogram.plot_settings["norm"]
    assert norm.vmin == pytest.approx(-norm.vmax)
    assert magnetogram.plot_settings["cmap"].name == "hmimag"


def test_hmi_continuum_is_not_a_magnetogram():
    continuum = heliox.map.Map(HMI_CONTINUUM_IMAGE)
    assert isinstance(continuum, HMIMap)
    assert continuum.measurement == "continuum"
    assert not continuum.is_magnetogram
    assert continuum.plot_settings["cmap"] == "afmhot"


def test_hmi_measurement_falls_back(magnetogram):
    del magnetogram.meta["content"]
    assert HMIMap(magnetogram.data, magnetogram.meta).measurement == "hmi"


def test_hmi_validation():
    assert HMIMap.is_datasource_for(None, {"instrume": "HMI"})
    assert not HMIMap.is_datasource_for(None, {"instrume": "AIA"})


# ---------------------------------------------------------------------------
# LASCO
# ---------------------------------------------------------------------------
@pytest.fixture
def lasco():
    return heliox.map.Map(LASCO_C2_IMAGE)


def test_lasco_is_recognised(lasco):
    assert isinstance(lasco, LASCOMap)
    assert lasco.observatory == "SOHO"
    assert lasco.detector == "C2"
    assert lasco.nickname == "LASCO C2"
    assert lasco.measurement == "white-light"


def test_lasco_uses_a_log_stretch(lasco):
    from astropy.visualization import LogStretch

    assert isinstance(lasco.plot_settings["norm"].stretch, LogStretch)
    assert lasco.plot_settings["cmap"].name == "soholasco2"


def test_lasco_c3_uses_its_own_table(lasco):
    lasco.meta["detector"] = "C3"
    assert LASCOMap(lasco.data, lasco.meta).plot_settings["cmap"].name == "soholasco3"


def test_lasco_handles_an_empty_image(lasco):
    empty = LASCOMap(np.zeros_like(lasco.data), lasco.meta)
    assert "norm" not in empty.plot_settings or empty.plot_settings["norm"] is None


def test_lasco_validation():
    assert LASCOMap.is_datasource_for(None, {"instrume": "LASCO"})
    assert not LASCOMap.is_datasource_for(None, {"instrume": "EIT"})


# ---------------------------------------------------------------------------
# EIT
# ---------------------------------------------------------------------------
def test_eit_is_recognised(aia):
    meta = aia.meta.copy()
    meta["instrume"] = "EIT"
    meta["telescop"] = "SOHO"
    meta["obsrvtry"] = "SOHO"
    meta["wavelnth"] = 195
    eit = heliox.map.Map(aia.data, meta)
    assert isinstance(eit, EITMap)
    assert eit.observatory == "SOHO"
    assert eit.nickname == "EIT 195"
    # 195 is closest to AIA's 193 channel.
    assert eit.plot_settings["cmap"].name == "sdoaia193"


def test_eit_validation():
    assert EITMap.is_datasource_for(None, {"instrume": "EIT"})
    assert not EITMap.is_datasource_for(None, {"instrume": "AIA"})

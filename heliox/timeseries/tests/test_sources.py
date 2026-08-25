import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import astropy.units as u  # noqa: E402

from heliox.data.sample import GOES_XRS_TIMESERIES, NOAA_INDICES_TIMESERIES  # noqa: E402
from heliox.timeseries import TimeSeries  # noqa: E402
from heliox.timeseries.sources.goes import (  # noqa: E402
    FLARE_CLASSES,
    XRSTimeSeries,
    flare_class,
    flux_from_flare_class,
)
from heliox.timeseries.sources.noaa import NOAAIndicesTimeSeries  # noqa: E402
from heliox.util.units import sfu  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def goes():
    return TimeSeries(GOES_XRS_TIMESERIES)


@pytest.fixture
def noaa():
    return TimeSeries(NOAA_INDICES_TIMESERIES)


# ---------------------------------------------------------------------------
# Flare classes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flux, expected",
    [
        (5.4e-6, "C5.4"),
        (2.3e-4, "X2.3"),
        (1.0e-5, "M1.0"),
        (1.0e-7, "B1.0"),
        (3.2e-8, "A3.2"),
        (1.0e-9, "A0.0"),
        (0.0, "A0.0"),
        (1.5e-3, "X15.0"),
    ],
)
def test_flare_class(flux, expected):
    assert flare_class(flux * u.W / u.m**2) == expected


def test_flare_class_accepts_plain_numbers():
    assert flare_class(5.4e-6) == "C5.4"


def test_flare_class_handles_nan():
    assert flare_class(np.nan) == "A0.0"


def test_flare_class_is_vectorised():
    result = flare_class(np.array([1e-6, 1e-5, 1e-4]))
    assert result == ["C1.0", "M1.0", "X1.0"]


@pytest.mark.parametrize("name", ["A1.0", "B5.0", "C2.5", "M1.5", "X10.0"])
def test_flare_class_round_trips(name):
    assert flare_class(flux_from_flare_class(name)) == name


def test_flux_from_flare_class():
    assert flux_from_flare_class("M1.5").to_value(u.W / u.m**2) == pytest.approx(1.5e-5)
    assert flux_from_flare_class("X").to_value(u.W / u.m**2) == pytest.approx(1e-4)


def test_flux_from_flare_class_rejects_nonsense():
    with pytest.raises(ValueError, match="not a flare class"):
        flux_from_flare_class("Z1.0")
    with pytest.raises(ValueError, match="not a flare class"):
        flux_from_flare_class("")
    with pytest.raises(ValueError, match="Could not read a magnitude"):
        flux_from_flare_class("Mbig")


def test_class_thresholds_are_decades():
    values = list(FLARE_CLASSES.values())
    assert all(
        pytest.approx(values[index + 1] / values[index]) == 10
        for index in range(len(values) - 1)
    )


# ---------------------------------------------------------------------------
# XRS
# ---------------------------------------------------------------------------
def test_goes_is_recognised(goes):
    assert isinstance(goes, XRSTimeSeries)
    assert goes.observatory == "GOES-15"
    assert goes.instrument == "XRS"


def test_goes_units(goes):
    assert goes.units["xrsa"] == u.Unit("W/m2")
    assert goes.units["xrsb"] == u.Unit("W/m2")


def test_peak_flux_and_time(goes):
    assert goes.peak_flux.to_value(u.W / u.m**2) == pytest.approx(
        goes.data["xrsb"].max()
    )
    assert goes.peak_time in goes.time


def test_flare_class_of_the_series(goes):
    assert goes.flare_class == flare_class(goes.peak_flux)
    assert goes.flare_class[0] in FLARE_CLASSES


def test_flare_classes_for_every_sample(goes):
    classes = goes.flare_classes()
    assert len(classes) == len(goes)


def test_long_channel_exceeds_the_short_one(goes):
    assert (goes.data["xrsb"] > goes.data["xrsa"]).all()


def test_goes_plot_is_logarithmic(goes):
    axes = goes.plot()
    assert axes.get_yscale() == "log"
    assert "W m$^{-2}$" in axes.get_ylabel()


def test_goes_plot_title(goes):
    assert "X-ray sensor" in goes.plot().get_title()


def test_goes_validation():
    frame = pd.DataFrame(
        {"xrsa": [1.0], "xrsb": [2.0]},
        index=pd.DatetimeIndex(["2013-10-28"]),
    )
    assert XRSTimeSeries.is_datasource_for(frame, {"instrume": "XRS"})
    assert XRSTimeSeries.is_datasource_for(frame, {})
    assert not XRSTimeSeries.is_datasource_for(
        pd.DataFrame({"a": [1.0]}, index=pd.DatetimeIndex(["2013-10-28"])), {}
    )


def test_goes_survives_truncation(goes):
    part = goes.truncate("2013-10-28T00:00", "2013-10-28T06:00")
    assert isinstance(part, XRSTimeSeries)
    assert part.flare_class


# ---------------------------------------------------------------------------
# NOAA
# ---------------------------------------------------------------------------
def test_noaa_is_recognised(noaa):
    assert isinstance(noaa, NOAAIndicesTimeSeries)
    assert noaa.instrument == "NOAA-Indices"


def test_noaa_attaches_the_radio_flux_unit(noaa):
    assert noaa.units["f10.7"] == sfu
    assert noaa.units["sunspot_number"] == u.dimensionless_unscaled


def test_sunspot_column(noaa):
    assert noaa.sunspot_column == "sunspot_number"


def test_solar_maximum_is_inside_the_series(noaa):
    assert noaa.solar_maximum in noaa.time


def test_solar_maximum_needs_a_sunspot_column(noaa):
    without = noaa.remove_column("sunspot_number")
    with pytest.raises(ValueError, match="no sunspot number column"):
        without.solar_maximum


def test_smoothing_reduces_the_scatter(noaa):
    smoothed = noaa.smooth(13)
    assert len(smoothed) == len(noaa)
    assert smoothed.data["sunspot_number"].std() < noaa.data["sunspot_number"].std()


def test_smoothing_rejects_a_bad_window(noaa):
    with pytest.raises(ValueError, match="at least one sample"):
        noaa.smooth(0)


def test_noaa_plot_title(noaa):
    assert "NOAA" in noaa.plot().get_title()


def test_noaa_validation():
    frame = pd.DataFrame(
        {"sunspot_number": [1.0]}, index=pd.DatetimeIndex(["2013-10-28"])
    )
    assert NOAAIndicesTimeSeries.is_datasource_for(frame, {"instrume": "NOAA-Indices"})
    assert NOAAIndicesTimeSeries.is_datasource_for(frame, {})
    assert not NOAAIndicesTimeSeries.is_datasource_for(
        pd.DataFrame({"a": [1.0]}, index=pd.DatetimeIndex(["2013-10-28"])), {}
    )


def test_generic_plotting(noaa):
    axes = noaa.plot(columns=["f10.7"])
    assert axes.get_ylabel() == "sfu"


def test_peek_returns_a_figure(goes):
    assert goes.peek().axes

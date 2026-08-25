"""
Synthetic solar images and light curves.

Real solar data comes from archives that need network access, credentials and
gigabytes of disk. For examples, tests and documentation that is more friction
than it is worth, so heliox generates its own: images with a plausible
limb-darkened disc, active regions, granulation and an off-limb corona, wrapped
in headers that carry a correct World Coordinate System and observer position.

The physics is deliberately simple. These images are for exercising the
software, not for doing science.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

import astropy.units as u
from astropy.io import fits

from heliox.coordinates import get_earth
from heliox.sun import constants
from heliox.sun.models import limb_darkening
from heliox.time import parse_time

__all__ = [
    "make_disc_image",
    "make_magnetogram",
    "make_coronagraph_image",
    "make_header",
    "make_hdu",
    "make_xray_lightcurve",
    "make_sunspot_series",
]


def _radial_grid(shape, centre, radius_in_pixels):
    """Fractional distance from disc centre, in units of the solar radius."""
    rows = np.arange(shape[0])[:, np.newaxis]
    cols = np.arange(shape[1])[np.newaxis, :]
    return np.hypot(cols - centre[0], rows - centre[1]) / radius_in_pixels


def _active_regions(shape, radial, rng, count, amplitude, size):
    """Place bright blobs on the visible disc, avoiding the poles."""
    field = np.zeros(shape, dtype=float)
    rows = np.arange(shape[0])[:, np.newaxis]
    cols = np.arange(shape[1])[np.newaxis, :]

    placed = 0
    attempts = 0
    while placed < count and attempts < 200 * count:
        attempts += 1
        # Active regions cluster within about 30 degrees of the equator, which
        # projects to the middle band of the disc.
        fractional_radius = rng.uniform(0, 0.9)
        angle = rng.uniform(0, 2 * np.pi)
        x = shape[1] / 2 + fractional_radius * np.cos(angle) * shape[1] / 2 * 0.95
        y = shape[0] / 2 + fractional_radius * np.sin(angle) * shape[0] / 2 * 0.55

        if radial[int(np.clip(y, 0, shape[0] - 1)), int(np.clip(x, 0, shape[1] - 1))] > 0.95:
            continue

        width = size * rng.uniform(0.6, 1.6)
        field += (
            amplitude
            * rng.uniform(0.4, 1.0)
            * np.exp(-((cols - x) ** 2 + (rows - y) ** 2) / (2 * width**2))
        )
        placed += 1
    return field


def make_disc_image(
    shape=(1024, 1024),
    *,
    wavelength=171 * u.AA,
    field_of_view=1.28,
    active_regions=6,
    seed=None,
):
    """
    Generate a synthetic image of the solar disc.

    Extreme ultraviolet wavelengths show a corona that is brighter at the limb
    than at disc centre, because the line of sight passes through more hot
    plasma there; visible wavelengths show the opposite, the familiar limb
    darkening. Both cases are produced here, selected by ``wavelength``.

    Parameters
    ----------
    shape : tuple of `int`, optional
        The shape of the array, as ``(rows, columns)``.
    wavelength : `~astropy.units.Quantity`, optional
        The observing wavelength. Anything below 2000 angstroms is treated as
        a coronal EUV channel.
    field_of_view : `float`, optional
        The width of the image in solar radii. The default of 1.28 puts the
        limb comfortably inside the frame, as SDO/AIA does.
    active_regions : `int`, optional
        How many bright regions to scatter over the disc.
    seed : `int`, optional
        Seed for the random number generator, so images can be reproduced.

    Returns
    -------
    `numpy.ndarray`
        A 2D array of floats.

    Examples
    --------
    >>> from heliox.data._synthetic import make_disc_image
    >>> image = make_disc_image((64, 64), seed=1)
    >>> image.shape
    (64, 64)
    >>> bool((image >= 0).all())
    True
    """
    rng = np.random.default_rng(seed)
    radius_in_pixels = min(shape) / (2 * field_of_view)
    centre = ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    radial = _radial_grid(shape, centre, radius_in_pixels)

    on_disc = radial <= 1.0
    is_euv = u.Quantity(wavelength, u.AA) < 2000 * u.AA

    image = np.zeros(shape, dtype=float)

    if is_euv:
        # Optically thin emission: brightness grows towards the limb because
        # the line of sight is longer, then falls off above it.
        with np.errstate(invalid="ignore", divide="ignore"):
            path_length = np.where(
                on_disc, 1.0 / np.sqrt(1.0 - np.clip(radial, 0, 0.995) ** 2), 0.0
            )
        image[on_disc] = 100.0 * np.clip(path_length[on_disc], 1.0, 4.0)
        # Coronal emission above the limb, decaying with height.
        above = ~on_disc
        image[above] = 400.0 * np.exp(-(radial[above] - 1.0) / 0.13)
        blob_amplitude = 1200.0
    else:
        image[on_disc] = 1000.0 * limb_darkening(radial[on_disc], wavelength=wavelength)
        blob_amplitude = 250.0

    # Granulation: correlated noise, only on the disc.
    granulation = gaussian_filter(rng.normal(size=shape), sigma=max(shape) / 400)
    granulation /= np.abs(granulation).max() or 1.0
    image = image * (1.0 + 0.08 * granulation * on_disc)

    if active_regions:
        image = image + _active_regions(
            shape, radial, rng, active_regions, blob_amplitude, max(shape) / 40
        )

    # Photon noise, and a small dark level so nothing is exactly zero.
    image = image + rng.normal(scale=np.sqrt(np.clip(image, 1.0, None)) * 0.3)
    return np.clip(image, 0.0, None)


def make_magnetogram(shape=(1024, 1024), *, field_of_view=1.15, pairs=5, seed=None):
    """
    Generate a synthetic line-of-sight magnetogram.

    Active regions on the Sun are bipolar, so each region is drawn as a pair of
    opposite-polarity blobs. Field strengths run to a couple of thousand gauss
    in the strongest spots and a few gauss in the quiet Sun.

    Parameters
    ----------
    shape : tuple of `int`, optional
        The shape of the array.
    field_of_view : `float`, optional
        The width of the image in solar radii.
    pairs : `int`, optional
        How many bipolar regions to place.
    seed : `int`, optional
        Seed for the random number generator.

    Returns
    -------
    `numpy.ndarray`
        Signed field strength in gauss, zero outside the disc.

    Examples
    --------
    >>> from heliox.data._synthetic import make_magnetogram
    >>> field = make_magnetogram((64, 64), seed=1)
    >>> bool(abs(field.sum()) < abs(field).sum())
    True
    """
    rng = np.random.default_rng(seed)
    radius_in_pixels = min(shape) / (2 * field_of_view)
    centre = ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    radial = _radial_grid(shape, centre, radius_in_pixels)
    on_disc = radial <= 1.0

    rows = np.arange(shape[0])[:, np.newaxis]
    cols = np.arange(shape[1])[np.newaxis, :]
    field = np.zeros(shape, dtype=float)

    for _ in range(pairs):
        fractional_radius = rng.uniform(0, 0.85)
        angle = rng.uniform(0, 2 * np.pi)
        x = shape[1] / 2 + fractional_radius * np.cos(angle) * shape[1] / 2 * 0.9
        y = shape[0] / 2 + fractional_radius * np.sin(angle) * shape[0] / 2 * 0.5

        separation = max(shape) / 25 * rng.uniform(0.8, 1.5)
        tilt = rng.uniform(-0.5, 0.5)  # Joy's law: leading spot closer to the equator
        width = max(shape) / 70 * rng.uniform(0.7, 1.4)
        strength = rng.uniform(500, 2500)

        for sign, offset in ((1, -separation / 2), (-1, separation / 2)):
            field += (
                sign
                * strength
                * np.exp(
                    -((cols - (x + offset)) ** 2 + (rows - (y + offset * tilt)) ** 2)
                    / (2 * width**2)
                )
            )

    # Quiet-Sun salt-and-pepper network.
    network = gaussian_filter(rng.normal(size=shape), sigma=max(shape) / 300)
    network *= 20.0 / (np.abs(network).max() or 1.0)
    return np.where(on_disc, field + network, 0.0)


def make_coronagraph_image(shape=(512, 512), *, occulter=2.2, field_of_view=6.0, seed=None):
    """
    Generate a synthetic white-light coronagraph image.

    A coronagraph blocks the disc with an occulting disc and records the faint
    corona around it, which falls off steeply with height and is structured
    into radial streamers.

    Parameters
    ----------
    shape : tuple of `int`, optional
        The shape of the array.
    occulter : `float`, optional
        The radius of the occulting disc, in solar radii.
    field_of_view : `float`, optional
        The width of the image in solar radii.
    seed : `int`, optional
        Seed for the random number generator.

    Returns
    -------
    `numpy.ndarray`

    Examples
    --------
    >>> from heliox.data._synthetic import make_coronagraph_image
    >>> image = make_coronagraph_image((64, 64), seed=1)
    >>> float(image[32, 32])
    0.0
    """
    rng = np.random.default_rng(seed)
    radius_in_pixels = min(shape) / (2 * field_of_view)
    centre = ((shape[1] - 1) / 2, (shape[0] - 1) / 2)

    rows = np.arange(shape[0])[:, np.newaxis]
    cols = np.arange(shape[1])[np.newaxis, :]
    dx = cols - centre[0]
    dy = rows - centre[1]
    radial = np.hypot(dx, dy) / radius_in_pixels
    position_angle = np.arctan2(dy, dx)

    # The K corona falls off roughly as r^-7 close in, flattening further out.
    with np.errstate(divide="ignore", invalid="ignore"):
        brightness = 1e4 * np.clip(radial, 1.0, None) ** -3.5

    # Streamers: a few broad enhancements at fixed position angles, brightest
    # near the equator as they are at solar minimum.
    streamers = np.zeros(shape)
    for _ in range(4):
        angle = rng.uniform(0, 2 * np.pi)
        width = rng.uniform(0.15, 0.4)
        separation = np.arctan2(np.sin(position_angle - angle), np.cos(position_angle - angle))
        streamers += rng.uniform(0.5, 1.5) * np.exp(-(separation**2) / (2 * width**2))
    brightness = brightness * (1.0 + streamers)

    brightness = np.where(radial < occulter, 0.0, brightness)
    brightness += rng.normal(scale=2.0, size=shape) * (radial >= occulter)
    return np.clip(brightness, 0.0, None)


def make_header(
    shape,
    *,
    obstime="2013-10-28T12:00:00",
    field_of_view=1.28,
    instrument="AIA",
    telescope="SDO",
    detector="AIA",
    wavelength=171 * u.AA,
    unit="DN",
    exposure_time=2.0 * u.s,
    observatory=None,
):
    """
    Build a FITS header describing a synthetic solar image.

    The header carries a complete helioprojective WCS, the observer's
    heliographic position and distance, and the usual instrument keywords, so
    that anything reading it -- including `heliox.map.Map` -- has everything it
    needs.

    Parameters
    ----------
    shape : tuple of `int`
        The shape of the image the header describes.
    obstime : time-like, optional
        The observation time.
    field_of_view : `float`, optional
        The width of the image in solar radii, which sets the plate scale.
    instrument, telescope, detector : `str`, optional
        Instrument identification keywords.
    wavelength : `~astropy.units.Quantity`, optional
        The observing wavelength.
    unit : `str`, optional
        The value of ``BUNIT``.
    exposure_time : `~astropy.units.Quantity`, optional
        The exposure time.
    observatory : `str`, optional
        The value of ``OBSRVTRY``; defaults to ``telescope``.

    Returns
    -------
    `astropy.io.fits.Header`

    Examples
    --------
    >>> from heliox.data._synthetic import make_header
    >>> header = make_header((64, 64))
    >>> header['CTYPE1']
    'HPLN-TAN'
    >>> round(float(header['CDELT1']), 2)
    38.62
    """
    time = parse_time(obstime)
    earth = get_earth(time)

    # arcsin, matching Helioprojective.angular_radius: the limb is the
    # tangent point on the solar sphere.
    angular_radius = np.arcsin(constants.radius / earth.radius).to(u.arcsec)
    half_width = field_of_view * angular_radius
    scale = float((2 * half_width / min(shape)).to_value(u.arcsec))

    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = shape[1]
    header["NAXIS2"] = shape[0]

    # World coordinate system. FITS pixel coordinates are one-based, so the
    # centre of an N-pixel axis is at (N + 1) / 2.
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = (shape[1] + 1) / 2
    header["CRPIX2"] = (shape[0] + 1) / 2
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = scale
    header["CDELT2"] = scale
    header["CROTA2"] = 0.0

    # Where the observer was.
    header["DATE-OBS"] = time.utc.isot
    header["DSUN_OBS"] = float(earth.radius.to_value(u.m))
    header["HGLN_OBS"] = float(earth.lon.to_value(u.deg))
    header["HGLT_OBS"] = float(earth.lat.to_value(u.deg))
    header["RSUN_REF"] = float(constants.radius.to_value(u.m))
    header["RSUN_OBS"] = float(angular_radius.to_value(u.arcsec))

    # Instrument identification.
    header["TELESCOP"] = telescope
    header["INSTRUME"] = instrument
    header["DETECTOR"] = detector
    header["OBSRVTRY"] = observatory or telescope
    header["WAVELNTH"] = float(u.Quantity(wavelength, u.AA).to_value(u.AA))
    header["WAVEUNIT"] = "angstrom"
    header["EXPTIME"] = float(u.Quantity(exposure_time, u.s).to_value(u.s))
    header["BUNIT"] = unit
    header["LVL_NUM"] = 1.5

    return header


def make_hdu(
    kind="aia",
    shape=(1024, 1024),
    *,
    obstime="2013-10-28T12:00:00",
    wavelength=None,
    seed=None,
):
    """
    Build a complete synthetic FITS HDU.

    Parameters
    ----------
    kind : {'aia', 'hmi', 'lasco', 'continuum'}, optional
        Which kind of instrument to imitate.
    shape : tuple of `int`, optional
        The shape of the image.
    obstime : time-like, optional
        The observation time.
    wavelength : `~astropy.units.Quantity`, optional
        The passband, for the ``'aia'`` kind. Defaults to 171 angstroms.
    seed : `int`, optional
        Seed for the random number generator.

    Returns
    -------
    `astropy.io.fits.PrimaryHDU`

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.data._synthetic import make_hdu
    >>> make_hdu('hmi', (64, 64), seed=1).header['INSTRUME']
    'HMI'
    >>> make_hdu('aia', (64, 64), wavelength=193 * u.AA, seed=1).header['WAVELNTH']
    193.0
    """
    kind = kind.lower()
    if kind == "aia":
        passband = 171 * u.AA if wavelength is None else u.Quantity(wavelength, u.AA)
        data = make_disc_image(shape, wavelength=passband, seed=seed)
        header = make_header(
            shape,
            obstime=obstime,
            instrument="AIA",
            telescope="SDO",
            detector="AIA",
            wavelength=passband,
            unit="DN",
        )
    elif kind == "hmi":
        data = make_magnetogram(shape, seed=seed)
        header = make_header(
            shape,
            obstime=obstime,
            field_of_view=1.15,
            instrument="HMI",
            telescope="SDO",
            detector="HMI_FRONT2",
            wavelength=6173 * u.AA,
            unit="Gauss",
            exposure_time=0.0 * u.s,
        )
        header["CONTENT"] = "MAGNETOGRAM"
    elif kind == "continuum":
        data = make_disc_image(shape, wavelength=6173 * u.AA, field_of_view=1.15, seed=seed)
        header = make_header(
            shape,
            obstime=obstime,
            field_of_view=1.15,
            instrument="HMI",
            telescope="SDO",
            detector="HMI_FRONT2",
            wavelength=6173 * u.AA,
            unit="DN",
        )
        header["CONTENT"] = "CONTINUUM INTENSITY"
    elif kind == "lasco":
        data = make_coronagraph_image(shape, seed=seed)
        header = make_header(
            shape,
            obstime=obstime,
            field_of_view=6.0,
            instrument="LASCO",
            telescope="SOHO",
            detector="C2",
            wavelength=5500 * u.AA,
            unit="DN",
            exposure_time=25.0 * u.s,
        )
    else:
        raise ValueError(
            f"Unknown sample image kind {kind!r}. Choose from 'aia', 'hmi', 'continuum' or 'lasco'."
        )

    # Single precision is plenty for synthetic data and halves the size of
    # the cached files.
    return fits.PrimaryHDU(data=data.astype(np.float32), header=header)


def make_xray_lightcurve(
    start="2013-10-28T00:00:00", duration=24 * u.hour, cadence=1 * u.minute, flares=4, seed=None
):
    """
    Generate a synthetic GOES X-ray sensor light curve.

    The X-ray sensor watches the whole Sun in two bands and is the instrument
    that defines flare classes. A quiet Sun sits at the A or B level, and each
    flare rises sharply and decays more slowly, which is what this reproduces.

    Parameters
    ----------
    start : time-like, optional
        The start of the series.
    duration : `~astropy.units.Quantity`, optional
        How long the series runs for.
    cadence : `~astropy.units.Quantity`, optional
        The interval between samples.
    flares : `int`, optional
        How many flares to inject.
    seed : `int`, optional
        Seed for the random number generator.

    Returns
    -------
    `pandas.DataFrame`
        Columns ``xrsa`` (0.5 to 4 angstroms) and ``xrsb`` (1 to 8 angstroms),
        both in watts per square metre, indexed by time.

    Examples
    --------
    >>> from heliox.data._synthetic import make_xray_lightcurve
    >>> curve = make_xray_lightcurve(seed=1)
    >>> list(curve.columns)
    ['xrsa', 'xrsb']
    >>> bool((curve['xrsb'] > curve['xrsa']).all())
    True
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_samples = int(
        u.Quantity(duration, u.s).to_value(u.s) / u.Quantity(cadence, u.s).to_value(u.s)
    )
    times = parse_time(start) + np.arange(n_samples) * u.Quantity(cadence, u.s)
    minutes = np.arange(n_samples) * u.Quantity(cadence, u.s).to_value(u.minute)

    # A slowly varying B-class background.
    background = 1.5e-7 * (
        1
        + 0.3 * np.sin(2 * np.pi * minutes / (minutes[-1] or 1))
        + 0.05 * rng.normal(size=n_samples)
    )
    long_channel = np.clip(background, 1e-9, None)

    for _ in range(flares):
        peak_time = rng.uniform(0.05, 0.95) * minutes[-1]
        # Flare classes are logarithmic; draw a peak between C1 and X1.
        peak_flux = 10 ** rng.uniform(-6, -4)
        rise = rng.uniform(2, 8)
        decay = rise * rng.uniform(3, 8)

        offset = minutes - peak_time
        profile = np.where(
            offset < 0,
            np.exp(offset / rise),
            np.exp(-offset / decay),
        )
        long_channel = long_channel + peak_flux * profile

    # The short channel is harder and so is relatively much weaker when quiet,
    # but brightens far more steeply during a flare.
    ratio = 0.02 + 0.25 * np.clip((np.log10(long_channel) + 7) / 3, 0, 1)
    short_channel = long_channel * ratio
    short_channel = short_channel * (1 + 0.05 * rng.normal(size=n_samples))

    return pd.DataFrame(
        {
            "xrsa": np.clip(short_channel, 1e-10, None),
            "xrsb": np.clip(long_channel, 1e-9, None),
        },
        index=pd.DatetimeIndex(times.datetime, name="time"),
    )


def make_sunspot_series(start="2008-01-01", years=13, seed=None):
    """
    Generate a synthetic monthly sunspot number and radio flux series.

    Follows an eleven year cycle with the characteristic fast rise and slower
    decline, plus month-to-month scatter.

    Parameters
    ----------
    start : time-like, optional
        The start of the series.
    years : `int`, optional
        How many years to cover.
    seed : `int`, optional
        Seed for the random number generator.

    Returns
    -------
    `pandas.DataFrame`
        Columns ``sunspot_number`` and ``f10.7``, indexed by month.

    Examples
    --------
    >>> from heliox.data._synthetic import make_sunspot_series
    >>> series = make_sunspot_series(seed=1)
    >>> bool((series['sunspot_number'] >= 0).all())
    True
    """
    import pandas as pd

    from heliox.sun.models import sunspot_number_to_flux

    rng = np.random.default_rng(seed)
    n_months = years * 12
    times = pd.date_range(parse_time(start).datetime, periods=n_months, freq="MS")

    # A skewed cycle: the rise to maximum takes about four years and the
    # decline about seven, which is the well known asymmetry.
    phase = (np.arange(n_months) / 12.0) % 11.0
    rising = phase < 4.0
    shape = np.where(
        rising,
        np.sin(np.pi * phase / 8.0),
        np.sin(np.pi * (4.0 + (phase - 4.0) * 4.0 / 7.0) / 8.0),
    )
    sunspots = np.clip(120 * shape + rng.normal(scale=12, size=n_months), 0, None)

    return pd.DataFrame(
        {
            "sunspot_number": sunspots,
            "f10.7": sunspot_number_to_flux(sunspots).to_value("sfu"),
        },
        index=pd.DatetimeIndex(times, name="time"),
    )

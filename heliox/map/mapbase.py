"""
The `GenericMap` class: a solar image that knows where it is pointing.

A map bundles three things that always travel together in solar physics: a 2D
array of numbers, the metadata that came with it, and the coordinate frame that
turns pixel indices into positions on the Sun. Once those are tied together,
operations like cropping, rotating and overplotting can all be expressed in
physical coordinates instead of pixel arithmetic.
"""

import copy
import textwrap
import warnings

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from heliox.coordinates import (
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
)
from heliox.sun import constants
from heliox.time import parse_time
from heliox.util.decorators import cached_property_based_on
from heliox.util.exceptions import HelioxMetadataWarning, MapMetaValidationError
from heliox.util.metadata import MetaDict

__all__ = ["GenericMap"]

_NOT_SET = object()


class GenericMap:
    """
    A 2D solar image and the metadata that describes it.

    Parameters
    ----------
    data : `numpy.ndarray`
        The image, indexed as ``[row, column]``.
    header : mapping
        FITS-style metadata. Must contain enough of a World Coordinate System
        for the map to know where it is pointing.
    plot_settings : `dict`, optional
        Keyword arguments handed to `~matplotlib.axes.Axes.imshow` when the map
        is plotted. Sensible defaults are chosen from the metadata.

    Notes
    -----
    Maps are immutable in the sense that every operation returns a new map
    rather than modifying this one. The underlying array is not copied unless
    it has to be, so this is cheap.

    Examples
    --------
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> aia = heliox.map.Map(AIA_171_IMAGE)  # doctest: +SKIP
    >>> aia.instrument                       # doctest: +SKIP
    'AIA'
    """

    #: Keywords a map cannot do without.
    _required_keywords = {"cdelt1", "cdelt2", "crpix1", "crpix2", "crval1", "crval2"}

    def __init__(self, data, header, plot_settings=None, **kwargs):
        self._data = np.asarray(data)
        self._meta = MetaDict(header)

        self._validate_meta()
        self._fix_missing_units()

        self.plot_settings = {
            "cmap": "gray",
            "interpolation": "nearest",
            "origin": "lower",
        }
        self.plot_settings.update(self._default_plot_settings())
        if plot_settings:
            self.plot_settings.update(plot_settings)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _new_instance(self, data=_NOT_SET, meta=_NOT_SET, plot_settings=_NOT_SET):
        """
        Build a map of the same class with some pieces replaced.

        Subclasses inherit this, so an `~heliox.map.sources.AIAMap` stays an
        AIAMap through a crop or a rotation.
        """
        new = object.__new__(type(self))
        new._data = self._data if data is _NOT_SET else np.asarray(data)
        new._meta = MetaDict(self._meta) if meta is _NOT_SET else MetaDict(meta)
        new.plot_settings = (
            copy.deepcopy(self.plot_settings)
            if plot_settings is _NOT_SET
            else dict(plot_settings)
        )
        return new

    def _validate_meta(self):
        """Check that the metadata is usable, and complain clearly if it is not."""
        missing = sorted(key for key in self._required_keywords if key not in self._meta)
        if missing:
            raise MapMetaValidationError(
                "This header is missing the keywords "
                + ", ".join(repr(key.upper()) for key in missing)
                + ", so heliox cannot work out where the image is pointing. "
                "Use heliox.map.make_fitswcs_header to build a valid header."
            )
        if self._data.ndim != 2:
            raise ValueError(
                f"A map holds a 2D image, but this array has {self._data.ndim} dimensions."
            )

    def _fix_missing_units(self):
        """Fill in axis units that the header left out, warning when we guess."""
        for axis in (1, 2):
            key = f"cunit{axis}"
            if key not in self._meta:
                default = "deg" if self._is_heliographic() else "arcsec"
                warnings.warn(
                    f"Missing metadata for {key.upper()}: assuming {default}.",
                    HelioxMetadataWarning,
                    stacklevel=4,
                )
                self._meta[key] = default

    def _is_heliographic(self):
        ctype = str(self._meta.get("ctype1", "")).upper()
        return ctype.startswith(("HGLN", "CRLN"))

    def _default_plot_settings(self):
        """Plot settings inferred from the metadata; subclasses override this."""
        return {}

    # ------------------------------------------------------------------
    # The data itself
    # ------------------------------------------------------------------
    @property
    def data(self):
        """The image, as a `numpy.ndarray` indexed ``[row, column]``."""
        return self._data

    @property
    def meta(self):
        """The metadata, as a case-insensitive `~heliox.util.metadata.MetaDict`."""
        return self._meta

    @property
    def dtype(self):
        """The data type of the image."""
        return self._data.dtype

    @property
    def ndim(self):
        """The number of dimensions, which is always 2 for a map."""
        return self._data.ndim

    @property
    def shape(self):
        """The shape of the image, as ``(rows, columns)``."""
        return self._data.shape

    @property
    def dimensions(self):
        """
        The size of the image in pixels, ordered ``(x, y)``.

        Note that this is the opposite order to `shape`, because it follows the
        FITS axis convention rather than the numpy one.
        """
        return u.Quantity([self._data.shape[1], self._data.shape[0]], u.pix)

    @property
    def unit(self):
        """The physical unit of the pixel values, from ``BUNIT``, or `None`."""
        raw = self._meta.get("bunit")
        if raw is None:
            return None
        try:
            return u.Unit(raw)
        except ValueError:
            try:
                return u.Unit(raw, parse_strict="silent")
            except Exception:  # pragma: no cover - astropy always parses silently
                return None

    @property
    def quantity(self):
        """The image as an `~astropy.units.Quantity`, using `unit`."""
        return u.Quantity(self._data, self.unit)

    def min(self, **kwargs):
        """The smallest pixel value, ignoring NaN."""
        return np.nanmin(self._data, **kwargs)

    def max(self, **kwargs):
        """The largest pixel value, ignoring NaN."""
        return np.nanmax(self._data, **kwargs)

    def mean(self, **kwargs):
        """The mean pixel value, ignoring NaN."""
        return np.nanmean(self._data, **kwargs)

    def std(self, **kwargs):
        """The standard deviation of the pixel values, ignoring NaN."""
        return np.nanstd(self._data, **kwargs)

    # ------------------------------------------------------------------
    # Instrument metadata
    # ------------------------------------------------------------------
    @property
    def instrument(self):
        """The instrument that took the image."""
        return str(self._meta.get("instrume", "")).replace("_", " ")

    @property
    def observatory(self):
        """The observatory or spacecraft the instrument is on."""
        return str(self._meta.get("obsrvtry", self._meta.get("telescop", "")))

    @property
    def detector(self):
        """The detector within the instrument."""
        return str(self._meta.get("detector", ""))

    @property
    def processing_level(self):
        """The calibration level of the data, if the header records one."""
        return self._meta.get("lvl_num")

    @property
    def exposure_time(self):
        """The exposure time, or `None` if the header does not give one."""
        for key in ("exptime", "xposure"):
            if key in self._meta:
                return float(self._meta[key]) * u.s
        return None

    @property
    def waveunit(self):
        """The unit of `wavelength`."""
        raw = self._meta.get("waveunit")
        return u.Unit(raw) if raw else None

    @property
    def wavelength(self):
        """The observing wavelength, or `None`."""
        raw = self._meta.get("wavelnth")
        if raw is None:
            return None
        unit = self.waveunit
        return u.Quantity(raw, unit) if unit else u.Quantity(raw)

    @property
    def measurement(self):
        """What was measured: usually a wavelength, sometimes a data product name."""
        wavelength = self.wavelength
        if wavelength is None:
            return str(self._meta.get("content", ""))
        return wavelength

    @property
    def name(self):
        """A human readable one-line description of the observation."""
        measurement = self.measurement
        if isinstance(measurement, u.Quantity):
            measurement = f"{measurement.value:g} {measurement.unit}"
        parts = [part for part in (self.observatory, self.detector) if part]
        label = " ".join(parts) or self.instrument or "Unknown"
        return f"{label} {measurement} {self.date}".replace("  ", " ").strip()

    @property
    def nickname(self):
        """A short label for the instrument, used in plot titles."""
        return getattr(self, "_nickname", None) or self.detector or self.instrument

    @nickname.setter
    def nickname(self, value):
        self._nickname = value

    # ------------------------------------------------------------------
    # Times
    # ------------------------------------------------------------------
    @property
    def date(self):
        """
        The time the image refers to.

        Prefers the midpoint of the exposure if the header gives one, since
        that is what the coordinates are really valid for, and falls back to
        the start of the exposure.
        """
        for key in ("date-avg", "date_avg", "date-obs", "date_obs", "t_obs"):
            if key in self._meta and self._meta[key]:
                return parse_time(self._meta[key])
        warnings.warn(
            "This header records no observation date; assuming the current time.",
            HelioxMetadataWarning,
            stacklevel=2,
        )
        return parse_time("now")

    @property
    def date_start(self):
        """The start of the exposure, if the header records one."""
        for key in ("date-beg", "date-obs", "date_obs"):
            if key in self._meta and self._meta[key]:
                return parse_time(self._meta[key])
        return None

    @property
    def date_end(self):
        """The end of the exposure, computed from the start and the exposure time."""
        for key in ("date-end", "date_end"):
            if key in self._meta and self._meta[key]:
                return parse_time(self._meta[key])
        start, exposure = self.date_start, self.exposure_time
        if start is not None and exposure is not None:
            return start + exposure
        return None

    # ------------------------------------------------------------------
    # Where the observer was
    # ------------------------------------------------------------------
    @property
    def dsun(self):
        """The distance from the observer to the centre of the Sun."""
        if "dsun_obs" in self._meta:
            return float(self._meta["dsun_obs"]) * u.m
        return get_earth(self.date).radius.to(u.m)

    @property
    def rsun_meters(self):
        """The solar radius assumed by the metadata."""
        if "rsun_ref" in self._meta:
            return float(self._meta["rsun_ref"]) * u.m
        return constants.radius.to(u.m)

    @property
    def rsun_obs(self):
        """The angular radius of the Sun as seen by the observer."""
        if "rsun_obs" in self._meta:
            return float(self._meta["rsun_obs"]) * u.arcsec
        for key in ("solar_r", "radius"):
            if key in self._meta:
                return float(self._meta[key]) * u.arcsec
        return np.arctan(self.rsun_meters / self.dsun).to(u.arcsec)

    @property
    def heliographic_latitude(self):
        """The observer's heliographic latitude, the ``B0`` angle."""
        for key in ("hglt_obs", "crlt_obs", "solar_b0"):
            if key in self._meta:
                return float(self._meta[key]) * u.deg
        return get_earth(self.date).lat.to(u.deg)

    @property
    def heliographic_longitude(self):
        """The observer's Stonyhurst heliographic longitude."""
        if "hgln_obs" in self._meta:
            return float(self._meta["hgln_obs"]) * u.deg
        if "crln_obs" in self._meta:
            return self.observer_coordinate.lon.to(u.deg)
        return 0 * u.deg

    @property
    def carrington_longitude(self):
        """The observer's Carrington longitude, the ``L0`` angle."""
        if "crln_obs" in self._meta:
            return float(self._meta["crln_obs"]) * u.deg
        return (
            self.observer_coordinate.transform_to(
                HeliographicCarrington(obstime=self.date, observer="earth")
            ).lon.to(u.deg)
        )

    @property
    def observer_coordinate(self):
        """
        Where the observer was, as a `~heliox.coordinates.HeliographicStonyhurst`.

        Built from the header keywords if they are present, and from the
        Earth's position at `date` if they are not.
        """
        if "hgln_obs" in self._meta and "hglt_obs" in self._meta:
            return HeliographicStonyhurst(
                float(self._meta["hgln_obs"]) * u.deg,
                float(self._meta["hglt_obs"]) * u.deg,
                self.dsun,
                obstime=self.date,
            )
        if "crln_obs" in self._meta and "crlt_obs" in self._meta:
            carrington = HeliographicCarrington(
                float(self._meta["crln_obs"]) * u.deg,
                float(self._meta["crlt_obs"]) * u.deg,
                self.dsun,
                obstime=self.date,
                observer="self",
            )
            return carrington.transform_to(HeliographicStonyhurst(obstime=self.date))
        return get_earth(self.date)

    # ------------------------------------------------------------------
    # The coordinate system
    # ------------------------------------------------------------------
    @property
    def reference_pixel(self):
        """
        The pixel that `reference_coordinate` refers to, zero-based.

        FITS counts pixels from one and heliox counts from zero, so this is
        ``CRPIX - 1``.
        """
        return u.Quantity(
            [float(self._meta["crpix1"]) - 1, float(self._meta["crpix2"]) - 1], u.pix
        )

    @property
    def reference_coordinate(self):
        """The world coordinate at `reference_pixel`."""
        return SkyCoord(
            float(self._meta["crval1"]) * self.spatial_units[0],
            float(self._meta["crval2"]) * self.spatial_units[1],
            frame=self.coordinate_frame,
        )

    @property
    def spatial_units(self):
        """The units of the two spatial axes."""
        return (u.Unit(self._meta["cunit1"]), u.Unit(self._meta["cunit2"]))

    @property
    def scale(self):
        """
        The plate scale of each axis, as a quantity per pixel.

        The result has ``axis1`` and ``axis2`` attributes as well as being
        indexable, so both ``map.scale[0]`` and ``map.scale.axis1`` work.
        """
        return _PairWithAxes(
            float(self._meta["cdelt1"]) * self.spatial_units[0] / u.pix,
            float(self._meta["cdelt2"]) * self.spatial_units[1] / u.pix,
        )

    @property
    def rotation_matrix(self):
        """
        The 2x2 matrix relating pixel axes to world axes.

        Reads whichever of ``PCi_j``, ``CDi_j`` or ``CROTA2`` the header
        provides, in that order of preference, and falls back to the identity.
        """
        meta = self._meta
        if "pc1_1" in meta:
            return np.array(
                [
                    [float(meta.get("pc1_1", 1)), float(meta.get("pc1_2", 0))],
                    [float(meta.get("pc2_1", 0)), float(meta.get("pc2_2", 1))],
                ]
            )
        if "cd1_1" in meta:
            cd = np.array(
                [
                    [float(meta.get("cd1_1", 0)), float(meta.get("cd1_2", 0))],
                    [float(meta.get("cd2_1", 0)), float(meta.get("cd2_2", 0))],
                ]
            )
            # CD folds the scale in with the rotation, so divide it back out.
            cdelt = np.array([float(meta["cdelt1"]), float(meta["cdelt2"])])
            return cd / cdelt[:, np.newaxis]
        if "crota2" in meta:
            return self._rotation_matrix_from_angle(float(meta["crota2"]) * u.deg)
        return np.identity(2)

    @staticmethod
    def _rotation_matrix_from_angle(angle):
        """A rotation matrix in the FITS ``CROTA2`` convention."""
        cos, sin = np.cos(angle).value, np.sin(angle).value
        return np.array([[cos, -sin], [sin, cos]])

    @property
    def rotation_angle(self):
        """
        The angle of solar north in the image, measured counter-clockwise from up.

        Zero means the image is already aligned with solar north.
        """
        matrix = self.rotation_matrix
        return (np.arctan2(matrix[1, 0], matrix[1, 1]) * u.rad).to(u.deg)

    @property
    def coordinate_system(self):
        """The ``CTYPE`` values of the two axes."""
        return _PairWithAxes(
            str(self._meta.get("ctype1", "HPLN-TAN")),
            str(self._meta.get("ctype2", "HPLT-TAN")),
        )

    @property
    def coordinate_frame(self):
        """
        The coordinate frame the image is expressed in.

        Built from the ``CTYPE`` keywords and the observer's position, so
        coordinates taken from one map can be transformed straight into
        another's frame.
        """
        ctype = self.coordinate_system.axis1.upper()
        if ctype.startswith("HPLN"):
            return Helioprojective(
                obstime=self.date,
                observer=self.observer_coordinate,
                rsun=self.rsun_meters,
            )
        if ctype.startswith("HGLN"):
            return HeliographicStonyhurst(obstime=self.date)
        if ctype.startswith("CRLN"):
            return HeliographicCarrington(
                obstime=self.date, observer=self.observer_coordinate
            )
        raise MapMetaValidationError(
            f"heliox does not know what coordinate frame {ctype!r} describes."
        )

    @cached_property_based_on("_meta")
    def wcs(self):
        """
        The map's World Coordinate System, as an `astropy.wcs.WCS`.

        Cached, and invalidated automatically if the metadata changes.
        """
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [float(self._meta["crpix1"]), float(self._meta["crpix2"])]
        wcs.wcs.cdelt = [float(self._meta["cdelt1"]), float(self._meta["cdelt2"])]
        wcs.wcs.crval = [float(self._meta["crval1"]), float(self._meta["crval2"])]
        wcs.wcs.ctype = [self.coordinate_system.axis1, self.coordinate_system.axis2]
        wcs.wcs.cunit = [str(unit) for unit in self.spatial_units]
        wcs.wcs.pc = self.rotation_matrix
        wcs.array_shape = self._data.shape

        wcs.wcs.dateobs = self.date.utc.isot
        wcs.wcs.aux.rsun_ref = self.rsun_meters.to_value(u.m)

        observer = self.observer_coordinate
        wcs.wcs.aux.hgln_obs = observer.lon.to_value(u.deg)
        wcs.wcs.aux.hglt_obs = observer.lat.to_value(u.deg)
        wcs.wcs.aux.dsun_obs = observer.radius.to_value(u.m)

        return wcs

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------
    def pixel_to_world(self, x, y):
        """
        Convert pixel coordinates into world coordinates.

        Parameters
        ----------
        x, y : `~astropy.units.Quantity`
            Zero-based pixel coordinates, in pixels.

        Returns
        -------
        `~astropy.coordinates.SkyCoord`
        """
        x = u.Quantity(x, u.pix).to_value(u.pix)
        y = u.Quantity(y, u.pix).to_value(u.pix)
        return SkyCoord(self.wcs.pixel_to_world(x, y))

    def world_to_pixel(self, coordinate):
        """
        Convert a world coordinate into pixel coordinates.

        Parameters
        ----------
        coordinate : `~astropy.coordinates.SkyCoord`
            The coordinate to locate. It is transformed into the map's frame
            first, so a coordinate from another map or another observer works.

        Returns
        -------
        `~astropy.units.Quantity`
            A pair of zero-based pixel coordinates, ``(x, y)``.
        """
        x, y = self.wcs.world_to_pixel(coordinate)
        return _PairWithAxes(u.Quantity(x, u.pix), u.Quantity(y, u.pix))

    @property
    def bottom_left_coord(self):
        """The world coordinate of the bottom left corner pixel."""
        return self.pixel_to_world(0 * u.pix, 0 * u.pix)

    @property
    def top_right_coord(self):
        """The world coordinate of the top right corner pixel."""
        return self.pixel_to_world(
            (self._data.shape[1] - 1) * u.pix, (self._data.shape[0] - 1) * u.pix
        )

    @property
    def center(self):
        """The world coordinate of the centre of the image."""
        return self.pixel_to_world(
            (self._data.shape[1] - 1) / 2 * u.pix, (self._data.shape[0] - 1) / 2 * u.pix
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def save(self, filepath, filetype="auto", **kwargs):
        """
        Write the map to a FITS file.

        Parameters
        ----------
        filepath : path-like
            Where to write the file.
        filetype : `str`, optional
            Only ``'auto'`` and ``'fits'`` are supported.
        **kwargs
            Passed to `astropy.io.fits.HDUList.writeto`, most usefully
            ``overwrite=True``.
        """
        from heliox.io._fits import write

        if filetype not in ("auto", "fits"):
            raise ValueError("heliox can only save maps as FITS files.")
        write(filepath, self._data, self._meta, **kwargs)

    def __repr__(self):
        return textwrap.dedent(
            f"""\
            <heliox.map.{type(self).__name__}
            Observatory:         {self.observatory}
            Instrument:          {self.instrument}
            Detector:            {self.detector}
            Measurement:         {self.measurement}
            Observation date:    {self.date.utc.isot}
            Dimensions:          {self._data.shape[1]} x {self._data.shape[0]} pixels
            Scale:               {self.scale.axis1.value:.4f} x {self.scale.axis2.value:.4f} {self.scale.axis1.unit}
            Reference pixel:     {self.reference_pixel.value[0]:.1f}, {self.reference_pixel.value[1]:.1f}
            Coordinate system:   {self.coordinate_system.axis1}, {self.coordinate_system.axis2}
            Data range:          {self.min():.4g} to {self.max():.4g}
            >"""
        )

    def __eq__(self, other):
        if not isinstance(other, GenericMap):
            return NotImplemented
        return np.array_equal(self._data, other._data, equal_nan=True) and dict(
            self._meta
        ) == dict(other._meta)

    def __hash__(self):
        return id(self)


class _PairWithAxes(tuple):
    """
    A two-element tuple whose entries can also be reached as ``axis1``/``axis2``.

    Lets ``map.scale[0]`` and ``map.scale.axis1`` both work, which keeps
    indexing convenient without sacrificing readability at call sites.
    """

    __slots__ = ()

    def __new__(cls, first, second):
        return super().__new__(cls, (first, second))

    @property
    def axis1(self):
        return self[0]

    @property
    def axis2(self):
        return self[1]

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

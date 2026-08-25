Solar coordinates
=================

heliox adds four coordinate frames to :mod:`astropy.coordinates`, so a position
on the Sun can be moved between them, and between them and any celestial frame
astropy knows about, with the ordinary
:meth:`~astropy.coordinates.SkyCoord.transform_to`.

The four frames
---------------

:class:`~heliox.coordinates.HeliographicStonyhurst`
    Longitude and latitude on the Sun, with zero longitude on the meridian
    facing the Earth. The natural frame for saying where something is on the
    Sun right now. Because the zero meridian is defined by the Earth, a fixed
    feature drifts through about 13 degrees of longitude a day.

:class:`~heliox.coordinates.HeliographicCarrington`
    The same, but the grid co-rotates with the Sun, so a long-lived active
    region keeps roughly the same longitude from one rotation to the next.

:class:`~heliox.coordinates.Heliocentric`
    Cartesian coordinates centred on the Sun with the z axis pointing at the
    observer. Mostly an intermediate step, but useful for line-of-sight work.

:class:`~heliox.coordinates.Helioprojective`
    What a telescope actually measures: angles on the sky from the centre of
    the solar disc, conventionally in arcseconds.

Observers
---------

Three of those four frames need to know where the observer is, because the
Sun looks different from different places. Pass a coordinate, or the name of a
solar system body:

.. doctest::

    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from heliox.coordinates import Helioprojective
    >>> disc_centre = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective,
    ...                        obstime='2013-10-28T12:00:00', observer='earth')
    >>> disc_centre.frame.angular_radius.round(1)
    <Quantity 965.5 arcsec>

Transforming
------------

.. doctest::

    >>> from heliox.coordinates import HeliographicStonyhurst
    >>> on_sun = disc_centre.transform_to(
    ...     HeliographicStonyhurst(obstime='2013-10-28T12:00:00')
    ... )
    >>> round(float(on_sun.lat.to_value(u.deg)), 3)
    4.723

That latitude is the familiar ``B0`` angle: the tilt of the Sun's rotation axis
towards the Earth, which oscillates between about -7.25 and +7.25 degrees over
the year.

A two-dimensional helioprojective coordinate is a direction with no distance,
so heliox places it on the solar surface by intersecting the line of sight with
the solar sphere. Lines of sight that miss the Sun come back as NaN:

.. doctest::

    >>> import numpy as np
    >>> off_disc = SkyCoord(2000 * u.arcsec, 0 * u.arcsec, frame=Helioprojective,
    ...                     obstime='2013-10-28T12:00:00', observer='earth')
    >>> bool(np.isnan(off_disc.transform_to(HeliographicStonyhurst(obstime='2013-10-28T12:00:00')).radius))
    True

Between observers
-----------------

Transforming between two helioprojective frames with different observers is how
you overlay images from two spacecraft. The route runs through heliocentric
coordinates, so the parallax between the viewpoints is handled properly:

.. doctest::

    >>> from heliox.coordinates import get_body_heliographic_stonyhurst
    >>> mars = get_body_heliographic_stonyhurst('mars', '2013-10-28T12:00:00')
    >>> from_mars = disc_centre.transform_to(
    ...     Helioprojective(obstime='2013-10-28T12:00:00', observer=mars)
    ... )
    >>> from_mars.Tx.round(1)
    <Longitude -581.1 arcsec>

The centre of the disc as seen from Earth is not the centre of the disc as seen
from Mars, which is exactly the point. In October 2013 Mars was about 91
degrees ahead of the Earth in heliographic longitude, so the sub-Earth point
sits right at the edge of Mars's view.

Ephemeris
---------

:mod:`heliox.sun` has the classical ephemeris quantities, computed
independently of the frames above:

.. doctest::

    >>> from heliox.sun import sun
    >>> sun.B0('2013-10-28T12:00:00').round(3)
    <Quantity 4.723 deg>
    >>> sun.carrington_rotation_number('2013-10-28')  # doctest: +SKIP
    2143.09...

The two routes agree to a fraction of an arcsecond, which is a useful check
that neither has a sign error in it.

Great arcs
----------

Distances on the Sun should follow the surface rather than cutting through it:

.. doctest::

    >>> from heliox.coordinates import GreatArc
    >>> frame = dict(frame=Helioprojective, obstime='2013-10-28', observer='earth')
    >>> a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **frame)
    >>> b = SkyCoord(500 * u.arcsec, 300 * u.arcsec, **frame)
    >>> arc = GreatArc(a, b)
    >>> arc.distance.to('Mm').round(1)
    <Quantity 449.2 Mm>

Differential rotation
---------------------

The Sun does not rotate as a solid body: the equator goes round in about 25
days and the poles take more than 30.
:func:`~heliox.physics.solar_rotate_coordinate` moves a feature to where it
will be later:

.. doctest::

    >>> from heliox.physics import solar_rotate_coordinate
    >>> feature = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst,
    ...                    obstime='2013-10-28T12:00:00')
    >>> later = solar_rotate_coordinate(feature, time='2013-10-29T12:00:00')
    >>> round(float(later.lon.to_value(u.deg)), 1)
    13.3

Thirteen degrees, not fourteen: the rotation rates are sidereal, but Stonyhurst
longitude is measured from the Earth-facing meridian, which is itself moving.

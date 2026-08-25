Working with maps
=================

A :class:`~heliox.map.GenericMap` bundles three things that always travel
together in solar physics: a 2D array of numbers, the metadata that came with
it, and the coordinate frame that turns pixel indices into positions on the
Sun. Once those are tied together, cropping, rotating and overlaying can all be
expressed in physical coordinates instead of pixel arithmetic.

Loading
-------

:func:`~heliox.map.Map` is the front door. Hand it a filename, a glob, a
directory, an ``(array, header)`` pair or a FITS HDU:

.. doctest::

    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> aia = heliox.map.Map(AIA_171_IMAGE)
    >>> type(aia).__name__
    'AIAMap'

Notice that you get an ``AIAMap``, not a plain ``GenericMap``. The factory asks
every registered instrument class whether it recognises the header, so an AIA
file arrives with AIA's colour table and display scaling already chosen.

What a map knows
----------------

.. doctest::

    >>> aia.instrument
    'AIA'
    >>> aia.wavelength
    <Quantity 171. Angstrom>
    >>> aia.date.isot
    '2013-10-28T12:00:00.000'
    >>> aia.data.shape
    (512, 512)

and, more usefully, where it was looking from:

.. doctest::

    >>> aia.observer_coordinate  # doctest: +SKIP
    <HeliographicStonyhurst Coordinate (obstime=2013-10-28T12:00:00.000): (lon, lat, radius) in (deg, deg, m)
        (0., 4.72340426, 1.48624...e+11)>

That observer is what makes the coordinate frame meaningful, and it is why a
coordinate taken from one map can be transformed straight into another's frame
even if the two were taken from different spacecraft.

Pixels and world coordinates
----------------------------

.. doctest::

    >>> import astropy.units as u
    >>> corner = aia.pixel_to_world(0 * u.pix, 0 * u.pix)
    >>> round(float(corner.Tx.to_value(u.arcsec)))
    -1233
    >>> x, y = aia.world_to_pixel(aia.center)
    >>> round(float(x.to_value(u.pix)), 1)
    255.5

Cropping
--------

:meth:`~heliox.map.GenericMap.submap` takes either pixels or world
coordinates. The world-coordinate form is usually what you want, because it
does not change if you reprocess the data at a different resolution:

.. doctest::

    >>> from astropy.coordinates import SkyCoord
    >>> bottom_left = SkyCoord(-500 * u.arcsec, -500 * u.arcsec, frame=aia.coordinate_frame)
    >>> top_right = SkyCoord(500 * u.arcsec, 500 * u.arcsec, frame=aia.coordinate_frame)
    >>> cropped = aia.submap(bottom_left, top_right=top_right)
    >>> cropped.data.shape
    (208, 208)

The requested rectangle is widened to whole pixels, so the result always
contains the region you asked for.

Resampling and binning
----------------------

:meth:`~heliox.map.GenericMap.resample` interpolates onto a new grid, and
:meth:`~heliox.map.GenericMap.superpixel` bins whole pixels together. Use the
second when the total signal matters, because with the default
:func:`numpy.sum` it is conserved exactly:

.. doctest::

    >>> binned = aia.superpixel([4, 4] * u.pix)
    >>> binned.data.shape
    (128, 128)
    >>> bool(abs(binned.data.sum() - aia.data.sum()) < 1)
    True

Both keep the world coordinate system consistent, so a feature does not move:

.. doctest::

    >>> target = SkyCoord(300 * u.arcsec, 200 * u.arcsec, frame=aia.coordinate_frame)
    >>> px = binned.world_to_pixel(target)
    >>> round(float(binned.pixel_to_world(*px).Tx.to_value(u.arcsec)), 6)
    300.0

Rotating
--------

:meth:`~heliox.map.GenericMap.rotate` turns the image and updates the rotation
matrix to match. Called with no angle it lines the image up with solar north:

.. doctest::

    >>> rotated = aia.rotate(30 * u.deg)
    >>> rotated.rotation_angle.round(6)
    <Quantity -30. deg>

Rotating resamples the image, so repeated rotations blur it. If you need
several, combine the angles and rotate once.

Plotting
--------

:meth:`~heliox.map.GenericMap.plot` draws onto world-coordinate axes, so
anything you add afterwards can be positioned physically:

.. code-block:: python

    import matplotlib.pyplot as plt
    import astropy.units as u

    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    aia.draw_limb(axes=axes)
    aia.draw_grid(axes=axes, grid_spacing=15 * u.deg)
    plt.show()

:meth:`~heliox.map.GenericMap.peek` does all of that in one call when you just
want a quick look.

Sequences and composites
------------------------

A :class:`~heliox.map.MapSequence` is an ordered set of maps, usually a time
series of images:

.. doctest::

    >>> from heliox.data.sample import AIA_171_SEQUENCE
    >>> sequence = heliox.map.Map(AIA_171_SEQUENCE, sequence=True)
    >>> len(sequence)
    4
    >>> difference = sequence.running_difference()
    >>> len(difference)
    3

Running differences are the standard way of finding faint moving features:
subtracting consecutive frames removes the static corona and leaves whatever
changed.

A :class:`~heliox.map.CompositeMap` stacks maps on one set of axes, which is
how you put magnetogram contours over an EUV image:

.. doctest::

    >>> from heliox.data.sample import HMI_MAGNETOGRAM
    >>> composite = heliox.map.Map([AIA_171_IMAGE, HMI_MAGNETOGRAM], composite=True)
    >>> composite.set_levels(1, [30, 60], percent=True)
    >>> len(composite)
    2

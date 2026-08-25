API reference
=============

Every public object, grouped by the module it lives in.

Maps
----

heliox.map.mapbase
^^^^^^^^^^^^^^^^^^

The map class itself.

.. automodule:: heliox.map.mapbase
   :members:
   :undoc-members:
   :show-inheritance:

heliox.map.map_factory
^^^^^^^^^^^^^^^^^^^^^^

Building maps from files and arrays.

.. automodule:: heliox.map.map_factory
   :members:
   :undoc-members:
   :show-inheritance:

heliox.map.header_helper
^^^^^^^^^^^^^^^^^^^^^^^^

Building headers for maps you construct yourself.

.. automodule:: heliox.map.header_helper
   :members:
   :undoc-members:
   :show-inheritance:

heliox.map.maputils
^^^^^^^^^^^^^^^^^^^

Geometric questions about a map.

.. automodule:: heliox.map.maputils
   :members:
   :undoc-members:
   :show-inheritance:

heliox.map.mapsequence
^^^^^^^^^^^^^^^^^^^^^^

Ordered collections of maps.

.. automodule:: heliox.map.mapsequence
   :members:
   :undoc-members:
   :show-inheritance:

heliox.map.compositemap
^^^^^^^^^^^^^^^^^^^^^^^

Overlaying maps on one set of axes.

.. automodule:: heliox.map.compositemap
   :members:
   :undoc-members:
   :show-inheritance:

heliox.map.sources.sdo
^^^^^^^^^^^^^^^^^^^^^^

Solar Dynamics Observatory instruments.

.. automodule:: heliox.map.sources.sdo
   :members:
   :undoc-members:
   :show-inheritance:

heliox.map.sources.soho
^^^^^^^^^^^^^^^^^^^^^^^

SOHO instruments.

.. automodule:: heliox.map.sources.soho
   :members:
   :undoc-members:
   :show-inheritance:

Time series
-----------

heliox.timeseries.timeseriesbase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The time series class itself.

.. automodule:: heliox.timeseries.timeseriesbase
   :members:
   :undoc-members:
   :show-inheritance:

heliox.timeseries.timeseries_factory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building time series from files and tables.

.. automodule:: heliox.timeseries.timeseries_factory
   :members:
   :undoc-members:
   :show-inheritance:

heliox.timeseries.metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata for stitched-together series.

.. automodule:: heliox.timeseries.metadata
   :members:
   :undoc-members:
   :show-inheritance:

heliox.timeseries.sources.goes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GOES X-ray sensors and flare classes.

.. automodule:: heliox.timeseries.sources.goes
   :members:
   :undoc-members:
   :show-inheritance:

heliox.timeseries.sources.noaa
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NOAA solar activity indices.

.. automodule:: heliox.timeseries.sources.noaa
   :members:
   :undoc-members:
   :show-inheritance:

Coordinates
-----------

heliox.coordinates.frames
^^^^^^^^^^^^^^^^^^^^^^^^^

The solar coordinate frames.

.. automodule:: heliox.coordinates.frames
   :members:
   :undoc-members:
   :show-inheritance:

heliox.coordinates.transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transformations between them.

.. automodule:: heliox.coordinates.transformations
   :members:
   :undoc-members:
   :show-inheritance:

heliox.coordinates.ephemeris
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Positions of the Earth and other bodies.

.. automodule:: heliox.coordinates.ephemeris
   :members:
   :undoc-members:
   :show-inheritance:

heliox.coordinates.utils
^^^^^^^^^^^^^^^^^^^^^^^^

Great arcs, limbs and rectangles.

.. automodule:: heliox.coordinates.utils
   :members:
   :undoc-members:
   :show-inheritance:

heliox.coordinates.wcs_utils
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bridging frames and FITS World Coordinate Systems.

.. automodule:: heliox.coordinates.wcs_utils
   :members:
   :undoc-members:
   :show-inheritance:

The Sun
-------

heliox.sun.constants
^^^^^^^^^^^^^^^^^^^^

Solar physical constants.

.. automodule:: heliox.sun.constants
   :members:
   :undoc-members:
   :show-inheritance:

heliox.sun.sun
^^^^^^^^^^^^^^

Ephemeris and rotation quantities.

.. automodule:: heliox.sun.sun
   :members:
   :undoc-members:
   :show-inheritance:

heliox.sun.models
^^^^^^^^^^^^^^^^^

Differential rotation, limb darkening and activity.

.. automodule:: heliox.sun.models
   :members:
   :undoc-members:
   :show-inheritance:

Physics
-------

heliox.physics.differential_rotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following features as the Sun rotates.

.. automodule:: heliox.physics.differential_rotation
   :members:
   :undoc-members:
   :show-inheritance:

Images
------

heliox.image.resample
^^^^^^^^^^^^^^^^^^^^^

Changing the pixel grid.

.. automodule:: heliox.image.resample
   :members:
   :undoc-members:
   :show-inheritance:

heliox.image.transform
^^^^^^^^^^^^^^^^^^^^^^

Rotation, scaling and shifting.

.. automodule:: heliox.image.transform
   :members:
   :undoc-members:
   :show-inheritance:

Searching for data
------------------

heliox.net.attrs
^^^^^^^^^^^^^^^^

Search attributes and the query algebra.

.. automodule:: heliox.net.attrs
   :members:
   :undoc-members:
   :show-inheritance:

heliox.net.base_client
^^^^^^^^^^^^^^^^^^^^^^

The client protocol and result tables.

.. automodule:: heliox.net.base_client
   :members:
   :undoc-members:
   :show-inheritance:

heliox.net.fido_factory
^^^^^^^^^^^^^^^^^^^^^^^

The unified search interface.

.. automodule:: heliox.net.fido_factory
   :members:
   :undoc-members:
   :show-inheritance:

heliox.net.sample_client
^^^^^^^^^^^^^^^^^^^^^^^^

The built-in sample data client.

.. automodule:: heliox.net.sample_client
   :members:
   :undoc-members:
   :show-inheritance:

Time
----

heliox.time.time
^^^^^^^^^^^^^^^^

Parsing times.

.. automodule:: heliox.time.time
   :members:
   :undoc-members:
   :show-inheritance:

heliox.time.timerange
^^^^^^^^^^^^^^^^^^^^^

Intervals of time.

.. automodule:: heliox.time.timerange
   :members:
   :undoc-members:
   :show-inheritance:

heliox.time.timeformats
^^^^^^^^^^^^^^^^^^^^^^^

Extra astropy time formats.

.. automodule:: heliox.time.timeformats
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

heliox.visualization.color_tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instrument colour tables.

.. automodule:: heliox.visualization.color_tables
   :members:
   :undoc-members:
   :show-inheritance:

heliox.visualization.drawing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Drawing solar features onto a plot.

.. automodule:: heliox.visualization.drawing
   :members:
   :undoc-members:
   :show-inheritance:

Input and output
----------------

heliox.io._fits
^^^^^^^^^^^^^^^

Reading and writing FITS.

.. automodule:: heliox.io._fits
   :members:
   :undoc-members:
   :show-inheritance:

heliox.io.file_tools
^^^^^^^^^^^^^^^^^^^^

Choosing a reader.

.. automodule:: heliox.io.file_tools
   :members:
   :undoc-members:
   :show-inheritance:

Sample data
-----------

heliox.data.sample
^^^^^^^^^^^^^^^^^^

The sample files.

.. automodule:: heliox.data.sample
   :members:
   :undoc-members:
   :show-inheritance:

heliox.data._synthetic
^^^^^^^^^^^^^^^^^^^^^^

How the sample data is generated.

.. automodule:: heliox.data._synthetic
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

heliox.util.metadata
^^^^^^^^^^^^^^^^^^^^

Case-insensitive metadata mappings.

.. automodule:: heliox.util.metadata
   :members:
   :undoc-members:
   :show-inheritance:

heliox.util.exceptions
^^^^^^^^^^^^^^^^^^^^^^

Exceptions and warnings.

.. automodule:: heliox.util.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

heliox.util.decorators
^^^^^^^^^^^^^^^^^^^^^^

Decorators used across the package.

.. automodule:: heliox.util.decorators
   :members:
   :undoc-members:
   :show-inheritance:

heliox.util.units
^^^^^^^^^^^^^^^^^

Units solar physics needs that astropy lacks.

.. automodule:: heliox.util.units
   :members:
   :undoc-members:
   :show-inheritance:

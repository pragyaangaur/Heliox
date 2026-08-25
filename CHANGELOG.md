# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

The first release. Everything below is new.

### Maps

- `GenericMap`, a 2D solar image tied to its metadata and coordinate frame,
  with cropping, resampling, superpixel binning, rotation and plotting.
- The `Map` factory, which reads filenames, globs, directories, arrays,
  FITS HDUs and existing maps, and selects an instrument-specific class.
- `MapSequence` for time series of images, with running and base differences
  and animation.
- `CompositeMap` for overlaying maps in world coordinates.
- Instrument sources for SDO/AIA, SDO/HMI, SOHO/LASCO and SOHO/EIT.
- `make_fitswcs_header` for turning your own arrays into maps.

### Coordinates

- `HeliographicStonyhurst`, `HeliographicCarrington`, `Heliocentric`,
  `Helioprojective` and `HeliocentricInertial`, registered in astropy's
  transform graph, with the rotation matrices derived from the IAU solar
  rotation elements.
- Transformations between all of them and to any astropy celestial frame,
  including between different observers.
- `GreatArc`, `get_limb_coordinates`, `get_rectangle_coordinates` and
  `solar_angle_equivalency`.
- WCS mappings, so `wcs_to_celestial_frame` understands solar headers.

### The Sun

- Solar physical constants with units, uncertainties and references.
- Ephemeris: `B0`, `L0`, `P`, `earth_distance`, `angular_radius`, obliquity,
  anomalies, and Carrington rotation numbers and times.
- Differential rotation, limb darkening, and the sunspot-number to radio-flux
  relation.

### Time series

- `GenericTimeSeries`, backed by pandas, with a unit for every column.
- The `TimeSeries` factory, reading FITS binary tables and CSV.
- `TimeSeriesMetaData`, for series stitched together from several sources.
- GOES XRS with flare classes, and the NOAA solar activity indices.

### Everything else

- `parse_time`, which reads the time formats solar archives actually use, and
  `TimeRange`.
- Image resampling and affine transforms.
- Differential rotation of coordinates and of whole maps.
- `Fido`, a unified search over pluggable clients, with a search attribute
  algebra.
- Instrument colour tables and drawing helpers for limbs, grids and
  quadrangles.
- Synthetic sample data, generated locally, so nothing needs a network.

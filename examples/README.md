# Examples

Every script here runs on its own with no network access: the sample data is
generated locally the first time it is used.

```bash
python examples/maps/plot_aia_with_overlays.py
```

Each script writes a PNG into `figures/` next to the repository root, and
prints the path it wrote.

| Script | What it shows |
| --- | --- |
| `maps/plot_aia_with_overlays.py` | Plotting a map with the limb and a heliographic grid |
| `maps/composite_magnetogram.py` | Overlaying magnetogram contours on an EUV image |
| `maps/running_difference.py` | Finding what changed between frames of a sequence |
| `maps/crop_and_rotate.py` | Cropping in world coordinates and rotating to solar north |
| `coordinates/observer_parallax.py` | The same feature seen from two different places |
| `coordinates/differential_rotation.py` | How the rotation rate varies with latitude |
| `timeseries/goes_flare.py` | A GOES X-ray light curve and its flare class |
| `timeseries/solar_cycle.py` | The sunspot cycle and the 10.7 cm radio flux |
| `sun/annual_ephemeris.py` | B0, L0 and P through a year |

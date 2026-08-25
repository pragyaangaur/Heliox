"""
Plotting a solar image with the limb and a heliographic grid
============================================================

The point of plotting a map onto world-coordinate axes is that everything you
add afterwards can be positioned physically. The limb and the heliographic
grid here are not drawn in pixels: they are computed from the observer's
position and projected through the map's own WCS, so they would land correctly
even if the image were rotated or cropped.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

import heliox.map  # noqa: E402
from heliox.data.sample import AIA_171_IMAGE  # noqa: E402

aia = heliox.map.Map(AIA_171_IMAGE)

figure = plt.figure(figsize=(8, 8))
axes = figure.add_subplot(projection=aia.wcs)

aia.plot(axes=axes, clip_interval=[1, 99.9] * u.percent)
aia.draw_limb(axes=axes, color="white", linestyle="--")
aia.draw_grid(axes=axes, grid_spacing=15 * u.deg, color="white", alpha=0.4)

# An active region, outlined in world coordinates rather than pixels.
corner = SkyCoord(-600 * u.arcsec, -200 * u.arcsec, frame=aia.coordinate_frame)
aia.draw_quadrangle(corner, axes=axes, width=400 * u.arcsec, height=400 * u.arcsec, color="cyan")

print(f"{aia.nickname} at {aia.date.isot}")
print(f"observer {aia.dsun.to(u.AU):.4f} from the Sun, B0 = {aia.heliographic_latitude:.3f}")
print(f"the disc subtends {aia.rsun_obs:.1f}")

save(figure, "aia_with_overlays.png")

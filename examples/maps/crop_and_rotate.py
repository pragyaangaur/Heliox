"""
Cropping in world coordinates and rotating to solar north
=========================================================

Both operations keep the coordinate metadata consistent, which is the whole
reason to do them through a map rather than on the array directly: a feature
picked out by its position on the Sun stays at that position afterwards.
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

# Pretend the telescope was mounted 25 degrees off solar north.
tilted = aia.rotate(25 * u.deg, order=3, missing=0)
print(f"tilted map north is at {tilted.rotation_angle:.1f}")

# Rotating with no angle puts solar north back at the top.
upright = tilted.rotate(order=3, missing=0)
print(f"after correction north is at {upright.rotation_angle:.6f}")

# Crop around an active region, in arcseconds from disc centre.
bottom_left = SkyCoord(-700 * u.arcsec, -300 * u.arcsec, frame=upright.coordinate_frame)
cropped = upright.submap(bottom_left, width=600 * u.arcsec, height=600 * u.arcsec)
print(f"cropped to {cropped.data.shape[1]} x {cropped.data.shape[0]} pixels")

# The same physical point, located in all three maps.
target = SkyCoord(-400 * u.arcsec, 0 * u.arcsec, frame=aia.coordinate_frame)
for name, each in (("original", aia), ("tilted", tilted), ("cropped", cropped)):
    x, y = each.world_to_pixel(target)
    recovered = each.pixel_to_world(x, y)
    print(
        f"{name:>9}: pixel ({x.value:7.2f}, {y.value:7.2f}) "
        f"-> {recovered.Tx:.3f}, {recovered.Ty:.3f}"
    )

figure = plt.figure(figsize=(13, 4.5))
for index, (title, each) in enumerate(
    (("original", aia), ("tilted 25 deg", tilted), ("corrected and cropped", cropped))
):
    axes = figure.add_subplot(1, 3, index + 1, projection=each.wcs)
    each.plot(axes=axes, annotate=False)
    each.draw_limb(axes=axes)
    axes.set_title(title)

save(figure, "crop_and_rotate.png")

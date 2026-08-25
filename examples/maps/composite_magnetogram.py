"""
Overlaying magnetogram contours on an EUV image
===============================================

Coronal loops are anchored in the photospheric magnetic field, so the natural
way to look at an active region is to put the magnetogram's contours on top of
the EUV image. The two instruments have different plate scales and slightly
different pointing, which is exactly why the overlay is done in world
coordinates rather than by lining up pixels.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

import heliox.map  # noqa: E402
from heliox.data.sample import AIA_171_IMAGE, HMI_MAGNETOGRAM  # noqa: E402

aia = heliox.map.Map(AIA_171_IMAGE)
hmi = heliox.map.Map(HMI_MAGNETOGRAM)

composite = heliox.map.CompositeMap(aia, hmi)
# Positive and negative field, at the same absolute levels.
composite.set_levels(1, [-1500, -750, 750, 1500])

figure = plt.figure(figsize=(8, 8))
axes = figure.add_subplot(projection=aia.wcs)
composite.plot(axes=axes)
aia.draw_limb(axes=axes)

print(f"{aia.nickname} scale: {aia.scale.axis1:.3f}")
print(f"{hmi.nickname} scale: {hmi.scale.axis1:.3f}")
print(f"field range: {hmi.min():.0f} to {hmi.max():.0f} gauss")

save(figure, "composite_magnetogram.png")

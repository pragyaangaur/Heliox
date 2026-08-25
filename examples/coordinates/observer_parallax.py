"""
The same feature seen from two different places
===============================================

Helioprojective coordinates are angles measured from a particular vantage
point, so the same point on the Sun has different coordinates for every
observer. Converting between them is what makes it possible to compare
observations from two spacecraft, and it is the step that is easy to get
silently wrong, because getting the axes wrong is self-inverse and so survives
a round trip.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

from heliox.coordinates import (  # noqa: E402
    HeliographicStonyhurst,
    Helioprojective,
    get_body_heliographic_stonyhurst,
    get_earth,
)

OBSTIME = "2013-10-28T12:00:00"

earth = get_earth(OBSTIME)
mars = get_body_heliographic_stonyhurst("mars", OBSTIME)

print(f"Earth: lon {earth.lon:.2f}, lat {earth.lat:.2f}, {earth.radius.to(u.AU):.3f}")
print(f"Mars:  lon {mars.lon:.2f}, lat {mars.lat:.2f}, {mars.radius.to(u.AU):.3f}")
separation = abs((mars.lon - earth.lon).to_value(u.deg))
print(f"separated by {separation:.1f} degrees of heliographic longitude")

# A ring of features spread around the solar equator.
longitudes = np.arange(-180, 180, 15) * u.deg
features = SkyCoord(
    longitudes,
    np.zeros(longitudes.size) * u.deg,
    frame=HeliographicStonyhurst(obstime=OBSTIME),
).frame.make_3d()
features = SkyCoord(features)

views = {
    "Earth": Helioprojective(obstime=OBSTIME, observer=earth),
    "Mars": Helioprojective(obstime=OBSTIME, observer=mars),
}

figure, axes_pair = plt.subplots(1, 2, figsize=(11, 5.5))

for axes, (name, frame) in zip(axes_pair, views.items(), strict=True):
    projected = features.transform_to(frame)
    visible = projected.frame.is_visible()

    limb = frame.angular_radius.to_value(u.arcsec)
    axes.add_patch(plt.Circle((0, 0), limb, color="0.9", zorder=0))

    tx = projected.Tx.to_value(u.arcsec)
    ty = projected.Ty.to_value(u.arcsec)
    axes.scatter(tx[visible], ty[visible], c="tab:red", label="visible", zorder=2)
    axes.scatter(
        tx[~visible],
        ty[~visible],
        facecolors="none",
        edgecolors="tab:blue",
        label="behind the Sun",
        zorder=2,
    )

    axes.set_aspect("equal")
    axes.set_xlim(-1.4 * limb, 1.4 * limb)
    axes.set_ylim(-1.4 * limb, 1.4 * limb)
    axes.set_title(f"as seen from {name}")
    axes.set_xlabel("solar-X (arcsec)")
    axes.set_ylabel("solar-Y (arcsec)")
    axes.legend(loc="upper right", fontsize="small")

    print(f"{name}: {int(visible.sum())} of {len(longitudes)} features visible")

figure.suptitle("The same 24 points on the solar equator, from two viewpoints")
save(figure, "observer_parallax.png")

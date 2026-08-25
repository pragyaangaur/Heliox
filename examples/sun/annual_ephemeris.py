"""
B0, L0 and P through a year
===========================

Three angles describe how the Sun is presented to an observer on Earth, and all
three come out of the tilt of the solar rotation axis relative to the ecliptic:

``B0``
    The heliographic latitude of the disc centre. The Earth sees the solar
    north pole tipped towards it in September and away in March, so this
    oscillates between about -7.25 and +7.25 degrees.

``L0``
    The Carrington longitude of the disc centre, running from 360 down to 0 over
    each 27.2753 day synodic rotation.

``P``
    The position angle of solar north on the sky. It is dominated by the angle
    between the ecliptic and the celestial equator, so it is largest near the
    equinoxes and passes through zero near the solstices.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

from heliox.sun import sun  # noqa: E402
from heliox.time import parse_time  # noqa: E402

start = parse_time("2013-01-01T00:00:00")
days = np.arange(0, 366)
times = start + days * u.day

b0 = sun.B0(times).to_value(u.deg)
l0 = sun.L0(times).to_value(u.deg)
p = sun.P(times).to_value(u.deg)
distance = sun.earth_distance(times).to_value(u.AU)
radius = sun.angular_radius(times).to_value(u.arcsec)

print(f"B0 runs from {b0.min():+.2f} to {b0.max():+.2f} degrees")
print(f"   maximum on {times[np.argmax(b0)].isot[:10]}")
print(f"P  runs from {p.min():+.2f} to {p.max():+.2f} degrees")
print(f"Earth-Sun distance: {distance.min():.4f} to {distance.max():.4f} AU")
print(f"   perihelion on {times[np.argmin(distance)].isot[:10]}")
print(f"angular radius: {radius.min():.1f} to {radius.max():.1f} arcsec")

rotations = np.flatnonzero(np.diff(l0) > 0) + 1
print(f"{len(rotations)} Carrington rotations begin during the year")
print(f"   first on {times[rotations[0]].isot[:10]}")

figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(days, b0, color="tab:blue")
axes[0].axhline(0, color="0.8", linewidth=0.8)
axes[0].set_ylabel("B0 (degrees)")
axes[0].set_title("Solar ephemeris through 2013")

axes[1].plot(days, l0, color="tab:green")
axes[1].set_ylabel("L0 (degrees)")

axes[2].plot(days, p, color="tab:orange")
axes[2].axhline(0, color="0.8", linewidth=0.8)
axes[2].set_ylabel("P (degrees)")

axes[3].plot(days, radius, color="tab:red")
axes[3].set_ylabel("angular radius (arcsec)")
axes[3].set_xlabel("day of year")

for each in axes:
    each.grid(alpha=0.3)

figure.tight_layout()
save(figure, "annual_ephemeris.png")

"""
How the solar rotation rate varies with latitude
================================================

The Sun is a ball of gas, not a solid body, so the equator completes a rotation
in about 25 days while material near the poles takes more than 30. The
consequence is that a row of features laid out along a meridian shears apart
over a couple of weeks, which is what the second panel shows.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

from heliox.coordinates import HeliographicStonyhurst  # noqa: E402
from heliox.physics import solar_rotate_coordinate  # noqa: E402
from heliox.sun.models import DIFFERENTIAL_ROTATION_MODELS, differential_rotation  # noqa: E402

START = Time("2013-10-28T12:00:00")

figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

# --- The rotation laws themselves -----------------------------------------
latitudes = np.linspace(-75, 75, 200) * u.deg
for name in DIFFERENTIAL_ROTATION_MODELS:
    rate = differential_rotation(1 * u.day, latitudes, model=name)
    left.plot(latitudes.to_value(u.deg), rate.to_value(u.deg), label=name)

left.set_xlabel("heliographic latitude (degrees)")
left.set_ylabel("sidereal rotation (degrees per day)")
left.set_title("Rotation rate against latitude")
left.legend(fontsize="small")
left.grid(alpha=0.3)

for name in DIFFERENTIAL_ROTATION_MODELS:
    equator = differential_rotation(1 * u.day, 0 * u.deg, model=name)
    period = 360 * u.deg / equator
    print(f"{name:>10}: equatorial sidereal period {period.value:.2f} days")

# --- What that does to a line of features ----------------------------------
# Longitudes wrap at 180 degrees, so plotting them raw would show a jump rather
# than the shear. Measuring each feature against the equatorial one keeps the
# numbers small and is what the shear actually means.
feature_latitudes = np.arange(-60, 61, 15) * u.deg
features = SkyCoord(
    np.zeros(feature_latitudes.size) * u.deg,
    feature_latitudes,
    frame=HeliographicStonyhurst(obstime=START),
)
equator = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst(obstime=START))

for days in (0, 5, 10, 15):
    when = START + days * u.day
    rotated = solar_rotate_coordinate(features, time=when)
    reference = solar_rotate_coordinate(equator, time=when)
    lag = ((rotated.lon - reference.lon).to_value(u.deg) + 180) % 360 - 180
    right.plot(lag, feature_latitudes.to_value(u.deg), marker="o", label=f"+{days} days")

right.set_xlabel("longitude relative to the equator (degrees)")
right.set_ylabel("heliographic latitude (degrees)")
right.set_title("A meridian of features, sheared by rotation")
right.legend(fontsize="small")
right.grid(alpha=0.3)

when = START + 15 * u.day
lag = (
    solar_rotate_coordinate(features, time=when).lon
    - solar_rotate_coordinate(equator, time=when).lon
).to_value(u.deg)
lag = (lag + 180) % 360 - 180
print(f"after 15 days the poles lag the equator by {abs(lag).max():.1f} degrees")

figure.tight_layout()
save(figure, "differential_rotation.png")

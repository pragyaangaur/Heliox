"""
The sunspot cycle and the 10.7 cm radio flux
============================================

The sunspot number is the oldest continuous measurement in astronomy, and the
10.7 cm radio flux has tracked it closely since 1947. A single month is far too
noisy to see the cycle in, which is why the number everyone quotes is a
thirteen month running mean.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

import heliox.timeseries  # noqa: E402
from heliox.data.sample import NOAA_INDICES_TIMESERIES  # noqa: E402

noaa = heliox.timeseries.TimeSeries(NOAA_INDICES_TIMESERIES)
smoothed = noaa.smooth(13)

print(f"{len(noaa)} monthly samples from {noaa.time_range.start.isot[:7]}")
print(f"solar maximum at {noaa.solar_maximum.isot[:7]}")
print(f"peak monthly sunspot number: {noaa.data['sunspot_number'].max():.0f}")
print(f"peak smoothed sunspot number: {smoothed.data['sunspot_number'].max():.0f}")
# The overall spread is dominated by the cycle itself, so the honest measure of
# what smoothing removes is the scatter about the smoothed curve.
residual = noaa.data["sunspot_number"] - smoothed.data["sunspot_number"]
print(f"scatter about the 13 month mean: {residual.std():.1f} sunspots")

figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

top.plot(noaa.index, noaa.data["sunspot_number"], color="0.7", label="monthly")
top.plot(
    smoothed.index,
    smoothed.data["sunspot_number"],
    color="tab:blue",
    linewidth=2,
    label="13 month mean",
)
top.axvline(noaa.solar_maximum.datetime, color="tab:red", linestyle=":", linewidth=1)
top.set_ylabel("sunspot number")
top.set_title("The solar cycle")
top.legend(fontsize="small")
top.grid(alpha=0.3)

bottom.plot(noaa.index, noaa.data["f10.7"], color="0.7", label="monthly")
bottom.plot(
    smoothed.index,
    smoothed.data["f10.7"],
    color="tab:orange",
    linewidth=2,
    label="13 month mean",
)
bottom.set_ylabel(f"F10.7 ({noaa.units['f10.7']})")
bottom.set_xlabel("year")
bottom.legend(fontsize="small")
bottom.grid(alpha=0.3)

correlation = noaa.data["sunspot_number"].corr(noaa.data["f10.7"])
print(f"sunspot number and F10.7 correlate at r = {correlation:.3f}")

figure.tight_layout()
save(figure, "solar_cycle.png")

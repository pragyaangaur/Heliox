"""
A GOES X-ray light curve and its flare class
============================================

The GOES X-ray sensor watches the whole Sun in two bands and is the instrument
that defines flare classes. The scale is logarithmic because the flux spans
five decades between a quiet Sun and a large flare, which is why the plot uses
a log axis with the class boundaries marked: on a linear axis everything below
the largest peak is a flat line at zero.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

import heliox.timeseries  # noqa: E402
from heliox.data.sample import GOES_XRS_TIMESERIES  # noqa: E402
from heliox.timeseries.sources.goes import flare_class  # noqa: E402

goes = heliox.timeseries.TimeSeries(GOES_XRS_TIMESERIES)

print(f"{goes.observatory} {goes.instrument}")
print(f"{len(goes)} samples over {goes.time_range.hours:.1f}")
print(f"largest flare: {goes.flare_class} at {goes.peak_time.isot}")

# Find every excursion above the C threshold, and report its class.
long_channel = goes.quantity("xrsb")
above_c = long_channel > 1e-6 * u.W / u.m**2
edges = np.diff(above_c.astype(int))
starts = np.flatnonzero(edges == 1) + 1
ends = np.flatnonzero(edges == -1) + 1
if above_c[0]:
    starts = np.r_[0, starts]
if above_c[-1]:
    ends = np.r_[ends, len(above_c)]

print(f"\n{len(starts)} events above the C threshold:")
for start, end in zip(starts, ends, strict=True):
    window = long_channel[start:end]
    peak = window.max()
    when = goes.time[start + int(np.argmax(window))]
    print(f"  {flare_class(peak):>6} peaking at {when.isot[11:19]}")

figure = plt.figure(figsize=(11, 5))
axes = figure.add_subplot()
goes.plot(axes=axes)

# Mark the largest peak.
axes.axvline(goes.peak_time.datetime, color="tab:red", linestyle=":", linewidth=1)
axes.annotate(
    goes.flare_class,
    xy=(goes.peak_time.datetime, goes.peak_flux.to_value(u.W / u.m**2)),
    xytext=(10, 6),
    textcoords="offset points",
    color="tab:red",
)

save(figure, "goes_flare.png")

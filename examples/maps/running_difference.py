"""
Finding what changed between frames
===================================

The corona is bright and mostly static, which hides the faint moving features
that are usually the interesting part. Subtracting consecutive frames removes
everything that did not change and leaves the motion behind, which is how
coronal waves and erupting loops are normally found.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import save  # noqa: E402

import heliox.map  # noqa: E402
from heliox.data.sample import AIA_171_SEQUENCE  # noqa: E402

sequence = heliox.map.Map(AIA_171_SEQUENCE, sequence=True)
difference = sequence.running_difference()

print(f"{len(sequence)} frames spanning {sequence.time_range.minutes:.0f}")
print(f"{len(difference)} differences")

figure = plt.figure(figsize=(12, 6))

for index, (original, changed) in enumerate(zip(sequence[1:], difference, strict=True)):
    axes = figure.add_subplot(2, len(difference), index + 1, projection=original.wcs)
    original.plot(axes=axes, annotate=False)
    axes.set_title(original.date.isot[11:19], fontsize="small")
    axes.coords[0].set_ticklabel_visible(False)
    axes.coords[1].set_ticklabel_visible(False)

    axes = figure.add_subplot(
        2, len(difference), len(difference) + index + 1, projection=changed.wcs
    )
    # A symmetric scale, so that brightening and dimming read the same.
    limit = np.nanpercentile(np.abs(changed.data), 99.5)
    changed.plot(axes=axes, annotate=False, cmap="RdBu_r", norm=None, vmin=-limit, vmax=limit)
    axes.set_title("difference", fontsize="small")
    axes.coords[0].set_ticklabel_visible(False)
    axes.coords[1].set_ticklabel_visible(False)

figure.suptitle("AIA 171 running difference")
save(figure, "running_difference.png")

"""Measure how far the Rust differential rotation is from sunpy, and print markdown tables.

usage: python accuracy_diffrot.py

It reuses the scenarios of `tests/test_diffrot.py` (the real AIA 171 sample image, set the
directory with SUNPY_SAMPLEDIR). The first table is the difference between the source pixel
coordinates of the Rust inverse map and sunpy's `_warp_sun_coordinates` over the whole pixel grid
of the 1024 x 1024 map, in pixels. The second is the difference between the rotated images (float32
as in the sample file, and float64 for the last rows) together with the number of pixels whose NaN
status differs. The relative difference is taken against max(|value|, 1). Run it under the memory watchdog like the other scripts.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import astropy.units as u  # noqa: E402
import sunpy.data.sample as sample  # noqa: E402
import sunpy.map as sunpy_map  # noqa: E402
import sunpy.physics.differential_rotation as dr  # noqa: E402
import test_diffrot as t  # noqa: E402

from heliox_rs import diffrot as rs  # noqa: E402


def main():
    aia = sunpy_map.Map(sample.AIA_171_IMAGE)
    print("Source pixel coordinates over the whole 1024 x 1024 grid (pixels)\n")
    print("| Rotation | Finite pixels | NaN masks differ | Median | 99th percentile | Maximum |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    xy = t.grid(1024)
    for name, (_, rotation, model) in t.SCENARIOS.items():
        obs = t.new_observer(aia, rotation(aia))
        expected = dr._warp_sun_coordinates(xy, aia, obs, **model)
        actual = rs.warp_sun_coordinates(xy, aia, obs, **model)
        masks = int((np.isnan(expected[:, 0]) != np.isnan(actual[:, 0])).sum())
        ok = ~np.isnan(expected[:, 0])
        d = np.abs(actual[ok] - expected[ok]).max(axis=1)
        print(f"| {name} | {ok.sum()} | {masks} | {np.median(d):.1e} | {np.percentile(d, 99):.1e} | {d.max():.1e} |")

    print("\nRotated images\n")
    print("| Case | dtype | NaN status differs | Max abs difference | Max relative difference |")
    print("| --- | --- | ---: | ---: | ---: |")
    aia64 = sunpy_map.Map(aia.data.astype(np.float64), aia.meta)
    cases = list(t.IMAGE_SCENARIOS.items()) + [("time +1h, float64", None), ("time -2d, float64", None)]
    for name, spec in cases:
        if spec is None:
            smap = aia64
            kwargs = dict(time=(1 * u.hr if "+1h" in name else -2 * u.day))
            model = {}
        else:
            build, rotation, model = spec
            smap = build(aia)
            kwargs = rotation(smap)
        e = dr.differential_rotate(smap, **kwargs, **model)
        a = rs.differential_rotate(smap, **kwargs, **model)
        x, y = np.asarray(e.data, dtype=float), np.asarray(a.data, dtype=float)
        masks = int((np.isnan(x) != np.isnan(y)).sum())
        ok = ~np.isnan(x)
        d = np.abs(y[ok] - x[ok])
        rel = (d / np.maximum(np.abs(x[ok]), 1.0)).max()
        print(f"| {name} | {e.data.dtype} | {masks} | {d.max():.2e} | {rel:.1e} |")


if __name__ == "__main__":
    main()

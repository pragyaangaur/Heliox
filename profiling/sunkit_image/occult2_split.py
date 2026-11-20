"""Where does occult2 spend its wall time? Wrapper timers around the stages, plus loop counts.

Example:
  python occult2_split.py --sample-dir DATA/sample --repeats 3
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import Data  # noqa: E402

from sunkit_image import trace  # noqa: E402

ACC = {}


def wrap(name):
    orig = getattr(trace, name)

    def w(*a, **k):
        t = time.perf_counter()
        r = orig(*a, **k)
        e = ACC.setdefault(name, [0.0, 0])
        e[0] += time.perf_counter() - t
        e[1] += 1
        return r

    setattr(trace, name, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--ski-data")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    d = Data(a.sample_dir, a.ski_data)
    for n in ("bandpass_filter", "_curvature_radius", "_initial_direction_finding", "_erase_loop_in_image", "_loop_add"):
        wrap(n)
    variants = {
        "docs params (nsm1=3 rmin=30 lmin=25 ngap=0 q2=3)": dict(nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0),
        "trace-heavy (rmin=15 lmin=10 ngap=5 q2=1.2)": dict(nsm1=3, rmin=15, lmin=10, nstruc=1000, ngap=5, qthresh1=0.0, qthresh2=1.2),
        "wide filter (nsm1=9 rmin=30 lmin=25 q2=3)": dict(nsm1=9, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0),
    }
    imgs = {"aia_171_cutout 705x769": d.aia_cutout().data, "AIA171 1024x1024": d.aia1k().data}
    # occult2 is wrapped by accept_array_or_map; call through trace.occult2 so patched helpers are used
    for iname, img in imgs.items():
        for vname, kw in variants.items():
            walls, splits = [], []
            for _ in range(a.repeats):
                ACC.clear()
                t = time.perf_counter()
                loops = trace.occult2(img, **kw)
                walls.append(time.perf_counter() - t)
                splits.append({k: tuple(v) for k, v in ACC.items()})
            s = splits[len(splits) // 2]
            w = statistics.median(walls)
            bp = s["bandpass_filter"][0]
            cr, ncr = s["_curvature_radius"]
            print(f"\n{iname} | {vname}\n  wall median {w:.3f}s  loops returned {len(loops)}  total loop pts {sum(len(l) for l in loops)}")
            print(f"  bandpass_filter (2 python-loop smooths): {bp:.3f}s = {100*bp/w:.0f}%")
            print(f"  _curvature_radius: {ncr} calls, {cr:.3f}s = {100*cr/w:.0f}%  ({1e6*cr/max(ncr,1):.1f} us/call)")
            idf = s.get("_initial_direction_finding", (0, 0))
            er = s.get("_erase_loop_in_image", (0, 0))
            la = s.get("_loop_add", (0, 0))
            print(f"  _initial_direction_finding: {idf[1]} calls {idf[0]:.3f}s; _erase_loop_in_image: {er[1]} calls {er[0]:.3f}s; _loop_add: {la[1]} calls {la[0]:.3f}s")
            rest = w - bp - cr - idf[0] - er[0] - la[0]
            print(f"  everything else in occult2 body (residual.max / np.where full-image per structure, np.max(zl[..]) per step, length loop): {rest:.3f}s = {100*rest/w:.0f}%")


if __name__ == "__main__":
    main()

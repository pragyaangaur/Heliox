"""Run the differential rotation benchmarks under the memory watchdog and print markdown tables.

usage: python run_diffrot_all.py --python VENV_PYTHON --memwatch PATH_TO_memwatch.py --data SUNPY_SAMPLEDIR
                                 [--limit-gb 4] [--sizes 512 1024]

Run it with any Python. It starts one watched process at a time for every implementation, case and
size, so each peak memory figure belongs to one implementation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

IMPLEMENTATIONS = [
    ("sunpy original", "original", {}),
    ("Rust, 1 thread", "rust", {"RAYON_NUM_THREADS": "1"}),
    ("Rust, all cores", "rust", {}),
]
REPEATS = {"original": 7, "rust": 21}
CASES = [("differential_rotate", "rotate"), ("inverse map only", "inverse")]


def run(args, cmd, env):
    full = [sys.executable, args.memwatch, "--limit-gb", args.limit_gb, "--", args.python, str(HERE / "bench_diffrot.py"), *cmd]
    p = subprocess.run(full, capture_output=True, text=True, env={**os.environ, "SUNPY_SAMPLEDIR": args.data, **env})
    m = re.search(r"peak RSS ([\d.]+) GB", p.stderr)
    killed = "KILLED" in p.stderr
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        res = None
        print(p.stderr[-2000:], file=sys.stderr)
    return res, (m.group(1) if m else None), killed


def fmt_time(t):
    return f"{t * 1e3:.2f} ms" if t < 1 else f"{t:.3f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--memwatch", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit-gb", default="4")
    ap.add_argument("--sizes", type=int, nargs="*", default=[512, 1024])
    a = ap.parse_args()

    base = {}
    for size in a.sizes:
        res, rss, _ = run(a, ["baseline", "rotate", str(size)], {})
        base[size] = res["peak_rss_gb"] if res else None
    rows = []
    for size in a.sizes:
        for title, kind in CASES:
            for label, impl, env in IMPLEMENTATIONS:
                res, rss, killed = run(a, [impl, kind, str(size), str(REPEATS[impl])], env)
                rows.append((size, title, label, impl, res, rss, killed))
                print(rows[-1], file=sys.stderr, flush=True)
    print("| Map | Case | Implementation | Median time | Speedup vs original | Peak RSS (GB) | NaN pixels | Checksum relative difference |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for size, title, label, impl, res, rss, killed in rows:
        if killed or res is None:
            print(f"| {size}x{size} | {title} | {label} | killed at the memory limit | | >{rss} | | |")
            continue
        ref = next(r[4] for r in rows if r[0] == size and r[1] == title and r[3] == "original")
        speed = "1x" if impl == "original" else f"{ref['median_s'] / res['median_s']:.1f}x"
        rel = "reference" if impl == "original" else f"{abs(res['checksum'] - ref['checksum']) / abs(ref['checksum']):.1e}"
        print(f"| {size}x{size} | {title} | {label} | {fmt_time(res['median_s'])} | {speed} | {res['peak_rss_gb']:.2f} | {res['nan_count']} | {rel} |")
    print("\nPeak RSS of a process that only loads the map and imports sunpy, GB:", {k: v for k, v in base.items()})


if __name__ == "__main__":
    main()

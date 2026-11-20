"""Run every benchmark combination under the memory watchdog and print a markdown table.

usage: python run_all.py --python VENV_PYTHON --memwatch PATH_TO_memwatch.py [--limit-gb 4]

Run it with any Python. It starts one watched process at a time.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (label, impl, field, repeats, environment)
CASES = [
    ("baseline (load data only)", "baseline", "real", 1, {}),
    ("sunkit-image original", "original", "real", 3, {}),
    ("NumPy sliding_window_view", "numpy", "real", 5, {}),
    ("Rust, 1 thread", "rust", "real", 9, {"RAYON_NUM_THREADS": "1"}),
    ("Rust, all cores", "rust", "real", 9, {}),
    ("baseline (load data only)", "baseline", "random1024", 1, {}),
    ("sunkit-image original", "original", "random1024", 3, {}),
    ("Rust, 1 thread", "rust", "random1024", 9, {"RAYON_NUM_THREADS": "1"}),
    ("Rust, all cores", "rust", "random1024", 9, {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--memwatch", required=True)
    ap.add_argument("--limit-gb", default="4")
    a = ap.parse_args()
    rows = []
    for label, impl, field, repeats, env in CASES:
        cmd = [sys.executable, a.memwatch, "--limit-gb", a.limit_gb, "--", a.python, str(HERE / "bench_asda.py"), impl, field, str(repeats)]
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
        m = re.search(r"peak RSS ([\d.]+) GB", p.stderr)
        killed = "KILLED" in p.stderr
        try:
            res = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            res = None
        rows.append((field, label, res, m.group(1) if m else None, killed))
        print(rows[-1], file=sys.stderr, flush=True)
    base = {f: res for f, label, res, _, _ in rows if label.startswith("sunkit")}
    print("| Field | Implementation | Median time (s) | Speedup vs original | Peak RSS, memwatch (GB) | Peak RSS, in process (GB) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for field, label, res, rss, killed in rows:
        name = {"real": "real 512x1024", "random1024": "random 1024x1024"}[field]
        if label.startswith("baseline"):
            print(f"| {name} | {label} | | | {rss} | {res['maxrss_gb']:.2f} |")
        elif killed or res is None:
            print(f"| {name} | {label} | killed at the memory limit | | >{a.limit_gb} | |")
        else:
            ref = base.get(field)
            sp = f"{ref['median'] / res['median']:.1f}x" if ref and not label.startswith("sunkit") else ""
            print(f"| {name} | {label} | {res['median']:.4g} | {sp} | {rss} | {res['maxrss_gb']:.2f} |")


if __name__ == "__main__":
    main()

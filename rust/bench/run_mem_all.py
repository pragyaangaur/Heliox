"""Run the MEM benchmarks under the memory watchdog and print markdown tables.

usage: python run_mem_all.py --python VENV_PYTHON --memwatch PATH_TO_memwatch.py --stix-data DIR
                             --args-file ARGS.pkl [--limit-gb 4] [--sizes 65 129 257]

Run it with any Python. It captures the proximal step arguments of a real 129 x 129 MEM run once
(unless ARGS.pkl exists), then starts one watched process at a time for every implementation.
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
    ("xrayvision original", "original", {}),
    ("NumPy, no units, in place", "numpy", {}),
    ("Rust, 1 thread", "rust", {"RAYON_NUM_THREADS": "1"}),
    ("Rust, all cores", "rust", {}),
]
REPEATS = {"original": 3, "numpy": 5, "rust": 7}


def run(args, cmd, env):
    full = [sys.executable, args.memwatch, "--limit-gb", args.limit_gb, "--", args.python, str(HERE / "bench_mem.py"), *cmd]
    p = subprocess.run(full, capture_output=True, text=True, env={**os.environ, "HELIOX_STIX_DATA": args.stix_data, **env})
    m = re.search(r"peak RSS ([\d.]+) GB", p.stderr)
    killed = "KILLED" in p.stderr
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        res = None
        print(p.stderr[-2000:], file=sys.stderr)
    return res, (m.group(1) if m else None), killed


def fmt_time(t):
    return f"{t * 1e3:.2f} ms" if t < 1 else f"{t:.2f} s"


def table(title, rows, first_column):
    print(f"\n{title}\n")
    print(f"| {first_column} | Implementation | Median time | Speedup vs original | Peak RSS (GB) | Identical result |")
    print("| --- | --- | ---: | ---: | ---: | :---: |")
    for case, label, impl, res, rss, killed in rows:
        if killed or res is None:
            print(f"| {case} | {label} | killed at the memory limit | | >{rss} | |")
            continue
        ref = next((r[3] for r in rows if r[0] == case and r[2] == "original"), None)
        speed = "1x" if impl == "original" else (f"{ref['median_s'] / res['median_s']:.1f}x" if ref else "")
        same = "yes" if ref and res["digest"] == ref["digest"] else ("reference" if impl == "original" else "NO")
        print(f"| {case} | {label} | {fmt_time(res['median_s'])} | {speed} | {res['peak_rss_gb']:.2f} | {same} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--memwatch", required=True)
    ap.add_argument("--stix-data", required=True)
    ap.add_argument("--args-file", required=True)
    ap.add_argument("--limit-gb", default="4")
    ap.add_argument("--sizes", type=int, nargs="*", default=[65, 129, 257])
    a = ap.parse_args()

    if not Path(a.args_file).exists():
        res, _, _ = run(a, ["capture", a.args_file, "129"], {})
        print(res, file=sys.stderr)

    rows = []
    for label, impl, env in IMPLEMENTATIONS:
        res, rss, killed = run(a, [impl, "prox", a.args_file, str(REPEATS[impl] + 2)], env)
        rows.append(("proximal step", label, impl, res, rss, killed))
        print(rows[-1], file=sys.stderr, flush=True)
    table("Proximal step on arguments captured from a 129 x 129 MEM run (median over 12 calls)", rows, "Case")

    rows = []
    for size in a.sizes:
        for label, impl, env in IMPLEMENTATIONS:
            res, rss, killed = run(a, [impl, "mem", str(size), str(REPEATS[impl])], env)
            rows.append((f"mem {size}x{size}", label, impl, res, rss, killed))
            print(rows[-1], file=sys.stderr, flush=True)
    table("Whole `mem` reconstruction, maxiter 1000, 24 visibilities, 2 arcsec pixels", rows, "Case")
    print("\nSetup RSS (STIX data and imports) before the timed runs, GB:")
    for case, label, impl, res, rss, killed in rows:
        if res:
            print(f"- {case}, {label}: {res['rss_after_setup_gb']}")


if __name__ == "__main__":
    main()

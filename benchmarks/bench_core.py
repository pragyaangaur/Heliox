"""
Benchmark the public Heliox API on whichever copy of Heliox is installed.

Install Heliox into a virtual environment, then run this file with that
environment's Python. Run it from a folder outside the source tree so that the
installed package is the one that gets imported:

    cd /some/scratch/folder
    python /path/to/repo/benchmarks/bench_core.py --label new --out new.json

The script writes one JSON file. For every task it holds the timed runs, the
median, the peak memory of the process and a fingerprint of the result. The
companion script `compare.py` reads two such files, checks that the fingerprints
agree and prints the table.

Each in-process task runs in a fresh Python process. Its peak memory is therefore
its own, and one task cannot warm up another. Every task runs once as a warm-up
and then a fixed number of times. The median of the timed runs is the headline
number. The import tasks run a new interpreter for every timed run and time the
import statement from inside that interpreter, so Python start-up is not counted.

Sample files are written to a folder created for this run and removed at the
end, so the installed package's own cache is never touched. The sample
generation task times the generators with an empty folder every time.
"""

import argparse
import collections
import gc
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta

SAMPLE_DIR_ENV = "HELIOX_SAMPLE_DIR"
WARMUP_RUNS = 1
SCRATCH_PREFIX = "heliox-bench-"

SAMPLE_NAMES = [
    "AIA_171_IMAGE",
    "AIA_193_IMAGE",
    "HMI_MAGNETOGRAM",
    "HMI_CONTINUUM_IMAGE",
    "LASCO_C2_IMAGE",
    "GOES_XRS_TIMESERIES",
    "NOAA_INDICES_TIMESERIES",
]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#: Bodies for the ephemeris task.
BODIES = ["earth", "venus", "mars", "jupiter", "saturn"]

#: Number of Helioprojective points in the coordinate transform task.
TRANSFORM_POINTS = 10**6

IN_PROCESS = {}
FRESH_PROCESS = {}


def in_process(key, label, repeats):
    """Register a task that is timed inside a fresh child process."""

    def register(builder):
        IN_PROCESS[key] = {"label": label, "repeats": repeats, "build": builder}
        return builder

    return register


def fresh_import(key, label, repeats, statement, check):
    """Register a task that times an import in a new interpreter."""
    FRESH_PROCESS[key] = {
        "label": label,
        "repeats": repeats,
        "statement": statement,
        "check": check,
    }


def _fingerprint(values):
    """Summarise a numeric array so that two runs can be compared."""
    import numpy as np

    flat = np.asarray(values, dtype=float).ravel()
    finite = flat[np.isfinite(flat)]
    return {
        "shape": list(np.shape(values)),
        "nan_count": int(np.count_nonzero(np.isnan(flat))),
        "sum": float(finite.sum()),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "head": [float(v) for v in flat[:3]],
        "tail": [float(v) for v in flat[-3:]],
    }


def _plain(quantity, unit):
    """Return a quantity in the given unit, or a plain array unchanged."""
    if not hasattr(quantity, "unit"):
        return quantity
    if unit is None:
        return quantity.value
    return quantity.to_value(unit)


def _fits_fingerprint(path):
    import numpy as np

    from astropy.io import fits

    entry = {}
    with fits.open(path) as hdul:
        for index, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            if hdu.is_image:
                entry[f"hdu{index}"] = _fingerprint(hdu.data)
                continue
            columns = {}
            for column in hdu.columns:
                values = np.asarray(hdu.data[column.name])
                if values.dtype.kind in "biuf":
                    columns[column.name] = _fingerprint(values)
                else:
                    columns[column.name] = {"length": int(len(values))}
            entry[f"hdu{index}"] = columns
    return entry


def _csv_fingerprint(path):
    with open(path, encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    rows = [line.split(",") for line in lines if line and not line.startswith("#")]
    header, body = rows[0], rows[1:]
    entry = {"header": header, "rows": len(body), "first_time": body[0][0]}
    for index, name in enumerate(header[1:], start=1):
        entry[name] = _fingerprint([float(row[index]) for row in body])
    return entry


# ---------------------------------------------------------------------------
# In-process tasks. Each builder returns (prepare, run, summarise, discard).
# `prepare` runs untimed and may import heliox. `run` is timed. `summarise`
# runs after timing and turns the result into plain numbers. `discard` removes
# whatever a run left on disk, and it also runs outside the timer.
# ---------------------------------------------------------------------------


@in_process("sample_generation", "Generate every sample file from scratch", 5)
def _build_sample_generation():
    def prepare():
        from heliox.data import sample

        return sample

    def run(sample):
        directory = tempfile.mkdtemp(prefix="heliox-bench-gen-")
        previous = os.environ.get(SAMPLE_DIR_ENV)
        os.environ[SAMPLE_DIR_ENV] = directory
        try:
            paths = {name: sample.get_sample_file(name) for name in SAMPLE_NAMES}
        finally:
            if previous is None:
                os.environ.pop(SAMPLE_DIR_ENV, None)
            else:
                os.environ[SAMPLE_DIR_ENV] = previous
        return {"directory": directory, "paths": paths}

    def summarise(result):
        out = {}
        for name in sorted(result["paths"]):
            path = result["paths"][name]
            if path.endswith(".csv"):
                out[name] = _csv_fingerprint(path)
            else:
                out[name] = _fits_fingerprint(path)
        return out

    def discard(result):
        shutil.rmtree(result["directory"], ignore_errors=True)

    return prepare, run, summarise, discard


@in_process("map_operations", "Load AIA map, submap, superpixel, resample, rotate", 7)
def _build_map_operations():
    def prepare():
        import heliox.map
        from heliox.data.sample import AIA_171_IMAGE

        return heliox.map, AIA_171_IMAGE

    def run(state):
        import astropy.units as u

        maps, path = state
        aia = maps.Map(path)
        return {
            "submap": aia.submap([100, 100] * u.pix, top_right=[400, 400] * u.pix),
            "superpixel": aia.superpixel((4, 4) * u.pix),
            "resample": aia.resample((256, 256) * u.pix),
            "rotate": aia.rotate(angle=30 * u.deg),
        }

    def summarise(result):
        return {name: _fingerprint(smap.data) for name, smap in result.items()}

    return prepare, run, summarise, None


@in_process("all_coordinates_from_map", "all_coordinates_from_map on the AIA map", 7)
def _build_all_coordinates():
    def prepare():
        import heliox.map
        from heliox.data.sample import AIA_171_IMAGE
        from heliox.map.maputils import all_coordinates_from_map

        return all_coordinates_from_map, heliox.map.Map(AIA_171_IMAGE)

    def run(state):
        function, aia = state
        return function(aia)

    def summarise(result):
        import astropy.units as u

        return {
            "Tx_arcsec": _fingerprint(result.Tx.to_value(u.arcsec)),
            "Ty_arcsec": _fingerprint(result.Ty.to_value(u.arcsec)),
        }

    return prepare, run, summarise, None


@in_process("helioprojective_transform", "Transform 1e6 Helioprojective points to Stonyhurst", 5)
def _build_transform():
    def prepare():
        import numpy as np

        import astropy.units as u
        from astropy.coordinates import SkyCoord

        import heliox.map
        from heliox.coordinates import HeliographicStonyhurst, Helioprojective
        from heliox.data.sample import AIA_171_IMAGE

        aia = heliox.map.Map(AIA_171_IMAGE)
        rng = np.random.default_rng(20261120)
        radius = 950.0 * np.sqrt(rng.uniform(size=TRANSFORM_POINTS))
        angle = 2 * np.pi * rng.uniform(size=TRANSFORM_POINTS)
        frame = Helioprojective(observer=aia.observer_coordinate, obstime=aia.date)
        coords = SkyCoord(
            radius * np.cos(angle) * u.arcsec,
            radius * np.sin(angle) * u.arcsec,
            frame=frame,
        )
        return coords, HeliographicStonyhurst(obstime=aia.date)

    def run(state):
        coords, target = state
        return coords.transform_to(target)

    def summarise(result):
        import astropy.units as u

        return {
            "lon_deg": _fingerprint(result.lon.to_value(u.deg)),
            "lat_deg": _fingerprint(result.lat.to_value(u.deg)),
        }

    return prepare, run, summarise, None


@in_process("sun_ephemeris", "sun.B0, L0, P and carrington_rotation_number, 10,000 times", 7)
def _build_sun_ephemeris():
    def prepare():
        import numpy as np

        import astropy.units as u
        from astropy.time import Time

        return Time("2000-01-01") + np.linspace(0, 7300, 10000) * u.day

    def run(times):
        from heliox.sun import B0, L0, P, carrington_rotation_number

        return {
            "B0": B0(times),
            "L0": L0(times),
            "P": P(times),
            "carrington": carrington_rotation_number(times),
        }

    def summarise(result):
        import astropy.units as u

        return {
            "B0_deg": _fingerprint(_plain(result["B0"], u.deg)),
            "L0_deg": _fingerprint(_plain(result["L0"], u.deg)),
            "P_deg": _fingerprint(_plain(result["P"], u.deg)),
            "carrington_number": _fingerprint(_plain(result["carrington"], None)),
        }

    return prepare, run, summarise, None


@in_process("body_positions", "get_body_heliographic_stonyhurst, 5 bodies, 1,000 times", 5)
def _build_body_positions():
    def prepare():
        import numpy as np

        import astropy.units as u
        from astropy.time import Time

        return Time("2010-01-01") + np.linspace(0, 1000, 1000) * u.day

    def run(times):
        from heliox.coordinates.ephemeris import get_body_heliographic_stonyhurst

        return {body: get_body_heliographic_stonyhurst(body, times) for body in BODIES}

    def summarise(result):
        import astropy.units as u

        out = {}
        for body, coord in result.items():
            out[body] = {
                "lon_deg": _fingerprint(coord.lon.to_value(u.deg)),
                "lat_deg": _fingerprint(coord.lat.to_value(u.deg)),
                "radius_km": _fingerprint(coord.radius.to_value(u.km)),
            }
        return out

    return prepare, run, summarise, None


def _time_strings(count=10000):
    """Build a mix of four common time formats, one quarter each."""
    base = datetime(1990, 1, 1)
    strings = []
    for index in range(count):
        stamp = base + timedelta(minutes=53 * index)
        kind = index % 4
        if kind == 0:
            strings.append(stamp.strftime("%Y-%m-%dT%H:%M:%S"))
        elif kind == 1:
            strings.append(stamp.strftime("%Y-%m-%d %H:%M:%S"))
        elif kind == 2:
            strings.append(stamp.strftime("%Y%m%d_%H%M%S"))
        else:
            month = MONTHS[stamp.month - 1]
            strings.append(f"{stamp:%d}-{month}-{stamp:%Y %H:%M:%S}")
    return strings


@in_process("parse_time", "parse_time on 10,000 strings in four formats", 7)
def _build_parse_time():
    def prepare():
        from heliox.time import parse_time

        return parse_time, _time_strings()

    def run(state):
        parse_time, strings = state
        return parse_time(strings)

    def summarise(result):
        return {"mjd": _fingerprint(result.mjd)}

    return prepare, run, summarise, None


@in_process("timerange_split_window", "TimeRange split and window over ten years", 5)
def _build_timerange():
    def prepare():
        from heliox.time import TimeRange

        return TimeRange

    def run(timerange_class):
        import astropy.units as u

        span = timerange_class("2010-01-01", "2020-01-01")
        return span.split(10000), span.window(6 * u.hour, 1 * u.day)

    def summarise(result):
        parts, windows = result
        return {
            "split_count": len(parts),
            "window_count": len(windows),
            "split_start_mjd": _fingerprint([part.start.mjd for part in parts]),
            "split_end_mjd": _fingerprint([part.end.mjd for part in parts]),
            "window_start_mjd": _fingerprint([win.start.mjd for win in windows]),
            "window_end_mjd": _fingerprint([win.end.mjd for win in windows]),
        }

    return prepare, run, summarise, None


@in_process("differential_rotate", "differential_rotate of the AIA map by one day", 5)
def _build_differential_rotate():
    def prepare():
        import astropy.units as u

        import heliox.map
        from heliox.data.sample import AIA_171_IMAGE
        from heliox.physics.differential_rotation import differential_rotate

        aia = heliox.map.Map(AIA_171_IMAGE)
        return differential_rotate, aia, aia.date + 1 * u.day

    def run(state):
        function, aia, target = state
        return function(aia, time=target)

    def summarise(result):
        return _fingerprint(result.data)

    return prepare, run, summarise, None


@in_process("goes_noaa_flare_class", "Load GOES and NOAA series, compute flare classes", 7)
def _build_goes_noaa():
    def prepare():
        from heliox.data.sample import GOES_XRS_TIMESERIES, NOAA_INDICES_TIMESERIES
        from heliox.timeseries import TimeSeries

        return TimeSeries, GOES_XRS_TIMESERIES, NOAA_INDICES_TIMESERIES

    def run(state):
        series_factory, goes_path, noaa_path = state
        goes = series_factory(goes_path)
        noaa = series_factory(noaa_path)
        return {
            "goes": goes,
            "noaa": noaa,
            "classes": goes.flare_classes(),
            "peak_class": goes.flare_class,
            "peak_flux": goes.peak_flux,
        }

    def summarise(result):
        import numpy as np

        import astropy.units as u

        goes, noaa = result["goes"], result["noaa"]
        counts = collections.Counter(str(name) for name in result["classes"])
        return {
            "goes_rows": len(goes),
            "goes_xrsb": _fingerprint(np.asarray(goes.data["xrsb"], dtype=float)),
            "noaa_rows": len(noaa),
            "sunspot_number": _fingerprint(np.asarray(noaa.data["sunspot_number"], dtype=float)),
            "flare_class_counts": [[name, counts[name]] for name in sorted(counts)],
            "peak_flare_class": str(result["peak_class"]),
            "peak_flux_W_per_m2": float(result["peak_flux"].to_value(u.W / u.m**2)),
        }

    return prepare, run, summarise, None


@in_process("fido_search", "Fido search on the sample catalogue (two queries)", 5)
def _build_fido():
    def prepare():
        import astropy.units as u

        from heliox.net import Fido
        from heliox.net import attrs as a

        return Fido, a, u

    def run(state):
        fido, a, _ = state
        aia = fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("AIA"))
        everything = fido.search(a.Time("2013-10-28", "2013-10-29"))
        return aia, everything

    def summarise(result):
        aia, everything = result
        return {"aia_file_count": int(aia.file_num), "all_file_count": int(everything.file_num)}

    return prepare, run, summarise, None


# ---------------------------------------------------------------------------
# Fresh-interpreter import tasks. The statement is timed inside the new process.
# ---------------------------------------------------------------------------

fresh_import(
    "import_heliox", "import heliox (fresh interpreter)", 7, "import heliox", "heliox.__version__"
)
fresh_import(
    "import_heliox_map",
    "import heliox.map (fresh interpreter)",
    7,
    "import heliox.map",
    "heliox.map.__name__",
)

FRESH_TEMPLATE = """
import time
_start = time.perf_counter()
{statement}
_elapsed = time.perf_counter() - _start
print(repr(_elapsed))
print({check})
"""

ENVIRONMENT_CODE = """
import importlib.util, json, os, sys
from importlib import metadata
import heliox
packages = {}
for name in ["numpy", "scipy", "astropy", "pandas", "matplotlib", "pyerfa"]:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    "python": sys.version.split()[0],
    "heliox_version": heliox.__version__,
    "heliox_file_in_environment": os.path.relpath(heliox.__file__, sys.prefix),
    "has_compiled_core": importlib.util.find_spec("heliox._core") is not None,
    "packages": packages,
}))
"""


# ---------------------------------------------------------------------------
# Child mode: run one in-process task and write its record as JSON.
# ---------------------------------------------------------------------------


def _peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(peak / divisor, 1)


def _run_child(key, out_path):
    spec = IN_PROCESS[key]
    prepare, run, summarise, discard = spec["build"]()
    state = prepare()

    times = []
    result = None
    for index in range(WARMUP_RUNS + spec["repeats"]):
        if result is not None and discard is not None:
            discard(result)
        gc.collect()
        start = time.perf_counter()
        result = run(state)
        elapsed = time.perf_counter() - start
        if index >= WARMUP_RUNS:
            times.append(elapsed)
    peak = _peak_rss_mb()

    fingerprint = summarise(result)
    if discard is not None:
        discard(result)

    record = {"key": key, "times_s": times, "peak_rss_mb": peak, "result": fingerprint}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)


# ---------------------------------------------------------------------------
# Driver mode: run every task in its own process and write the result file.
# ---------------------------------------------------------------------------


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _hardware():
    info = {}
    for key, name in [
        ("cpu", "machdep.cpu.brand_string"),
        ("logical_cpus", "hw.ncpu"),
        ("memory_bytes", "hw.memsize"),
    ]:
        try:
            info[key] = subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            info[key] = None
    return info


def _load_average():
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except OSError:
        return None


def _run_fresh(key, spec, env, cwd):
    code = FRESH_TEMPLATE.format(statement=spec["statement"], check=spec["check"])
    times, check_value = [], None
    for index in range(WARMUP_RUNS + spec["repeats"]):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            check=True,
        )
        lines = proc.stdout.strip().splitlines()
        if index >= WARMUP_RUNS:
            times.append(float(lines[0]))
        check_value = lines[1]
    return {
        "key": key,
        "times_s": times,
        "peak_rss_mb": None,
        "result": {"printed": check_value},
    }


def _run_driver(args):
    script = os.path.abspath(__file__)
    scratch = tempfile.mkdtemp(prefix=SCRATCH_PREFIX)
    env = dict(os.environ)
    env[SAMPLE_DIR_ENV] = os.path.join(scratch, "samples")
    try:
        environment = json.loads(
            subprocess.run(
                [sys.executable, "-c", ENVIRONMENT_CODE],
                capture_output=True,
                text=True,
                env=env,
                cwd=scratch,
                check=True,
            ).stdout
        )
        load_start = _load_average()

        wanted = args.only or list(IN_PROCESS) + list(FRESH_PROCESS)
        records = []
        for key in wanted:
            if key in IN_PROCESS:
                spec = IN_PROCESS[key]
                child_out = os.path.join(scratch, f"{key}.json")
                subprocess.run(
                    [sys.executable, script, "--child", key, "--child-out", child_out],
                    env=env,
                    cwd=scratch,
                    check=True,
                )
                with open(child_out, encoding="utf-8") as handle:
                    record = json.load(handle)
                kind = "in-process"
            elif key in FRESH_PROCESS:
                spec = FRESH_PROCESS[key]
                record = _run_fresh(key, spec, env, scratch)
                kind = "fresh interpreter"
            else:
                raise SystemExit(f"Unknown task {key!r}")
            times = record["times_s"]
            records.append(
                {
                    "key": key,
                    "label": spec["label"],
                    "kind": kind,
                    "warmup_runs": WARMUP_RUNS,
                    "repeats": spec["repeats"],
                    "times_s": times,
                    "median_s": _median(times),
                    "min_s": min(times),
                    "peak_rss_mb": record["peak_rss_mb"],
                    "result": record["result"],
                }
            )
            print(f"{key}: median {_median(times):.4g} s", file=sys.stderr)

        if args.append and os.path.exists(args.out):
            with open(args.out, encoding="utf-8") as handle:
                earlier = {task["key"]: task for task in json.load(handle)["tasks"]}
            earlier.update({record["key"]: record for record in records})
            records = [
                earlier[key] for key in list(IN_PROCESS) + list(FRESH_PROCESS) if key in earlier
            ]

        output = {
            "label": args.label,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hardware": _hardware(),
            "heliox": environment,
            "load_average_start": load_start,
            "load_average_end": _load_average(),
            "tasks": records,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1)
        handle.write("\n")
    print(f"Wrote {args.out}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--label", help="A name for this run, for example old or new.")
    parser.add_argument("--out", help="Path of the JSON file to write.")
    parser.add_argument("--only", nargs="+", metavar="TASK", help="Run only these tasks.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge the tasks run now into an existing result file with the same label.",
    )
    parser.add_argument("--child", metavar="TASK", help=argparse.SUPPRESS)
    parser.add_argument("--child-out", metavar="PATH", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        _run_child(args.child, args.child_out)
        return
    if not args.label or not args.out:
        parser.error("--label and --out are required")
    _run_driver(args)


if __name__ == "__main__":
    main()

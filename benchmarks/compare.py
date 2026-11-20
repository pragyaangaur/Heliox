"""
Compare two result files written by bench_core.py and print a markdown table.

    python benchmarks/compare.py benchmarks/results_old.json benchmarks/results_new.json

The speedup is the old median divided by the new median, so a value below 1
means the new code is slower. Every task also has its result fingerprint
checked against the old one. A number is compared with its difference divided
by the larger of 1 and the two magnitudes, so small angles and large sums are
judged on the same scale. A task passes when every number agrees within
TOLERANCE and every shape, count and string is identical.
"""

import argparse
import json
import math
import sys

TOLERANCE = 1e-9


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _walk(old, new, path, state):
    """Compare two fingerprints and record the largest scaled difference."""
    if isinstance(old, dict) and isinstance(new, dict):
        if sorted(old) != sorted(new):
            state["mismatches"].append(f"{path}: keys differ")
            return
        for key in old:
            _walk(old[key], new[key], f"{path}/{key}", state)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            state["mismatches"].append(f"{path}: length {len(old)} != {len(new)}")
            return
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            _walk(a, b, f"{path}[{index}]", state)
    elif _is_number(old) and _is_number(new):
        a, b = float(old), float(new)
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                state["mismatches"].append(f"{path}: NaN on one side only")
            return
        if a == b:
            return
        scaled = abs(a - b) / max(1.0, abs(a), abs(b))
        if scaled > state["max_diff"]:
            state["max_diff"] = scaled
            state["max_at"] = path
    elif old != new:
        state["mismatches"].append(f"{path}: {old!r} != {new!r}")


def check_results(old_result, new_result):
    """Return (largest scaled difference, where it is, list of mismatches)."""
    state = {"max_diff": 0.0, "max_at": "", "mismatches": []}
    _walk(old_result, new_result, "", state)
    return state["max_diff"], state["max_at"], state["mismatches"]


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _by_key(data):
    return {task["key"]: task for task in data["tasks"]}


def _speed(old, new):
    return old / new


def _format_speedup(value):
    text = f"{value:.2f}x"
    return f"{text} (slower)" if value < 1 else text


def _format_seconds(value):
    return f"{value:.4g}"


def _format_rss(value):
    return "n/a" if value is None else f"{value:.0f}"


def _geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _environment_lines(label, data):
    info = data["heliox"]
    has_core = info["has_compiled_core"]
    return (
        f"- {label}: Heliox {info['heliox_version']} at `{info['heliox_file_in_environment']}` inside the environment, "
        f"compiled core present: {has_core}, Python {info['python']}"
    )


def build_report(old_data, new_data):
    """Return the markdown report as a string."""
    old_tasks = _by_key(old_data)
    new_tasks = _by_key(new_data)
    keys = [key for key in old_tasks if key in new_tasks]

    lines = []
    lines.append(_environment_lines("Old", old_data))
    lines.append(_environment_lines("New", new_data))
    if old_data["heliox"]["has_compiled_core"] or not new_data["heliox"]["has_compiled_core"]:
        lines.append("")
        lines.append(
            "Warning: the old install should not have a compiled core and the new one should."
        )
    lines.append("")
    lines.append(
        "| Task | Old median (s) | New median (s) | Speedup | Largest scaled difference | Check |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")

    speedups = {}
    mismatched = []
    for key in keys:
        old, new = old_tasks[key], new_tasks[key]
        speedup = _speed(old["median_s"], new["median_s"])
        speedups[key] = speedup
        max_diff, max_at, mismatches = check_results(old["result"], new["result"])
        if mismatches or max_diff > TOLERANCE:
            verdict = "FAIL"
            mismatched.append(key)
        else:
            verdict = "pass"
        diff_text = "0" if max_diff == 0.0 else f"{max_diff:.2e}"
        lines.append(
            f"| {old['label']} | {_format_seconds(old['median_s'])} | "
            f"{_format_seconds(new['median_s'])} | {_format_speedup(speedup)} | "
            f"{diff_text} | {verdict} |"
        )

    lines.append("")
    all_values = list(speedups.values())
    import_keys = [key for key in keys if old_tasks[key]["kind"] == "fresh interpreter"]
    work_values = [speedups[key] for key in keys if key not in import_keys]
    lines.append(
        f"Geometric mean speedup, all {len(all_values)} tasks: {_geometric_mean(all_values):.2f}x"
    )
    lines.append(
        f"Geometric mean speedup, {len(work_values)} tasks without the import rows: "
        f"{_geometric_mean(work_values):.2f}x"
    )
    lines.append("")
    passed = len(keys) - len(mismatched)
    lines.append(f"Result checks: {passed} of {len(keys)} tasks agree within {TOLERANCE:g}.")
    if mismatched:
        lines.append("Failing tasks: " + ", ".join(mismatched))
    lines.append("")
    lines.append("| Task | Old peak RSS (MB) | New peak RSS (MB) |")
    lines.append("| --- | ---: | ---: |")
    for key in keys:
        old, new = old_tasks[key], new_tasks[key]
        if old["kind"] != "in-process":
            continue
        lines.append(
            f"| {old['label']} | {_format_rss(old['peak_rss_mb'])} | "
            f"{_format_rss(new['peak_rss_mb'])} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("old", help="Result file from the old Heliox.")
    parser.add_argument("new", help="Result file from the new Heliox.")
    args = parser.parse_args()
    sys.stdout.write(build_report(_load(args.old), _load(args.new)))


if __name__ == "__main__":
    main()

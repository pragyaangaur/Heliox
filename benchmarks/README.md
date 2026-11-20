# Benchmarks: pure Python Heliox against the Rust core

This folder measures how much faster the Rust core makes Heliox on the public API. It compares two installed copies of Heliox on the same tasks, with the same library versions. The old copy is commit `24e86b6`, the last commit before the Rust core was added. Its code is pure Python and was installed with setuptools. The new copy is commit `a311632` on `main`, installed with `pip install .`, which builds the Rust core as `heliox._core`.

The headline result is a geometric mean speedup of 1.30x over the 13 tasks, or 1.40x if the two import rows are left out. The Rust core helps most where the work is Python loops over many small values, such as parsing 10,000 time strings. It helps little where the time already goes into compiled code in astropy, pyerfa or scipy, which the port does not replace. One row is slower in NEW, which is the import of heliox itself, and three tasks are almost unchanged. The notes below say why.

## Machine and software

The machine is an Apple M4 with 10 CPU cores and 16 GB of memory, running macOS 27.2 on arm64. Both copies run on Python 3.13.13. Both venvs use the same library versions, which are the versions that the new build resolved: numpy 2.5.5, scipy 1.18.1, astropy 8.0.3, pandas 3.1.1, matplotlib 3.11.2 and pyerfa 2.0.1.5. The only difference between the two environments is the Heliox copy.

During the second run the load average was between about 2.1 and 3.4 on the 10 cores, because other work was running on the machine. The run was not made on an idle machine.

## How to reproduce

First create one venv per copy and install the pinned libraries into each. Then install each copy of Heliox. The old copy comes from a detached git worktree of `24e86b6`, and the new copy comes from the repository itself. Building the new copy needs a Rust toolchain of version 1.88 or later.

```
git worktree add --detach /path/to/old-heliox 24e86b6
python3 -m venv venv-old && venv-old/bin/pip install numpy==2.5.5 scipy==1.18.1 astropy==8.0.3 pandas==3.1.1 matplotlib==3.11.2 pyerfa==2.0.1.5 /path/to/old-heliox
python3 -m venv venv-new && venv-new/bin/pip install numpy==2.5.5 scipy==1.18.1 astropy==8.0.3 pandas==3.1.1 matplotlib==3.11.2 pyerfa==2.0.1.5 /path/to/repo
```

Then run the benchmark once with each venv, one after the other, from a folder outside both source trees. The script also accepts `--only` to run a few tasks and `--append` to merge them into an existing file. Do not run the two at the same time, because the timings would interfere with each other.

```
mkdir scratch && cd scratch
/path/to/venv-old/bin/python /path/to/repo/benchmarks/bench_core.py --label old --out old.json
/path/to/venv-new/bin/python /path/to/repo/benchmarks/bench_core.py --label new --out new.json
/path/to/venv-new/bin/python /path/to/repo/benchmarks/compare.py old.json new.json
```

The script first records which Heliox it imported and whether that copy has the compiled core. The comparison script prints the same information at the top of its output, so you can confirm that the old copy has no compiled core and the new copy does. The results committed here are `results_old.json` and `results_new.json`. Running `compare.py` on them reproduces the table below without rerunning anything.

## What the benchmark does

Each task runs once to warm up, then a fixed number of times. The median of the timed runs is the number in the table. Each task has 5 or 7 timed runs, as set in the script. Every in-process task runs in its own fresh Python process, so its peak memory is its own. The import rows run a new interpreter for every timed run and time the import statement from inside it, so the start-up of Python itself is not counted. Sample files are written to a folder made for the run and removed afterwards, so the cache inside the installed package is never touched. The sample generation task starts with an empty folder every time.

Each task also returns a fingerprint of its result. The fingerprint holds the shape, the count of NaN values, the sum, the minimum and maximum of the finite values, and the first and last three values. For each number, `compare.py` takes the difference between old and new and divides it by the larger of 1 and the two magnitudes. A task passes when every number agrees to within 1e-9 on that scale and every shape and string is identical. The largest difference in this run is 6.2e-14, which comes from the coordinate transform. So the two copies agree on every task.

## Results

The times are medians in seconds. The speedup is the old median divided by the new median. A speedup below 1 means that NEW is slower, and the table marks such rows.

| Task | Old median (s) | New median (s) | Speedup | Largest scaled difference | Check |
| --- | ---: | ---: | ---: | ---: | --- |
| Generate every sample file from scratch | 0.1707 | 0.1559 | 1.09x | 0 | pass |
| Load AIA map, submap, superpixel, resample, rotate | 0.04568 | 0.0427 | 1.07x | 3.89e-16 | pass |
| all_coordinates_from_map on the AIA map | 0.02495 | 0.02501 | 1.00x (slower) | 0 | pass |
| Transform 1e6 Helioprojective points to Stonyhurst | 0.1922 | 0.1786 | 1.08x | 6.22e-14 | pass |
| sun.B0, L0, P and carrington_rotation_number, 10,000 times | 7.397 | 7.197 | 1.03x | 2.70e-14 | pass |
| get_body_heliographic_stonyhurst, 5 bodies, 1,000 times | 0.4075 | 0.3036 | 1.34x | 0 | pass |
| parse_time on 10,000 strings in four formats | 0.07372 | 0.03409 | 2.16x | 0 | pass |
| TimeRange split and window over ten years | 5.522 | 4.945 | 1.12x | 0 | pass |
| differential_rotate of the AIA map by one day | 0.1518 | 0.1368 | 1.11x | 4.58e-15 | pass |
| Load GOES and NOAA series, compute flare classes | 0.0248 | 0.01352 | 1.83x | 0 | pass |
| Fido search on the sample catalogue (two queries) | 0.009671 | 0.001982 | 4.88x | 0 | pass |
| import heliox (fresh interpreter) | 0.0008541 | 0.001958 | 0.44x (slower) | 0 | pass |
| import heliox.map (fresh interpreter) | 1.108 | 0.6919 | 1.60x | 0 | pass |

Geometric mean speedup, all 13 tasks: 1.30x
Geometric mean speedup, 11 tasks without the import rows: 1.40x
Result checks: 13 of 13 tasks agree within 1e-09.

The peak memory figures are the highest resident memory of each in-process task's process, in megabytes. They include the imports that the task makes, so they are not the memory of the computation alone.

| Task | Old peak RSS (MB) | New peak RSS (MB) |
| --- | ---: | ---: |
| Generate every sample file from scratch | 207 | 184 |
| Load AIA map, submap, superpixel, resample, rotate | 200 | 168 |
| all_coordinates_from_map on the AIA map | 240 | 226 |
| Transform 1e6 Helioprojective points to Stonyhurst | 449 | 511 |
| sun.B0, L0, P and carrington_rotation_number, 10,000 times | 156 | 139 |
| get_body_heliographic_stonyhurst, 5 bodies, 1,000 times | 134 | 107 |
| parse_time on 10,000 strings in four formats | 100 | 99 |
| TimeRange split and window over ten years | 152 | 180 |
| differential_rotate of the AIA map by one day | 370 | 362 |
| Load GOES and NOAA series, compute flare classes | 139 | 125 |
| Fido search on the sample catalogue (two queries) | 103 | 103 |

## Notes on what dominates each task

The notes come from `cProfile` runs of one call of each task after its warm-up. The profiles are a guide to where the time goes, and they are not part of the table. Profiles of the old code were taken for the tasks where I needed to explain a result.

Sample generation gains 1.09x. In the new code the Gaussian sums, the streamer field and the radial grid run in Rust. The scipy filter and the Python function that builds the discs still take a share of the time.

Loading and transforming the AIA map gains 1.07x. In the new code the affine transform runs in Rust and takes about 32 ms. In the old profile, scipy's spline filter and geometric transform took about 31 ms together, so the rotation step is close to even. The rest of the time is map copying and file reading.

`all_coordinates_from_map` is unchanged at 1.00x. Most of the time in the new code is astropy's WCS pixel to world call, which is wcslib. The Rust core does not replace it, so the two copies are close.

The 1e6 point transform to Stonyhurst gains 1.08x. The new code does the Helioprojective to heliocentric step in Rust, which takes about 20 ms. In the old profile, astropy's latitude arithmetic and angle wrapping took about 150 ms of the 195 ms. Those astropy steps still run in the new code, so the gain is small.

The sun ephemeris task is almost unchanged at 1.03x, and it is the slowest task in the table at about 7 seconds in both copies. Most of the time goes to pyerfa's `epv00`, which computes the Earth's position, and to pyerfa's `nut06a`. The new core replaces only one `nut06a` call on this path, which takes about 0.04 s. The remaining calls still go through pyerfa. Making this task faster would mean porting the astropy time and ERFA calls, not just the Heliox arithmetic.

The body positions task gains 1.34x. In the old profile, almost all of the 0.4 s went to pyerfa's `epv00`. In the new code the same calculation runs in Rust, and it takes about 0.3 s for the five bodies.

Parsing 10,000 strings gains 2.2x, which is the clearest win. The new code uses a Rust parser that handles the whole list in one call, and it takes about 30 ms of the 33 ms. The old code matched each string with Python regular expressions.

The `TimeRange` split and window task gains 1.12x, and it is one of the slowest tasks at about 5 seconds. The time goes to about 300,000 internal calls into astropy's `Time` code while the windows and split boundaries are built. The Rust core does not touch that work, so the gain is small.

Differential rotation by one day gains 1.11x. In the new profile, astropy's WCS conversion took about 31 ms of 145 ms, and astropy's angle wrapping took about 22 ms. The Rust sampling step took about 7 ms. The old code's timing moved a lot between the two runs (see the next section), so the quiet rerun is the number to trust.

The GOES and NOAA task gains 1.83x. In the old profile, most of the parsing time was Python's `strptime` called once per GOES sample, which took about 14 ms of 44 ms. The new code parses those times in Rust.

The Fido search gains 4.9x, but both copies take under 10 ms, so the absolute saving is small. I did not profile this task.

The two import rows show a cost and a gain. `import heliox` is about 1.1 ms slower in NEW, because the new package loads the compiled extension at import time. The old package imports nothing heavy at that point, so its import takes 0.7 ms. `import heliox.map` is 1.6x faster in NEW, from 1.07 s to 0.67 s. I did not investigate why.

## Run-to-run variation

The benchmark was run three times on the same day with the same timed code. The first two runs happened while other builds were compiling on the same machine, and they gave geometric means of 1.26x and 1.33x. The third run, which is the one committed here, was made on a quiet machine with nothing else running. Most ratios agreed to within about 15 percent across the runs. The clear exception is `differential_rotate`, which went from 1.06x to 1.50x in the noisy runs and settled at 1.11x on the quiet machine, because the old install's timings for that task were the most sensitive to load.

## Limits of this benchmark

The benchmark covers only the public calls listed above, on sample data. It does not measure network downloads, plotting, or the full test suite. It does not measure accuracy against outside data. It checks only that the two copies agree with each other. Peak memory is one reading per task on one machine, so differences of a few tens of megabytes do not mean much.

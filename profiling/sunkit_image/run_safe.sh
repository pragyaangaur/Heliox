#!/bin/bash
# Run every bench group in its own process under memwatch (kills the tree above LIMIT_GB, prints peak RSS).
# usage: run_safe.sh MEMWATCH_PY VENV_PYTHON SAMPLE_DIR OUT_DIR [LIMIT_GB]
MW=$1; PY=$2; DATA=$3; OUT=$4; LIM=${5:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for g in trace.smooth trace.bandpass trace.occult2 asda.gamma asda.edges radial.nrgf radial.rhef radial.intensity_enhance radial.fnrgf enhance.mgn enhance.wow time_lag coalignment granule.segment stara utils.radii utils.radial_summary utils.polygon utils.noise; do
  echo "=== $g"
  /Users/pragyaangaur/miniconda3/bin/python3 "$MW" --limit-gb "$LIM" -- "$PY" -I "$HERE/bench.py" --sample-dir "$DATA" --ops "$g" --repeats 3 --budget 20 --out "$OUT/$g.json" 2>&1 | grep -vE "INFO|leaked|warnings.warn"
done

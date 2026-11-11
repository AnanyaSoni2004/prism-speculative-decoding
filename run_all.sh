#!/usr/bin/env bash
# Run every PRISM experiment. Usage: bash run_all.sh [--skip-models] [--cpu] [--n N]
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_XET=1
export PYTHONHASHSEED=0
PY=.venv/bin/python

SKIP=0
E3_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-models) SKIP=1 ;;
    --cpu) E3_ARGS+=(--cpu) ;;
    --n) E3_ARGS+=(--n "$2"); shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done

mkdir -p results/figures
rm -f results/runtime.json results/seeds.json results/e3_deviations.json
if [[ $SKIP == 1 ]]; then
  # make sure a skipped run never reports stale E3 numbers
  rm -f results/e3_*.csv results/e3_*.npz results/e3_*.json results/traces_*.npz results/timing.csv results/table3_*
fi
START=$(date +%s)

echo "== tests"; $PY -m pytest -q tests
$PY -m pip freeze > results/env.txt

echo "== E0"; $PY e0_checks.py
echo "== E1"; $PY e1_fixed_alpha.py
echo "== E2"; $PY e2_variable_alpha.py
if [[ $SKIP == 0 ]]; then
  echo "== E3"; $PY e3_gpt2_traces.py "${E3_ARGS[@]+"${E3_ARGS[@]}"}"
else
  echo "== E3 skipped (--skip-models)"
fi
echo "== figures"; $PY figures.py

END=$(date +%s)
$PY -c "from prism.util import record_runtime; record_runtime('total', $END - $START)"
$PY make_report.py
echo "== E4 follow-up"
if [[ $SKIP == 1 ]]; then $PY e4_followup.py --skip-models; else $PY e4_followup.py; fi

echo
echo "REPORT: $(pwd)/results/REPORT.md"
$PY summary.py

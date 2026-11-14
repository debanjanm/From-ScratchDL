#!/usr/bin/env bash
# Runs every topic headless: notebooks via nbconvert, remaining scripts directly.
# Each one asserts its own result. Notebooks are executed to a temp dir (repo files untouched).
cd "$(dirname "$0")" || exit 1
export MPLBACKEND=Agg
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
run() {  # run <label> <cmd...>
  label=$1; shift; start=$(date +%s)
  if "$@" >"$tmp/log" 2>&1; then
    printf "PASS  %-52s %ss\n" "$label" "$(( $(date +%s) - start ))"
  else
    printf "FAIL  %-52s\n" "$label"; tail -5 "$tmp/log" | sed 's/^/        /'; fail=1
  fi
}
for f in common/gradcheck.py common/framework.py [0-9][0-9]_*/*.py; do
  [ -e "$f" ] && run "$f" python3 "$f"
done
for f in [0-9][0-9]_*/*.ipynb; do
  [ -e "$f" ] && run "$f" python3 -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 --output-dir "$tmp" "$f"
done
exit $fail

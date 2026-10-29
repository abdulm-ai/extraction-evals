#!/usr/bin/env bash
# Rerun run_gemini.sh after each daily reset until every run in
# configs/gemini.json has written its predictions.
cd "$(dirname "$0")/.."
done_all() { .venv/Scripts/python -c "import json,pathlib,sys; g=json.load(open('configs/gemini.json')); sys.exit(0 if all((pathlib.Path('runs')/r['name']/'config.json').exists() for r in g['runs']) else 1)"; }
while ! done_all; do
  now=$(date +%s); next=$(date -d "tomorrow 00:10" +%s)
  echo "$(date) waiting $(( (next-now)/60 )) min for the quota reset"
  sleep $(( next - now ))
  echo "$(date) starting"
  bash scripts/run_gemini.sh
done
echo "$(date) all runs done"

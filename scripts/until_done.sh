#!/usr/bin/env bash
# Rerun run_gemini.sh every 2 hours until every run in configs/gemini.json
# has written its predictions. A run that hits the free-tier daily quota
# stops on its first rejected call, so an early retry costs almost nothing.
cd "$(dirname "$0")/.."
done_all() { .venv/Scripts/python -c "import json,pathlib,sys; g=json.load(open('configs/gemini.json')); sys.exit(0 if all((pathlib.Path('runs')/r['name']/'config.json').exists() for r in g['runs']) else 1)"; }
while ! done_all; do
  echo "$(date) starting"
  bash scripts/run_gemini.sh
  done_all && break
  echo "$(date) waiting 2 h"
  sleep 7200
done
echo "$(date) all runs done"

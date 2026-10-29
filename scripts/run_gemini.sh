#!/usr/bin/env bash
# One process per model: the free tier's limits are per model (15 requests a
# minute and 500 a day for the Flash-Lite models; the Flash models allow 20 a
# day, too few for this grid). A run that hits the daily cap stops; finished
# calls are cached, so rerunning this script after the reset (midnight
# Pacific) picks up where it left off.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
mkdir -p logs
$PY -u -m xeval run configs/gemini.json lite-3.5 lite-3.5-v2 lite-3.5-free lite-3.5-degrade >> logs/lite-3.5.log 2>&1 &
$PY -u -m xeval run configs/gemini.json lite-3.1 lite-3.1-v2 lite-3.1-free lite-3.1-degrade lite-3.1-ocr lite-3.1-gt_text lite-3.1-image+ocr >> logs/lite-3.1.log 2>&1 &
wait

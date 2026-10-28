"""python -m xeval <command>

  data                      download CORD, render the invoices, write gold/
  run <config.json> [names] run configurations from a grid file
  score <run> [<run> ...]   per-field table for runs
  diff <run_a> <run_b>      paired per-field comparison; exit 1 on a regression
  report                    every results table and chart, into results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    load_dotenv(ROOT / ".env")
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "data":
        from .data import cord, invoices
        from .ocr import fetch_models
        fetch_models()
        cord.build(ROOT / "data" / "raw" / "cord", ROOT / "data" / "cord", ROOT / "gold")
        invoices.build(ROOT / "data" / "invoices", ROOT / "gold")
        return 0
    if cmd == "run":
        from .run import RunConfig, run
        grid = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        defaults = grid.get("defaults", {})
        only = set(args[1:])
        for spec in grid["runs"]:
            if only and spec["name"] not in only:
                continue
            run(RunConfig(**(defaults | spec)))
        return 0
    if cmd == "score":
        from .report import print_scores
        print_scores(args)
        return 0
    if cmd == "diff":
        from .diff import main as diff_main
        return diff_main(args)
    if cmd == "report":
        from .report import build_report
        build_report()
        return 0
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

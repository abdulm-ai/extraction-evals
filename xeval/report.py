"""Tables for the terminal. The charts and results/ are built by
scripts/results.py, which reads the same RunScores."""

from __future__ import annotations

import math

from .results import cost_latency, field_table, load_run


def _pct(x: float) -> str:
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:5.1f}"


def print_scores(names: list[str]) -> None:
    for name in names:
        rs = load_run(name)
        for s in sorted({d.record["set"] for d in rs.docs}):
            sub = rs.by_set(s)
            print(f"\n{name}  [{s}, {len(sub.docs)} docs]  {rs.config['model']}  {rs.config['input_mode']}/{rs.config['output_mode']}/{rs.config['prompt']}")
            print(f"  {'field':<20} {'acc':>5}  {'95% CI':>13}  {'n':>4}  {'halluc':>6}")
            for r in field_table(sub):
                ci = f"{_pct(r['lo'])}-{_pct(r['hi']).strip()}"
                print(f"  {r['field']:<20} {_pct(r['acc'])}  {ci:>13}  {r['n']:>4}  {_pct(r['halluc'])}")
            cl = cost_latency(sub)
            print(f"  ${cl['cost_per_1k']:.2f} per 1,000 docs, p50 {cl['p50_s']:.1f}s, p95 {cl['p95_s']:.1f}s, "
                  f"{cl['failed']} failed, {cl['repaired']} repaired")


def build_report() -> None:
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "results.py"), run_name="__main__")

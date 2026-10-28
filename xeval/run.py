"""Run one configuration over a set of documents.

Every model call goes through a disk cache keyed by everything that can change
the answer (model, prompt, input bytes, schema, sample index), so re-scoring
is free and a crashed grid resumes where it stopped. Predictions are written
to runs/<name>/predictions.jsonl, which is committed: every table in the
README can be rebuilt from it without an API key.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import providers
from .checks import failed, run_checks
from .prompts import REPAIR, system_prompt, user_parts
from .schema import DOC_TYPES, json_schema
from .validate import parse_json, schema_errors

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"


@dataclass
class RunConfig:
    name: str
    model: str
    sets: list[str]                 # e.g. ["cord", "invoices"]
    split: str = "test"
    input_mode: str = "image"       # image | ocr | gt_text | image+ocr
    output_mode: str = "strict"     # strict | free
    prompt: str = "v1"
    repair: int = 1                 # repair turns allowed on sample 0
    samples: int = 1                # extra samples are first attempts only
    logprobs: bool = False
    max_docs: int | None = None
    workers: int = 8
    budget_usd: float = 5.0


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, cap: float):
        self.cap, self.spent, self.lock = cap, 0.0, threading.Lock()

    def check(self):
        if self.spent > self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.2f} of a ${self.cap:.2f} cap")

    def add(self, x: float):
        with self.lock:
            self.spent += x


def load_records(sets: list[str], split: str, max_docs: int | None = None) -> list[dict]:
    recs = []
    for s in sets:
        path = ROOT / "gold" / f"{s}_{split}.jsonl"
        rs = [json.loads(l) for l in path.open(encoding="utf-8")]
        recs += rs[:max_docs] if max_docs else rs
    return recs


def _file_hash(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]


def _key(model, system, turns, schema, sample, logprobs) -> str:
    norm = []
    for t in turns:
        if t["role"] == "assistant":
            norm.append({"role": "assistant", "text": t["text"]})
        else:
            norm.append({"role": "user", "parts": [
                {"type": "image", "sha": _file_hash(p["path"])} if p["type"] == "image" else p for p in t["parts"]]})
    blob = json.dumps({"m": model, "s": system, "t": norm, "x": schema, "k": sample, "lp": logprobs}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cached_call(model, system, turns, schema, sample, logprobs, ledger: Ledger | None) -> tuple[providers.Response, bool]:
    k = _key(model, system, turns, schema, sample, logprobs)
    path = CACHE / k[:2] / f"{k}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return providers.Response(**d), True
    if ledger:
        ledger.check()
    c = providers.client(model)
    abs_turns = [t if t["role"] == "assistant" else {"role": "user", "parts": [
        dict(p, path=str(ROOT / p["path"])) if p["type"] == "image" else p for p in t["parts"]]} for t in turns]
    if isinstance(c, providers.OpenAI):
        r = c.call(model, system, abs_turns, schema, logprobs=logprobs)
    else:
        r = c.call(model, system, abs_turns, schema)
    if ledger:
        ledger.add(r.cost(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(r)), encoding="utf-8")
    return r, False


def _ocr_text(record: dict, mode: str) -> str | None:
    if mode == "gt_text":
        return (ROOT / record["gt_text"]).read_text(encoding="utf-8")
    if mode in ("ocr", "image+ocr"):
        from .ocr import ocr_record
        return ocr_record({**record, "pages": [str(ROOT / p) for p in record["pages"]]})
    return None


def _attempt(doc, r: providers.Response, model: str) -> dict:
    parsed, perr = parse_json(r.text)
    errs = [perr] if perr else schema_errors(doc, parsed)
    checks = run_checks(doc, parsed) if not errs else []
    a = {
        "text": r.text, "parsed": parsed if not errs else None, "schema_errors": errs,
        "checks": [dataclasses.asdict(c) for c in checks],
        "usage": {"input": r.input_tokens, "output": r.output_tokens, "cached": r.cached_tokens, "reasoning": r.reasoning_tokens},
        "latency_s": round(r.latency_s, 3), "cost": r.cost(model), "stop_reason": r.stop_reason, "model_id": r.model_id,
    }
    if r.logprobs is not None:
        a["logprobs"] = r.logprobs
    return a


def run_doc(cfg: RunConfig, record: dict, sample: int, ledger: Ledger | None) -> dict:
    doc = DOC_TYPES[record["doc_type"]]
    system = system_prompt(doc, cfg.prompt, cfg.output_mode)
    schema = json_schema(doc) if cfg.output_mode == "strict" else None
    turns = [{"role": "user", "parts": user_parts(record, cfg.input_mode, _ocr_text(record, cfg.input_mode))}]
    attempts, cache_hits = [], 0
    repairs = cfg.repair if sample == 0 else 0
    for i in range(repairs + 1):
        try:
            r, hit = cached_call(cfg.model, system, turns, schema, sample, cfg.logprobs, ledger)
        except BudgetExceeded:
            raise
        except Exception as e:  # an API error is a failed document, recorded as such
            attempts.append({"error": f"{type(e).__name__}: {e}"[:500], "parsed": None, "schema_errors": ["api error"], "checks": [], "cost": 0.0,
                             "usage": {"input": 0, "output": 0, "cached": 0, "reasoning": 0}, "latency_s": 0.0})
            break
        cache_hits += hit
        a = _attempt(doc, r, cfg.model)
        attempts.append(a)
        problems = a["schema_errors"] + [c["message"] for c in a["checks"] if not c["ok"]]
        if not problems or i == repairs:
            break
        turns = turns + [
            {"role": "assistant", "text": r.text, "raw_content": r.extra.get("raw_content")},
            {"role": "user", "parts": [{"type": "text", "text": REPAIR.format(problems="\n".join(f"- {p}" for p in problems))}]},
        ]
    final = next((i for i in range(len(attempts) - 1, -1, -1) if attempts[i]["parsed"] is not None), None)
    return {"doc_id": record["id"], "sample": sample, "attempts": attempts, "final": final, "cache_hits": cache_hits}


def run(cfg: RunConfig, quiet: bool = False) -> Path:
    records = load_records(cfg.sets, cfg.split, cfg.max_docs)
    out_dir = RUNS / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(cfg.budget_usd)
    jobs = [(r, s) for r in records for s in range(cfg.samples)]
    results = []
    with ThreadPoolExecutor(cfg.workers) as ex:
        futs = {ex.submit(run_doc, cfg, r, s, ledger): (r["id"], s) for r, s in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except BudgetExceeded as e:
                for g in futs:
                    g.cancel()
                raise SystemExit(f"{cfg.name}: stopped, {e}. Finished documents are cached; raise the budget and rerun.")
            if not quiet and n % 25 == 0:
                print(f"  {cfg.name}: {n}/{len(jobs)}  ${ledger.spent:.2f} new spend", flush=True)
    results.sort(key=lambda d: (d["doc_id"], d["sample"]))
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for d in results:
            for a in d["attempts"]:
                a.pop("raw_content", None)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    meta = dataclasses.asdict(cfg) | {
        "model_ids": sorted({a.get("model_id", "") for d in results for a in d["attempts"] if a.get("model_id")}),
        "documents": len(records), "price_date": providers.PRICE_DATE,
        "cost_usd": round(sum(a["cost"] for d in results for a in d["attempts"]), 4),
        "written": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if not quiet:
        print(f"{cfg.name}: {len(records)} docs x {cfg.samples} samples, ${meta['cost_usd']:.2f} total, ${ledger.spent:.2f} new")
    return out_dir

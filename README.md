# extraction-evals

Schema-constrained extraction from receipts and invoices, with an eval harness that scores every field separately, tests which confidence signals actually predict errors, and catches a regression that the average accuracy hides.

<!-- RESULTS -->

## Architecture

```mermaid
flowchart LR
  D[("Documents<br/>CORD receipts, synthetic invoices,<br/>degradation set")] --> I["Input mode<br/>image / OCR text / both /<br/>perfect transcription"]
  I --> M["Model call<br/>Anthropic or OpenAI<br/>strict schema or free JSON"]
  M --> C[("Response cache<br/>hash of model, prompt,<br/>input bytes, schema, sample")]
  C --> V["Validation<br/>1. schema<br/>2. domain checks"]
  V -->|"failed, one repair turn"| R["Repair turn<br/>failed checks, named"]
  R --> M
  V --> P[("runs/*/predictions.jsonl<br/>committed")]
  G[("gold/*.jsonl<br/>+ corrections.jsonl")] --> SC
  P --> SC["Scorer<br/>normalise, per field,<br/>Hungarian line-item match"]
  SC --> RP["Report<br/>intervals, cost, latency,<br/>risk-coverage, calibration"]
  SC --> S["Signals<br/>stated, checks, samples,<br/>other model, logprobs"]
  SC --> DF["xeval diff A B<br/>McNemar per field, Holm<br/>exit 1 on a regression"]
```

| File | What it does |
|---|---|
| [`xeval/schema.py`](xeval/schema.py) | One field spec per document type, used as the model's JSON schema, the validator's field list and the scorer's field list. |
| [`xeval/normalize.py`](xeval/normalize.py) | Parses printed amounts in the gold (`16,500`, `Rp 9.736`, `@ 12.000`) and canonicalises predictions: dates, money, text, IDs. ANLS. |
| [`xeval/checks.py`](xeval/checks.py) | Domain checks. Each one names the fields it touches. |
| [`xeval/score.py`](xeval/score.py) | Per-field scoring; line items matched with the Hungarian algorithm. |
| [`xeval/stats.py`](xeval/stats.py) | Clustered bootstrap intervals, McNemar's exact test, Holm correction, detectable-difference calculation, AUROC, ECE, risk-coverage. |
| [`xeval/providers.py`](xeval/providers.py) | Anthropic and OpenAI behind one call. Prices with their date. |
| [`xeval/run.py`](xeval/run.py) | The runner: cache, repair loop, samples, spend cap, resume. |
| [`xeval/confidence.py`](xeval/confidence.py) | Confidence signals per field, the combined logistic model. |
| [`xeval/diff.py`](xeval/diff.py) | The regression diff. |
| [`xeval/data/cord.py`](xeval/data/cord.py), [`xeval/data/invoices.py`](xeval/data/invoices.py) | Dataset conversion and the invoice generator. |
| [`scripts/results.py`](scripts/results.py) | Every table and chart in this README, from the committed predictions. |

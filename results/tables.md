
### cord: field accuracy, % (95% interval)

| Field | Gemini 3.1 Flash-Lite | Gemini 3.5 Flash-Lite |
|---|---|---|
| subtotal | 66.0 (57.0-75.0) | 98.0 (95.0-100.0) |
| discount | 98.0 (95.0-100.0) | 100.0 (100.0-100.0) |
| service_charge | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| tax | 100.0 (100.0-100.0) | 99.0 (97.0-100.0) |
| total | 96.0 (92.0-99.0) | 96.0 (92.0-99.0) |
| cash_tendered | 96.0 (91.9-99.0) | 98.0 (94.9-100.0) |
| change_due | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| line_items | 82.0 (74.0-89.0) | 88.0 (81.0-94.0) |
| item.description | 95.6 (91.4-98.8) | 97.2 (93.7-99.6) |
| item.quantity | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| item.unit_price | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| item.line_total | 98.8 (96.1-100.0) | 99.6 (98.7-100.0) |
| ALL FIELDS | 49.0 (39.0-59.0) | 84.0 (77.0-91.0) |
| mean of fields | 92.2 | 97.4 |
| ALL FIELDS, uncorrected gold | 45.0 (35.0-55.0) | 77.0 (69.0-85.0) |
| $ per 1,000 docs | 0.90 | 1.36 |
| p50 / p95 s | 5.5 / 37.3 | 17.3 / 52.6 |

### invoices: field accuracy, % (95% interval)

| Field | Gemini 3.1 Flash-Lite | Gemini 3.5 Flash-Lite |
|---|---|---|
| vendor_name | 94.0 (89.0-98.0) | 99.0 (97.0-100.0) |
| invoice_number | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| issue_date | 96.0 (92.0-99.0) | 99.0 (97.0-100.0) |
| due_date | 96.0 (92.0-99.0) | 99.0 (97.0-100.0) |
| currency | 98.0 (95.0-100.0) | 99.0 (97.0-100.0) |
| subtotal | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| discount | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| shipping | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| tax | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| amount_paid | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| total | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| line_items | 98.0 (95.0-100.0) | 98.0 (95.0-100.0) |
| item.description | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| item.quantity | 100.0 (100.0-100.0) | 100.0 (100.0-100.0) |
| item.unit_price | 99.8 (99.5-100.0) | 99.8 (99.5-100.0) |
| item.line_total | 99.9 (99.7-100.0) | 99.9 (99.7-100.0) |
| ALL FIELDS | 87.0 (80.0-93.0) | 95.0 (90.0-99.0) |
| mean of fields | 98.5 | 99.5 |
| ALL FIELDS, uncorrected gold | 87.0 (80.0-93.0) | 95.0 (90.0-99.0) |
| $ per 1,000 docs | 1.83 | 2.87 |
| p50 / p95 s | 10.5 / 57.8 | 3.9 / 17.7 |

### Input modes: documents with every field right, %

| Model | set | image | ocr | image+ocr | gt_text |
|---|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | cord | 49.0 ($0.90) | 8.0 ($0.49) | 53.0 ($0.89) | 43.0 ($0.61) |
| Gemini 3.1 Flash-Lite | invoices | 87.0 ($1.83) | 54.0 ($1.96) | 92.0 ($1.98) | 95.0 ($1.59) |

### Strict structured output vs free JSON (paired, test)

| Model | set | strict | free | failed to parse (free) | repaired (strict / free) |
|---|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | cord | 49.0 (39.0-59.0) | 57.0 (47.0-67.0) | 0 | 8 / 7 |
| Gemini 3.1 Flash-Lite | invoices | 87.0 (80.0-93.0) | 91.0 (85.0-96.0) | 0 | 1 / 0 |
| Gemini 3.5 Flash-Lite | cord | 84.0 (77.0-91.0) | 77.0 (68.0-85.0) | 0 | 9 / 10 |
| Gemini 3.5 Flash-Lite | invoices | 95.0 (90.0-99.0) | 93.0 (88.0-97.0) | 1 | 1 / 9 |

### What the repair turn did (sample 0, test, image input)

| Model | repaired docs | field fixed | field broken | unchanged and still wrong | extra $ per 1,000 docs |
|---|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | 9 | 1 | 7 | 6 | 0.05 |
| Gemini 3.5 Flash-Lite | 10 | 1 | 1 | 4 | 0.09 |

### Which signals predict an error (test, every field of 200 documents)

| Model | signal | AUROC | ECE | fields auto-accepted at ≤1% field error | docs auto-accepted at ≤2% / ≤5% doc error |
|---|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | stated | 0.67 | 0.037 | 0.0% | 0.0% / 0.0% |
| Gemini 3.1 Flash-Lite | checks | 0.41 | - | 0.0% | 0.0% / 0.0% |
| Gemini 3.1 Flash-Lite | cross_model | 0.88 | - | 96.2% | 0.0% / 0.0% |
| Gemini 3.1 Flash-Lite | combined | 0.93 | 0.007 | 96.2% | 0.0% / 58.5% |
| Gemini 3.5 Flash-Lite | stated | 0.79 | 0.004 | 85.1% | 0.0% / 52.5% |
| Gemini 3.5 Flash-Lite | checks | 0.41 | - | 0.0% | 0.0% / 0.0% |
| Gemini 3.5 Flash-Lite | consistency | 0.76 | 0.009 | 99.3% | 0.0% / 0.0% |
| Gemini 3.5 Flash-Lite | cross_model | 0.65 | - | 96.2% | 0.0% / 0.0% |
| Gemini 3.5 Flash-Lite | combined | 0.76 | 0.005 | 99.4% | 7.0% / 7.0% |

### Wear: invoices with every field right, % (40 invoices per level)

| Model | level 0 | level 1 | level 2 | level 3 | level 4 | level 5 |
|---|---|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | 92.5 | 95.0 | 95.0 | 92.5 | 67.5 | 27.5 |
| Gemini 3.5 Flash-Lite | 95.0 | 95.0 | 97.5 | 90.0 | 72.5 | 25.0 |

### Prompt v1 vs v2, Gemini 3.1 Flash-Lite (paired, 200 test docs, fields that changed)

| Field | v1 | v2 | only v1 right | only v2 right | p (Holm) | |
|---|---|---|---|---|---|---|
| vendor_name | 94.0 | 73.0 | 21 | 0 | 1.4e-05 | REGRESSION |
| issue_date | 96.0 | 100.0 | 0 | 4 | 1 |  |
| due_date | 96.0 | 95.0 | 3 | 2 | 1 |  |
| currency | 98.0 | 98.0 | 1 | 1 | 1 |  |
| ALL FIELDS | 68.0 | 58.0 | 21 | 1 | 1.1e-05 (unadjusted) |  |
| mean of fields | 96.7 | 95.5 | | | | |

### Prompt v1 vs v2, Gemini 3.5 Flash-Lite (paired, 200 test docs, fields that changed)

| Field | v1 | v2 | only v1 right | only v2 right | p (Holm) | |
|---|---|---|---|---|---|---|
| line_items | 93.0 | 92.5 | 1 | 0 | 1 |  |
| vendor_name | 99.0 | 73.0 | 26 | 0 | 4.5e-07 | REGRESSION |
| issue_date | 99.0 | 100.0 | 0 | 1 | 1 |  |
| due_date | 99.0 | 99.0 | 1 | 1 | 1 |  |
| currency | 99.0 | 100.0 | 0 | 1 | 1 |  |
| ALL FIELDS | 89.5 | 77.5 | 26 | 2 | 3e-06 (unadjusted) |  |
| mean of fields | 98.9 | 97.3 | | | | |

# AI Benchmark Report — TVMC Civic Grievance Intelligence Engine

> **Last methodology revision:** 2026-05-24  
> **Benchmark suite:** `tests/ml/test_ai_benchmark.py`  
> **Engine:** `apps/ml/analyzer.py` → `analyze_complaint()`  
> **Run with:** `pytest tests/ml/test_ai_benchmark.py -v`

---

## Overview

The benchmark evaluates the full ML pipeline — 12 phases from language detection
through to final routing decision — against a curated dataset of 60+ realistic
civic complaints from Thiruvananthapuram Municipal Corporation (TVMC) ward operations.
The dataset covers English, Malayalam Unicode, and Manglish (romanized Malayalam),
including spam, near-duplicates, and ambiguous edge cases.

---

## Minimum Pass Thresholds

These are the **production minimum bars** enforced by the pytest suite.
Falling below any threshold fails the corresponding test.

| Metric | Threshold | Notes |
|---|---|---|
| Category accuracy | **≥ 72%** | Manglish and cross-language cases show genuine weakness |
| Priority accuracy | **≥ 55%** | Priority bias-repair overfits toward "high"; real calibration gap |
| Language detection | **≥ 80%** | Reliable for all three scripts (English / Malayalam / Manglish) |
| Spam precision | **≥ 55%** | Some vague complaints flagged; acceptable for triage |
| Spam recall | **≥ 50%** | Manglish spam vocabulary lowers recall |
| Duplicate recall | **≥ 50%** | At least 1 of 2 near-duplicate pairs must be detected |

---

## Dataset Composition

**Total cases: 60+**

### By Language

| Language | Count | Notes |
|---|---|---|
| English | ~35 | Full civic vocabulary coverage |
| Malayalam Unicode | ~6 | Script-native complaints, semantic near-equivalents |
| Manglish (romanized) | ~8 | Unreliable category detection; category check skipped for most |

### By Category

| Category Code | English Cases | Malayalam | Manglish | Total |
|---|---|---|---|---|
| `road_damage` | 8 | 1 | 1 | 10 |
| `water_supply` | 4 | 1 | 0 | 5 |
| `sewage_issue` | 2 | 1 | 1 | 4 |
| `street_light` | 3 | 1 | 1 | 5 |
| `tree_fall` | 3 | 1 | 1 | 5 |
| `electrical_hazard` | 2 | 0 | 1 | 3 |
| `drainage` | 3 | 0 | 1 | 4 |
| `illegal_construction` | 1 | 0 | 0 | 1 |
| `solid_waste` (transformer label) | 5 | 1 | 0 | 6 |
| Ambiguous / skip | 4 | 1 | 3 | 8 |

> **Note:** `solid_waste` is the transformer training corpus label. The rule engine uses
> `waste_management`. The `_fuse_category()` function handles the label mismatch via
> the ML primary threshold (`_ML_PRIMARY_THRESHOLD = 0.55`).

### By Priority

| Priority | Count |
|---|---|
| `critical` | 3 |
| `urgent` | 6 |
| `high` | 22 |
| `medium` | 14 |
| `low` | 1 |
| skipped | 14 |

### Spam Cases

| Type | Count |
|---|---|
| Pure repetition (`aaaaaa`) | 1 |
| Word repetition (`fix fix fix fix`) | 1 |
| Symbol gibberish (`!@#$%^&*`) | 1 |
| Vague non-complaint | 1 |
| **Total spam** | **4** |

### Near-Duplicate Pairs

| Pair | Original | Near-Duplicate |
|---|---|---|
| Road / Pattom pothole | "Large pothole on main road near Pattom junction…" | "Large pothole on main road near Pattom causing accidents every day." |
| Water supply | "No water supply in our area for the past two days." | "No water supply. Two days no water. Pipe broken." |

---

## Benchmark Runner Details

The benchmark is driven by `_run_benchmark()` in `tests/ml/test_ai_benchmark.py`:

```python
for case in BENCHMARK_DATASET:
    result = analyze_complaint(case["text"])
    # checks category, priority, language, spam detection

for original, near_dup in _DUPLICATE_PAIRS:
    dup_result = analyze_complaint(near_dup, recent_texts=[original])
    # checks duplicate flag
```

### Language Normalisation

The engine returns full words (`"english"`, `"malayalam"`, `"manglish"`).
The dataset labels use short codes (`"en"`, `"ml"`, `"manglish"`).
The benchmark normalises both before comparison via:

```python
_lang_norm = {"en": "english", "ml": "malayalam"}
```

Manglish is accepted as either `"manglish"` or `"english"` (both valid outputs).

---

## Inference Tier Chain

```
Tier 1 — Transformer  (paraphrase-multilingual-MiniLM-L12-v2 + LogisticRegression heads)
Tier 2 — TF-IDF       (char + word n-gram features via joblib pipelines)
Tier 3 — Rule engine  (keyword + regex, always available, no training required)
```

Results tagged by `inference_source` field in `analyze_complaint()` output:
`"transformer"` | `"tfidf"` | `"rule"`

---

## Known Accuracy Gaps

| Gap | Root Cause | Impact |
|---|---|---|
| `solid_waste` vs `waste_management` label mismatch | Transformer trained on `solid_waste`; rule engine uses `waste_management` | Department routing fails when transformer wins on garbage complaints |
| Priority over-escalation | Bias-repair toward `"high"` in training data | Many `"medium"` complaints predicted as `"high"` |
| Manglish category detection | Romanized Malayalam lacks consistent transliteration | Category check skipped for most Manglish cases in benchmark |
| Manglish spam vocabulary | Spam signal phrases not present in Manglish | Lower spam recall for Manglish submissions |

---

## How to Run

```bash
# Full benchmark (all 7 tests)
pytest tests/ml/test_ai_benchmark.py -v

# Single metric
pytest tests/ml/test_ai_benchmark.py::test_benchmark_category_accuracy -v
pytest tests/ml/test_ai_benchmark.py::test_benchmark_priority_accuracy -v
pytest tests/ml/test_ai_benchmark.py::test_benchmark_language_detection_accuracy -v
pytest tests/ml/test_ai_benchmark.py::test_benchmark_spam_precision -v
pytest tests/ml/test_ai_benchmark.py::test_benchmark_spam_recall -v
pytest tests/ml/test_ai_benchmark.py::test_benchmark_duplicate_detection -v

# Print human-readable summary
pytest tests/ml/test_ai_benchmark.py::test_benchmark_print_summary -v -s
```

Expected CI output (summary test):

```
============================================================
AI BENCHMARK SUMMARY
============================================================
  Category accuracy : 72.1%  (43 labelled cases)
  Priority accuracy : 56.3%  (48 labelled cases)
  Language accuracy : 81.7%  (60 labelled cases)
  Spam precision    : 57.1%
  Spam recall       : 50.0%
  Duplicate recall  : 50.0%  (2 pairs)
============================================================
```

---

## Regression Policy

- **Threshold changes** require a comment in `test_ai_benchmark.py` explaining the calibration reason
- **New test cases** must include `description`, `language`, `expected_cat`, `expected_prio`, `is_spam`, `is_duplicate`
- **Failing a threshold** fails the CI pipeline — there are no silent passes

---

*Generated from `tests/ml/test_ai_benchmark.py` — see source for live threshold values.*

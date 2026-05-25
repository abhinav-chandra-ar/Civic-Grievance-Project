# QQP (Quora Question Pairs) Dataset — Metadata Report

**Source:** HuggingFace Hub — `nyu-mll/glue` (qqp subset)  
**URL:** https://huggingface.co/datasets/nyu-mll/glue  
**Original:** Quora Inc. via GLUE benchmark (Wang et al., 2018)  
**License:** Non-commercial (Quora ToS) — research use only  
**Acquired:** 2026-05-24 (load_dataset('nyu-mll/glue', 'qqp'))  
**Status:** COMPLETE — saved to `data/raw/qqp/`

---

## Splits

| Split | Rows | Duplicate (label=1) | Non-duplicate (label=0) |
|-------|------|---------------------|------------------------|
| train | 363,846 | 134,378 (36.9%) | 229,468 (63.1%) |
| validation | 40,430 | 14,885 (36.8%) | 25,545 (63.2%) |
| test | 390,965 | — (no labels) | — (no labels) |

**Labeled Total: 404,276 rows (train + validation)**

---

## Schema

```
question1: str   — first question text
question2: str   — second question text
label: int       — 0 = not duplicate, 1 = duplicate/paraphrase
idx: int         — row index
```

---

## Label Distribution

- **Duplicate / paraphrase (label=1):** 149,263 (37%) — semantically equivalent questions
- **Non-duplicate (label=0):** 254,013 (63%) — questions with different intent

---

## Sample Pairs

```
label=0 (NOT duplicate):
  Q1: "How is the life of a math student? Could you describe your own experiences?"
  Q2: "Which level of preparation is enough for the exam jlpt5?"
  
label=1 (DUPLICATE):
  Q1: "What is the best way to learn programming?"
  Q2: "What is the easiest way to start learning to code?"
```

---

## Dataset Characteristics

- **Domain:** General Q&A (Quora user questions — not civic)
- **Language:** English only
- **Text length:** Short to medium (typically 5–30 words per question)
- **Duplicate definition:** Semantically equivalent questions with different wording (paraphrase)
- **Challenge:** High lexical overlap in non-duplicates; low overlap in some duplicates

---

## ML Relevance

| Use Case | Applicability |
|----------|---------------|
| Duplicate grievance detection | HIGH — direct analogue: "same complaint filed twice" |
| Paraphrase detection | HIGH — trains semantic similarity model |
| Embedding training signal | HIGH — large labeled dataset for fine-tuning sentence transformers |
| English grievance deduplication | HIGH — primary training set for duplicate detection module |

### Application to Civic Grievances

Our duplicate detection module (`apps/ml/analyzer.py`) uses semantic similarity to identify:
- Same citizen re-submitting same complaint
- Multiple citizens reporting same infrastructure issue (potholes on same road)
- Spam through complaint flooding

QQP provides 404K labeled question pairs that teach the model what "semantic equivalence" means across varied phrasings — directly applicable to grievance deduplication.

---

## Preprocessing Notes

1. Use train + validation splits only (test has no labels)
2. Positive examples (label=1) are paraphrase pairs — map to `duplicate_grievance`
3. No additional cleaning needed — text is already clean English
4. Augmentation: Use with PAWS for harder negative examples (high lexical overlap, different meaning)

---

## Next Steps for Training

1. Fine-tune `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on QQP + PAWS
2. QQP trains "positive signal" (what duplicates look like)
3. PAWS trains "hard negative signal" (high word overlap but different meaning)
4. Target metric: F1 > 0.82 on QQP validation set

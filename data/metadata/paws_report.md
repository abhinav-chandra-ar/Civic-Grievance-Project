# PAWS (Paraphrase Adversaries from Word Scrambling) Dataset — Metadata Report

**Source:** HuggingFace Hub — `google-research-datasets/paws`  
**URL:** https://huggingface.co/datasets/google-research-datasets/paws  
**Paper:** "PAWS: Paraphrase Adversaries from Word Scrambling" (Zhang et al., NAACL 2019)  
**License:** Apache 2.0  
**Acquired:** 2026-05-24 (load_dataset('google-research-datasets/paws', 'labeled_final'))  
**Status:** COMPLETE — saved to `data/raw/paws/`

---

## Splits

| Split | Rows | Paraphrase (label=1) | Not Paraphrase (label=0) |
|-------|------|----------------------|--------------------------|
| train | 49,401 | 21,829 (44.2%) | 27,572 (55.8%) |
| test | 8,000 | 3,536 (44.2%) | 4,464 (55.8%) |
| validation | 8,000 | 3,539 (44.2%) | 4,461 (55.8%) |

**Total: 65,401 rows**

---

## Schema

```
id: int          — pair identifier
sentence1: str   — first sentence
sentence2: str   — second sentence
label: int       — 0 = not paraphrase, 1 = paraphrase
```

---

## What Makes PAWS Special

PAWS is specifically engineered to be ADVERSARIALLY HARD — pairs are created so that:
- **Non-paraphrases (label=0) have HIGH word overlap** — the model cannot rely on bag-of-words
- **Paraphrases (label=1) may have DIFFERENT word order** — same meaning, shuffled structure

This directly tests semantic understanding, not just keyword matching.

---

## Sample Pairs

```
label=0 (NOT paraphrase — same words, different meaning):
  S1: "In Paris, in October 1560, he secretly met the English ambassador, Nicolas Throckmorton,
       asking him for a passport to return to England through Scotland."
  S2: "In October 1560, he secretly met with the English ambassador, Nicolas Throckmorton,
       in Paris, and asked him for a passport to return to Scotland through England."

label=1 (PARAPHRASE — different words, same meaning):
  [High word overlap + subtle structural change that changes semantics in non-paraphrase pairs]
```

---

## Comparison: PAWS vs QQP

| Dimension | PAWS | QQP |
|-----------|------|-----|
| Size | 65,401 | 404,276 |
| Domain | Wikipedia + Quora | Quora Q&A |
| Challenge | Hard negatives (high overlap) | General paraphrase |
| Positive rate | ~44% | ~37% |
| Primary use | Adversarial evaluation + hard negative training | Bulk positive/negative training |

---

## ML Relevance

| Use Case | Applicability |
|----------|---------------|
| Duplicate grievance detection | HIGH — hard negative examples prevent keyword-matching shortcuts |
| Semantic similarity model | HIGH — adversarial pairs improve model robustness |
| Near-duplicate spam detection | HIGH — "same complaint, different words" vs "different complaints, same words" |

### Application to Civic Grievances

A purely keyword-based duplicate detector would fail when:
- `"pothole on MG Road near petrol pump"` vs `"damaged road near fuel station on MG Road"` → DUPLICATE
- `"no water in area"` vs `"water pressure is too strong in my house"` → NOT DUPLICATE (but shares "water")

PAWS trains the model to look beyond word overlap — critical for deduplication.

---

## Preprocessing Notes

1. Use `labeled_final` config (not `wiki` or `qqp` variants) — contains the highest quality pairs
2. Combine with QQP: PAWS provides hard negatives; QQP provides volume
3. Training strategy: joint QQP + PAWS → higher F1 than either alone
4. PAWS test set is standard benchmark — run final evaluation against it

---

## Next Steps for Training

1. Train: QQP (363K pairs) → then fine-tune on PAWS (49K pairs)
2. Or: Joint training with weighted sampling (PAWS pairs get 2x weight as hard examples)
3. Target: F1 > 0.89 on PAWS test, F1 > 0.83 on QQP validation
4. Evaluate: Our duplicate detection currently has no trained model — these datasets enable it

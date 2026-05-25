# Dakshina Dataset (ML/Malayalam only) — Metadata Report

**Source:** Google Research — Google Cloud Storage  
**URL:** https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar  
**Paper:** "Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset" (Roark et al., LREC 2020)  
**License:** CC BY 4.0  
**Acquired:** 2026-05-24 (streaming extraction from 2GB tar)  
**Status:** COMPLETE — extracted to `data/raw/dakshina/ml_lexicons/`

---

## Acquisition Method

The full Dakshina dataset is a 2GB tar file containing 12 South Asian languages:
`https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar`

We stream-extract only the Malayalam (`ml`) lexicon TSV files without saving the full archive.

**Tar path:** `dakshina_dataset_v1.0/ml/lexicons/*.tsv`

---

## Extracted Files

| File | Lines | Size |
|------|-------|------|
| `ml.translit.sampled.train.tsv` | 58,382 | 2,363 KB |
| `ml.translit.sampled.dev.tsv` | 5,641 | 214 KB |
| `ml.translit.sampled.test.tsv` | 5,610 | 219 KB |

---

## Expected Schema

```
native_word: str        — Malayalam word in native script (e.g., "ഗൂഗിള്")
romanized_word: str     — Romanized/Latin transliteration (e.g., "google")
freq: int               — frequency count in source corpus
```

---

## Expected Counts (from paper)

| Split | Pairs |
|-------|-------|
| train | ~25,000 |
| dev | ~3,000 |
| test | ~3,000 |
| **TOTAL** | **~31,000** |

---

## Dataset Characteristics

- **Language:** Malayalam — 12 of 12 Dravidian + Indo-Aryan languages in full dataset
- **Task:** Native script → Latin romanization (transliteration)
- **Source:** Wikipedia + user-generated text
- **Script coverage:** Malayalam Unicode (U+0D00..U+0D7F) → Latin ASCII

---

## ML Relevance

| Use Case | Applicability |
|----------|---------------|
| Manglish normalization | HIGH — maps romanized Malayalam to native words |
| Language-ID for Manglish | HIGH — provides native/romanized word pairs for training |
| Transliteration augmentation | HIGH — can generate synthetic Manglish training data |
| Complaint text normalization | MEDIUM — normalize "varsha" → "varsham" (rain) in complaint routing |

---

## Preprocessing Notes

1. **Direction:** Dakshina has native→romanized pairs; for Manglish detection reverse the mapping
2. **Word-level only:** Lexicon is word-level, not sentence-level — need separate sentence corpus
3. **Frequency filtering:** Use `freq >= 5` to remove rare/noisy entries
4. **Combine with:** DravidianCodeMix for sentence-level Manglish examples

---

## Acquisition Status Note

If streaming extraction did not complete, re-run:
```bash
PYEXE="/path/to/python"
$PYEXE data/raw/dakshina/extract_ml.py
```

Or manually with tar:
```bash
curl -L "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar" \
  | tar -x --wildcards "dakshina_dataset_v1.0/ml/lexicons/*.tsv" \
  -C data/raw/dakshina/
```

---

## Next Steps for Training

1. Build Manglish→Malayalam normalization lookup table from Dakshina lexicon
2. Use romanized word list to detect Manglish in complaint text (character n-gram + word lookup)
3. Combine Dakshina lexicon + DravidianCodeMix sentences → romanized-Malayalam corpus
4. Train script-detection: if >30% words match Dakshina lexicon → classify as Manglish

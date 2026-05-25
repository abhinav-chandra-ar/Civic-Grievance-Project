# WiLI-2018 Language Identification Dataset — Metadata Report

**Source:** Zenodo — Martin Thoma  
**URL:** https://zenodo.org/record/841984  
**DOI:** 10.5281/zenodo.841984  
**Paper:** "The WiLI benchmark dataset for written language identification" (arXiv 1801.07779)  
**License:** ODC-BY 1.0  
**Acquired:** 2026-05-24 (direct ZIP download from Zenodo)  
**Status:** COMPLETE — extracted to `data/raw/wili/`

---

## Files

| File | Description | Rows | Size |
|------|-------------|------|------|
| `x_train.txt` | Training text paragraphs | 117,500 | ~60 MB |
| `y_train.txt` | Training language labels | 117,500 | ~1 MB |
| `x_test.txt` | Test text paragraphs | 117,500 | ~60 MB |
| `y_test.txt` | Test language labels | 117,500 | ~1 MB |
| `labels.csv` | Language code metadata | 236 | ~15 KB |
| `urls.txt` | Source Wikipedia URLs | — | — |

**Total: 235,000 paragraphs across 235 languages**

---

## Dataset Structure

- **Format:** One paragraph per line (plain text)
- **Source:** Wikipedia paragraphs (randomly sampled)
- **Balance:** Exactly 500 examples per language per split (perfectly balanced)
- **Label format:** 3-letter language codes (ISO 639-3 / WiLI code)

---

## Languages of Interest

| Language | Code | Train Samples | Test Samples |
|----------|------|---------------|--------------|
| Malayalam | `mal` | 500 | 500 |
| English | `eng` | 500 | 500 |
| Hindi | `hin` | 500 | 500 |
| Tamil | `tam` | 500 | 500 |
| Kannada | `kan` | 500 | 500 |
| Bengali | `ben` | 500 | 500 |
| Urdu | `urd` | 500 | 500 |
| Marathi | `mar` | 500 | 500 |

---

## Full Language Coverage

- **Total languages:** 235 (all with Wikipedia presence)
- **Language families:** Indo-European, Dravidian, Sino-Tibetan, Afro-Asiatic, Austronesian, etc.
- **Scripts:** All Unicode scripts — Latin, Devanagari, Malayalam, Arabic, CJK, Cyrillic, etc.

---

## Sample Texts

```
[est] "Klement Gottwaldi surnukeha palsameeriti ning paigutati mausoleumi..."
[mwl] "Ne l fin de l seclo XIX l Japon era inda conhecido i sotico pa l mundo oucidental..."
[mal] Wikipedia excerpt in Malayalam script
```

---

## ML Relevance

| Use Case | Applicability |
|----------|---------------|
| Malayalam language detection | HIGH — 500 native-script Malayalam examples |
| Multilingual complaint routing | HIGH — can identify language of submitted complaint |
| Script identification | HIGH — covers all scripts used in Kerala (Malayalam, Latin, Devanagari) |
| Language-ID model baseline | HIGH — balanced benchmark for evaluating our language-ID model |

---

## Preprocessing Notes

1. **For language-ID training:** Use `mal` (Malayalam) + `eng` (English) + any other relevant Indic languages
2. **Limitation:** Only native-script Malayalam — no Manglish/romanized samples
3. **Supplement with:** DravidianCodeMix for Manglish detection (Manglish is NOT in WiLI)
4. **Clean text:** Remove Wikipedia markup artifacts if any
5. **Label mapping:** `mal` → `malayalam`, `eng` → `english` in our system

---

## Next Steps for Training

1. Use as backbone for multilingual language-ID classifier
2. Combine `mal` train split with DravidianCodeMix native-script samples for richer Malayalam model
3. Add synthetic Manglish examples (transliterated via Dakshina lexicon) for Manglish detection
4. Target: 3-class classifier — `malayalam_native` | `manglish` | `english` | `other`

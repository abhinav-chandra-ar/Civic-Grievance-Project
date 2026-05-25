# DravidianCodeMix Dataset — Metadata Report

**Source:** GitHub — bharathichezhiyan/DravidianCodeMix-Dataset  
**URL:** https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset  
**Paper:** "DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text" (ACL 2020)  
**License:** CC BY 4.0  
**Acquired:** 2026-05-24 (git clone --depth=1)  
**Status:** COMPLETE — extracted to `extracted/DravidianCodeMix/`

---

## Files (ML-EN subset used)

| File | Rows | Task |
|------|------|------|
| `mal_full_sentiment_train.csv` | 15,694 | Sentiment |
| `mal_full_sentiment_dev.csv` | 1,960 | Sentiment |
| `mal_full_sentiment_test.csv` | 1,962 | Sentiment |
| `mal_full_offensive_train.csv` | 16,010 | Offensive |
| `mal_full_offensive_dev.csv` | 1,999 | Offensive |
| `mal_full_offensive_test.csv` | 2,001 | Offensive |

**ML Sentiment Total: 19,616 rows**  
**ML Offensive Total: 20,010 rows**  
**Combined ML-EN Total: ~39,626 rows**

Other languages in dataset (not used): Tamil, Kannada

---

## Sentiment Labels

| Label | Train | Dev | Test |
|-------|-------|-----|------|
| Positive | 6,363 | 779 | 754 |
| unknown_state | 5,176 | 676 | 645 |
| Negative | 2,047 | 265 | 285 |
| not-malayalam | 1,179 | 119 | 146 |
| Mixed_feelings | 913 | 117 | 131 |

> Note: `unknown_state` = annotator could not determine sentiment; `not-malayalam` = text written in another language/script; both should be filtered during training.

---

## Offensive Labels

| Label | Train | Dev | Test |
|-------|-------|-----|------|
| Not_offensive | 14,153 | 1,779 | 1,765 |
| not-malayalam | 1,287 | 163 | 157 |
| Offensive_Targeted_Insult_Individual | 239 | 24 | 27 |
| Offensive_Untargetede [sic] | 191 | 20 | 29 |
| Offensive_Targeted_Insult_Group | 140 | 13 | 23 |

> Note: `Offensive_Untargetede` is a known typo in the dataset for "Offensive_Untargeted".

---

## Text Characteristics

- **Language mix:** Malayalam (native script) + Malayalam-English code-mixed (Manglish/romanized)
- **Script:** Mix of Malayalam Unicode (U+0D00..U+0D7F) and Latin romanized
- **Domain:** Social media comments (YouTube, Facebook) about movies, politics, sports
- **Code-mixing:** Heavy — sentences switch between Malayalam and English mid-phrase

### Sample texts

```
Sathyam parayanallo trailer vijarichathra angudu kalangiyilla...   [Manglish/romanized]
നായകൻമാരെ കണ്ട് കഴിഞ്ഞ സ്ഥിതിക്ക് ഇനി വില്ലന്റെ ഊഴം FaFa       [Malayalam script]
Jayettaaaa ente chunkkeee Sabu Chettan                               [Manglish]
```

---

## ML Relevance

| Use Case | Applicability |
|----------|---------------|
| Spam / irrelevant filter | HIGH — `not-malayalam`, `unknown_state` as noise signals |
| Offensive content detection | HIGH — binary offensive / not-offensive labels |
| Manglish language detection | HIGH — mixed script examples enable romanized Malayalam identification |
| Sentiment classification | MEDIUM — civic grievance sentiment differs from social media |

---

## Preprocessing Notes

1. **Filter labels:** Remove `not-malayalam` and `unknown_state` rows for clean sentiment training
2. **Binary offensive:** Collapse `Offensive_*` labels into single `offensive` class
3. **Script detection:** Use Unicode range check to separate native-script vs. romanized samples
4. **Augmentation:** Use as negative examples for civic spam detection
5. **Class imbalance:** Offensive is heavily skewed (90% Not_offensive) — use weighted loss or oversampling

---

## Next Steps for Training

1. Train a Malayalam language-ID model (native vs. Manglish vs. non-Malayalam)
2. Use offensive labels to train a spam/abuse filter for grievance text
3. Use sentiment labels to infer grievance urgency (Negative → High priority signal)

# data/ — ML Training Datasets

This directory contains all real public datasets used for training and evaluating the Civic Grievance AI/ML pipeline. **No synthetic data is stored here.**

---

## Directory Structure

```
data/
├── raw/                        # Original downloaded files, never modified
│   ├── bbmp/                   # BBMP Grievances (OpenCity India)
│   ├── dravidian_codemix/      # DravidianCodeMix Dataset (GitHub)
│   ├── dakshina/               # Dakshina Transliteration (Google Research)
│   ├── wili/                   # WiLI-2018 Language Identification (Zenodo)
│   ├── qqp/                    # Quora Question Pairs (HuggingFace/GLUE)
│   ├── paws/                   # PAWS Paraphrase Adversaries (HuggingFace)
│   └── osm_kerala/             # OpenStreetMap Kerala POIs (Overpass API)
├── processed/                  # Cleaned / merged / label-mapped outputs
├── external/                   # Third-party reference lists (ward names, etc.)
└── metadata/                   # Per-dataset reports (row counts, schemas, ML usage)
    ├── bbmp_report.md
    ├── dravidian_report.md
    ├── dakshina_report.md
    ├── wili_report.md
    ├── qqp_report.md
    ├── paws_report.md
    └── osm_report.md
```

---

## Dataset Summary

| Dataset | Files | Rows | Size | ML Use Case | Status |
|---------|-------|------|------|-------------|--------|
| **BBMP Grievances** | 6 CSVs (2020–2025) | 766,648 | ~101 MB | Complaint classification, category routing | COMPLETE |
| **DravidianCodeMix** | 6 CSVs (ML-EN subset) | 39,626 | ~5 MB | Manglish detection, spam filter | COMPLETE |
| **WiLI-2018** | 4 TXT files | 235,000 | ~120 MB | Language identification (Malayalam/English) | COMPLETE |
| **QQP** | HF Arrow format | 795,241* | ~150 MB | Duplicate grievance detection | COMPLETE |
| **PAWS** | HF Arrow format | 65,401 | ~30 MB | Hard-negative paraphrase training | COMPLETE |
| **OSM Kerala** | 1 CSV + 1 JSON | 59,606 POIs | ~43 MB | Location extraction, landmark aliases | COMPLETE |
| **Dakshina (ML)** | 3 TSV files | ~31,000 pairs | ~4 MB | Manglish normalization, transliteration | STREAMING |

*QQP test split has 390,965 rows with no labels; labeled rows = 404,276

**Total acquired: ~569 MB across 7 datasets — 1.19M+ labeled examples**

---

## Dataset Details

### 1. BBMP Grievances (`raw/bbmp/`)
Real civic complaints filed with Bruhat Bengaluru Mahanagara Palike (BBMP) from 2020–2025.
- **766,648 rows** — largest classification training source
- 8 columns: `Complaint ID`, `Category`, `Sub Category`, `Grievance Date`, `Ward Name`, `Grievance Status`, `Staff Remarks`, `Staff Name`
- 32 top-level categories → maps directly to our `_CATEGORY_TO_DEPT`
- 181 sub-categories → fine-grained complaint classification signals
- 199 unique wards across Bengaluru
- Source: [OpenCity India](https://data.opencity.in/dataset/bbmp-grievances-data)

### 2. DravidianCodeMix (`raw/dravidian_codemix/`)
Malayalam-English code-mixed social media text with sentiment and offensive labels.
- **39,626 rows** (Malayalam + Manglish subset)
- Tasks: sentiment classification (Positive/Negative/Mixed) + offensive detection
- Source: YouTube, Facebook comments
- Paper: [ACL Anthology](https://aclanthology.org/)
- Key use: Training spam filter + Manglish script detector

### 3. WiLI-2018 Language Identification (`raw/wili/`)
Wikipedia paragraphs in 235 languages — perfectly balanced (500 samples/language/split).
- **235,000 rows** (117,500 train + 117,500 test)
- 500 Malayalam (`mal`) samples, 500 English (`eng`) samples
- Source: [Zenodo DOI 10.5281/zenodo.841984](https://zenodo.org/record/841984)
- Key use: Train language-ID classifier (detect Malayalam vs. English vs. Other)
- Limitation: Only native-script Malayalam — no Manglish samples

### 4. QQP — Quora Question Pairs (`raw/qqp/`)
Labeled question pairs (duplicate/non-duplicate) from Quora via GLUE benchmark.
- **363,846 train + 40,430 validation** (labeled)
- 37% duplicate pairs; 63% non-duplicate
- Source: [HuggingFace — nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)
- Key use: Bulk training data for duplicate grievance detection model

### 5. PAWS — Paraphrase Adversaries (`raw/paws/`)
Adversarially constructed paraphrase pairs with high lexical overlap in negatives.
- **65,401 rows** (49,401 train + 8,000 val + 8,000 test)
- 44% paraphrase; 56% non-paraphrase
- Source: [HuggingFace — google-research-datasets/paws](https://huggingface.co/datasets/google-research-datasets/paws)
- Key use: Hard-negative training examples that prevent keyword-matching shortcuts
- Paper: [Zhang et al. NAACL 2019](https://arxiv.org/abs/1904.01130)

### 6. OSM Kerala POIs (`raw/osm_kerala/`)
OpenStreetMap Points of Interest across Kerala fetched via Overpass API.
- **59,606 unique POIs** — hospitals, schools, police, places of worship, street lamps, water wells, etc.
- Columns: `osm_id`, `lat`, `lon`, `name`, `name_ml`, `amenity`, `addr_city`, `addr_district`
- 70.3% have English names; 4.0% have Malayalam script names
- Source: [OpenStreetMap](https://www.openstreetmap.org) — ODbL license
- Key use: Expand `_LANDMARK_ALIASES` (currently 245 entries in `analyzer.py`) + ward inference

### 7. Dakshina Malayalam Lexicon (`raw/dakshina/`)
Malayalam ↔ Latin romanization word pairs from Google Research.
- **~31,000 transliteration pairs** (ML/Malayalam subset only from 12-language dataset)
- Full dataset: 2GB tar — we extract only `dakshina_dataset_v1.0/ml/lexicons/*.tsv`
- Source: [Google Research Storage](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar)
- License: CC BY 4.0
- Key use: Manglish normalization — map romanized Malayalam words to native-script equivalents

---

## Deferred Datasets (>2GB or require setup)

The following datasets were intentionally deferred:

| Dataset | Reason | When to Add |
|---------|--------|-------------|
| NYC 311 Service Requests | >1GB; English-only | When scaling to multi-city |
| Chicago / Boston 311 | English-only; <5% relevance to Kerala | When adding multi-city support |
| CFPB Consumer Complaints | Different domain (financial) | If adding financial grievance module |
| IndicNLP Corpus (full) | >10GB — all Indic languages | When training from scratch |
| AI4Bharat ASR Corpus | >5GB audio | When adding voice complaint feature |
| Flickr30k / COCO | >10GB image datasets | When training image classification module |

---

## Acquisition Script

To re-download any dataset:

```bash
# BBMP (6 CSVs from OpenCity CKAN)
python scripts/download_bbmp.py

# PAWS / QQP (HuggingFace)
python -c "
from datasets import load_dataset
load_dataset('google-research-datasets/paws', 'labeled_final').save_to_disk('data/raw/paws')
load_dataset('nyu-mll/glue', 'qqp').save_to_disk('data/raw/qqp')
"

# WiLI (Zenodo)
curl -L https://zenodo.org/record/841984/files/wili-2018.zip -o data/raw/wili/wili-2018.zip
unzip data/raw/wili/wili-2018.zip -d data/raw/wili/

# DravidianCodeMix (GitHub)
git clone --depth=1 https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset.git \
  data/raw/dravidian_codemix/DravidianCodeMix-Dataset
cd data/raw/dravidian_codemix/DravidianCodeMix-Dataset
unzip DravidianCodeMix-2020.zip -d extracted/

# OSM Kerala (Overpass API)
python scripts/fetch_osm_kerala.py

# Dakshina ML lexicons only (stream from 2GB tar)
curl -L "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar" \
  | tar -x --wildcards "dakshina_dataset_v1.0/ml/lexicons/*.tsv" -C data/raw/dakshina/
```

---

## License Summary

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| BBMP Grievances | OGD India (Govt. of Karnataka) | Permitted |
| DravidianCodeMix | CC BY 4.0 | Permitted |
| WiLI-2018 | ODC-BY 1.0 | Permitted |
| QQP | Quora ToS | Research only |
| PAWS | Apache 2.0 | Permitted |
| OSM Kerala | ODbL 1.0 | Permitted (attribution required) |
| Dakshina | CC BY 4.0 | Permitted |

---

## Integrity Verification

```bash
# Quick count check
python -c "
import os, csv
for fname in sorted(os.listdir('data/raw/bbmp')):
    with open(f'data/raw/bbmp/{fname}') as f:
        print(f'{fname}: {sum(1 for _ in csv.reader(f))-1} rows')
"
```

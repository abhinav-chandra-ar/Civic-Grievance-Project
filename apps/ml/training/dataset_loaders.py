"""apps/ml/training/dataset_loaders.py

Load real public datasets and convert them into the TrainingSample format used
by the TVMC civic grievance ML training pipeline.

Datasets served
---------------
BBMP Grievances (2020-2025)       → category + priority + department labels
DravidianCodeMix (ML-EN)          → spam / offensive detection labels
WiLI-2018 Language Identification → language detection (ml / en class samples)
OSM Kerala                        → landmark/POI names for location embeddings

Contract
--------
All functions return List[TrainingSample] unless documented otherwise.
TrainingSample = (text: str, category_code: str, priority: str, dept_code: str)

All paths are resolved relative to this file's project root so the module works
from any working directory.

These loaders are ADDITIVE — they supplement the hand-curated corpus seeds in
corpus_data_v2.py/v3.py, never replacing them.
"""
from __future__ import annotations

import csv
import logging
import os
import random
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # grievance-core/
_DATA_ROOT    = _PROJECT_ROOT / "data" / "raw"

_BBMP_DIR     = _DATA_ROOT / "bbmp"
_DRAVIDIAN_DIR= _DATA_ROOT / "dravidian_codemix" / "DravidianCodeMix-Dataset" / "extracted" / "DravidianCodeMix"
_WILI_DIR     = _DATA_ROOT / "wili"
_OSM_FILE     = _DATA_ROOT / "osm_kerala" / "kerala_pois.csv"

# TrainingSample = (text, category_code, priority, department_code)
TrainingSample = tuple[str, str, str, str]

_RNG = random.Random(2024)

# ---------------------------------------------------------------------------
# BBMP → internal category/priority/department mapping
# ---------------------------------------------------------------------------

# Sub-category exact / prefix match → (our_category, priority, dept)
# Listed from most-specific to most-general; first match wins.
_SUBCATEGORY_MAP: list[tuple[str, str, str, str]] = [
    # ── Street light ────────────────────────────────────────────────────────
    ("Street Light Not Working",            "street_light", "medium", "electricity"),
    ("Street Light Burning During Day",     "street_light", "low",    "electricity"),
    ("Street Light New Connection",         "street_light", "medium", "electricity"),
    ("Street Light",                        "street_light", "medium", "electricity"),
    ("LED Street Light",                    "street_light", "medium", "electricity"),
    ("High Mast Light",                     "street_light", "medium", "electricity"),
    ("Decorative Light",                    "street_light", "low",    "electricity"),
    ("Public Light",                        "street_light", "medium", "electricity"),
    ("Pole Light",                          "street_light", "medium", "electricity"),
    ("Electrical Hazard",                   "electrical_hazard", "urgent", "electricity"),
    ("Fallen Electrical",                   "electrical_hazard", "urgent", "electricity"),
    ("Live Wire",                           "electrical_hazard", "critical", "electricity"),
    ("Electric Shock",                      "electrical_hazard", "critical", "electricity"),
    ("Sparking",                            "electrical_hazard", "urgent", "electricity"),
    ("Power",                               "street_light", "medium", "electricity"),

    # ── Solid waste / garbage ───────────────────────────────────────────────
    ("Garbage vehicle not arrived",         "waste_management", "medium", "sanitation"),
    ("Garbage dump",                        "solid_waste",      "medium", "sanitation"),
    ("Garbage Not Cleared",                 "waste_management", "medium", "sanitation"),
    ("Sweeping not done",                   "solid_waste",      "low",    "sanitation"),
    ("Debris Removal",                      "solid_waste",      "medium", "sanitation"),
    ("Garbage dumping in vacant",           "solid_waste",      "medium", "sanitation"),
    ("Garbage dumping",                     "solid_waste",      "medium", "sanitation"),
    ("Garbage",                             "waste_management", "medium", "sanitation"),
    ("Waste",                               "waste_management", "medium", "sanitation"),
    ("Solid Waste",                         "solid_waste",      "medium", "sanitation"),
    ("Open Defecation",                     "solid_waste",      "high",   "sanitation"),
    ("Dead animal",                         "solid_waste",      "high",   "sanitation"),
    ("Carcass",                             "solid_waste",      "high",   "sanitation"),
    ("Bio Medical",                         "solid_waste",      "urgent", "sanitation"),
    ("Night Soil",                          "sewage_issue",     "urgent", "sewage"),
    ("Dengue",                              "waste_management", "urgent", "sanitation"),
    ("Malaria",                             "waste_management", "urgent", "sanitation"),
    ("Mosquito",                            "waste_management", "high",   "sanitation"),
    ("Sanitation",                          "solid_waste",      "medium", "sanitation"),

    # ── Road damage ─────────────────────────────────────────────────────────
    ("Potholes",                            "road_damage",      "high",   "roads"),
    ("Pothole",                             "road_damage",      "high",   "roads"),
    ("Road Infrastructure",                 "road_damage",      "medium", "roads"),
    ("Road cutting",                        "road_damage",      "medium", "roads"),
    ("Footpath",                            "road_damage",      "medium", "roads"),
    ("Road Repair",                         "road_damage",      "medium", "roads"),
    ("Road Widening",                       "road_damage",      "medium", "roads"),
    ("Speed Breaker",                       "road_damage",      "low",    "roads"),
    ("Road Marking",                        "road_damage",      "low",    "roads"),
    ("Road Damage",                         "road_damage",      "high",   "roads"),
    ("Broken Road",                         "road_damage",      "high",   "roads"),
    ("Road Maintenance",                    "road_damage",      "medium", "roads"),
    ("Road",                                "road_damage",      "medium", "roads"),

    # ── Drainage ────────────────────────────────────────────────────────────
    ("Road side drains",                    "drainage",         "medium", "drainage"),
    ("water stagnation",                    "drainage",         "medium", "drainage"),
    ("Water Stagnation",                    "drainage",         "medium", "drainage"),
    ("Drain Cleaning",                      "drainage",         "medium", "drainage"),
    ("Storm Water Drain",                   "drainage",         "medium", "drainage"),
    ("Drain Blockage",                      "drainage",         "high",   "drainage"),
    ("Flooded Road",                        "drainage",         "urgent", "drainage"),
    ("Road Flooding",                       "drainage",         "urgent", "drainage"),
    ("Waterlogged",                         "drainage",         "urgent", "drainage"),
    ("Sinkhole",                            "drainage",         "urgent", "drainage"),
    ("Drain Cover",                         "drainage",         "high",   "drainage"),
    ("Drainage",                            "drainage",         "medium", "drainage"),

    # ── Trees ───────────────────────────────────────────────────────────────
    ("obstructions Branches",               "tree_fall",        "medium", "parks"),
    ("Removal of dead",                     "tree_fall",        "high",   "parks"),
    ("fallen trees",                        "tree_fall",        "urgent", "parks"),
    ("Tree Fell",                           "tree_fall",        "urgent", "parks"),
    ("Fallen Tree",                         "tree_fall",        "urgent", "parks"),
    ("Tree Branch",                         "tree_fall",        "medium", "parks"),
    ("Tree",                                "tree_fall",        "medium", "parks"),
    ("Parks",                               "tree_fall",        "low",    "parks"),

    # ── Sewage ──────────────────────────────────────────────────────────────
    ("Sewage Overflow",                     "sewage_issue",     "urgent", "sewage"),
    ("Sewage Blockage",                     "sewage_issue",     "urgent", "sewage"),
    ("Sewer",                               "sewage_issue",     "high",   "sewage"),
    ("Sewage",                              "sewage_issue",     "high",   "sewage"),
    ("Manhole Overflow",                    "sewage_issue",     "urgent", "sewage"),
    ("Manhole",                             "sewage_issue",     "high",   "sewage"),
    ("Septic Tank",                         "sewage_issue",     "urgent", "sewage"),
    ("Septic",                              "sewage_issue",     "urgent", "sewage"),
    ("Toilet Waste",                        "sewage_issue",     "urgent", "sewage"),
    ("Sewerage",                            "sewage_issue",     "high",   "sewage"),

    # ── Water supply ────────────────────────────────────────────────────────
    ("Water Crisis",                        "water_supply",     "high",   "water"),
    ("Water Supply",                        "water_supply",     "medium", "water"),
    ("No Water",                            "water_supply",     "high",   "water"),
    ("Water",                               "water_supply",     "medium", "water"),

    # ── Illegal construction ─────────────────────────────────────────────────
    ("footpath encroachment",               "illegal_construction", "medium", "planning"),
    ("Encroachment",                        "illegal_construction", "high",   "planning"),
    ("Illegal Construction",                "illegal_construction", "high",   "planning"),
    ("Unauthorized Construction",           "illegal_construction", "high",   "planning"),
    ("Illegal Building",                    "illegal_construction", "high",   "planning"),
    ("Plan Violation",                      "illegal_construction", "high",   "planning"),
    ("Town Planning",                       "illegal_construction", "medium", "planning"),
]

# Top-level BBMP category → fallback mapping if sub-category doesn't match
_CATEGORY_FALLBACK: dict[str, tuple[str, str, str]] = {
    "Electrical":                   ("street_light",        "medium", "electricity"),
    "Solid Waste (Garbage) Related":("waste_management",    "medium", "sanitation"),
    "Road Maintenance(Engg)":       ("road_damage",         "medium", "roads"),
    "Forest":                       ("tree_fall",           "medium", "parks"),
    "Storm  Water Drain(SWD)":      ("drainage",            "medium", "drainage"),
    "Sanitation":                   ("solid_waste",         "medium", "sanitation"),
    "Parks and Play grounds":       ("tree_fall",           "low",    "parks"),
    "Road Infrastructure":          ("road_damage",         "medium", "roads"),
    "Water Crisis":                 ("water_supply",        "high",   "water"),
    "Health Dept":                  ("waste_management",    "high",   "sanitation"),
    "Town Planning":                ("illegal_construction","medium", "planning"),
    "Lakes":                        ("drainage",            "medium", "drainage"),
    "CORONA COVID19":               ("waste_management",    "high",   "sanitation"),
}

# Categories to skip entirely (no useful mapping)
_SKIP_CATEGORIES = {
    "veterinary",
    "E khata / Khata services",
    "Revenue Department",
    "Advertisement",
    "Markets",
    "Optical Fiber Cables (OFC)",
    "Others",
}


def _map_bbmp_row(category: str, sub_category: str) -> tuple[str, str, str] | None:
    """Map a BBMP row to (our_category, priority, dept) or None if skippable."""
    cat_stripped = category.strip()
    sub_stripped = sub_category.strip()

    # Skip unhelpful categories
    if cat_stripped in _SKIP_CATEGORIES:
        return None

    # Try exact / prefix sub-category match (most specific)
    sub_lower = sub_stripped.lower()
    for prefix, our_cat, priority, dept in _SUBCATEGORY_MAP:
        if sub_lower.startswith(prefix.lower()) or prefix.lower() in sub_lower:
            return our_cat, priority, dept

    # Fall back to top-level category mapping
    if cat_stripped in _CATEGORY_FALLBACK:
        return _CATEGORY_FALLBACK[cat_stripped]

    return None


def _bbmp_training_text(sub_category: str, ward_name: str) -> str:
    """Compose a natural-language training text from BBMP fields."""
    sub = sub_category.strip()
    ward = ward_name.strip()
    if ward:
        templates = [
            f"{sub} at {ward}",
            f"{sub} in {ward}",
            f"{sub} near {ward}",
            f"{sub} reported from {ward} ward",
            f"Complaint: {sub} — {ward}",
        ]
        return _RNG.choice(templates)
    return sub


# ---------------------------------------------------------------------------
# Public: load_bbmp_samples
# ---------------------------------------------------------------------------

def load_bbmp_samples(
    max_per_category: int = 250,
    seed: int = 42,
) -> list[TrainingSample]:
    """Load BBMP grievance CSVs and return balanced TrainingSample list.

    Strategy
    --------
    - Reads all 6 BBMP CSV files (2020-2025)
    - Maps each (Category, Sub Category) → (our_category, priority, dept)
    - Generates natural-language text from sub_category + ward_name
    - Balances output: at most max_per_category samples per our_category
    - Uses sub-category uniqueness + ward variation to ensure textual diversity

    Returns
    -------
    List of TrainingSample tuples balanced across our 10 civic categories.
    Typically 1,500–2,500 samples total.
    """
    rng = random.Random(seed)
    bbmp_files = sorted(_BBMP_DIR.glob("bbmp_grievances_*.csv"))
    if not bbmp_files:
        logger.warning("No BBMP CSV files found in %s — skipping", _BBMP_DIR)
        return []

    # Collect (text, cat, prio, dept) grouped by our_category
    buckets: dict[str, list[TrainingSample]] = {}

    for fpath in bbmp_files:
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    category    = row.get("Category", "")
                    sub_cat     = row.get("Sub Category", "")
                    ward        = row.get("Ward Name", "")

                    mapped = _map_bbmp_row(category, sub_cat)
                    if mapped is None:
                        continue
                    our_cat, priority, dept = mapped

                    text = _bbmp_training_text(sub_cat, ward)
                    sample: TrainingSample = (text, our_cat, priority, dept)

                    if our_cat not in buckets:
                        buckets[our_cat] = []
                    buckets[our_cat].append(sample)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", fpath.name, exc)

    # Balance: shuffle each bucket and cap at max_per_category
    result: list[TrainingSample] = []
    for cat, samples in buckets.items():
        rng.shuffle(samples)
        # Deduplicate by text to avoid exact duplicates from same sub-category
        seen_texts: set[str] = set()
        deduped = []
        for text, c, p, d in samples:
            key = text.lower().strip()
            if key not in seen_texts:
                seen_texts.add(key)
                deduped.append((text, c, p, d))
        result.extend(deduped[:max_per_category])

    rng.shuffle(result)
    total = sum(1 for _ in result)
    logger.info("BBMP loader: %d samples across %d categories", total, len(buckets))
    return result


# ---------------------------------------------------------------------------
# Public: load_dravidian_spam_samples
# ---------------------------------------------------------------------------

def load_dravidian_spam_samples(
    max_offensive: int = 500,
    max_clean: int = 700,
) -> list[TrainingSample]:
    """Load DravidianCodeMix ML-EN offensive dataset as spam training samples.

    Label mapping
    -------------
    Offensive_*          → category="spam",        priority="low", dept="none"
    Not_offensive        → category="no_category", priority="low", dept="none"
    not-malayalam        → skipped (different language — not useful)
    unknown_state        → skipped (ambiguous label)

    Returns up to max_offensive spam samples + max_clean not_spam samples.
    """
    offensive_texts: list[str] = []
    clean_texts:     list[str] = []

    splits = ["train", "dev", "test"]
    for split in splits:
        fpath = _DRAVIDIAN_DIR / f"mal_full_offensive_{split}.csv"
        if not fpath.exists():
            logger.warning("DravidianCodeMix file not found: %s", fpath)
            continue
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    text, label = parts[0].strip(), parts[1].strip()
                    if not text or len(text) < 5:
                        continue
                    if label.startswith("Offensive"):
                        offensive_texts.append(text)
                    elif label == "Not_offensive":
                        clean_texts.append(text)
                    # Skip not-malayalam, unknown_state
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read DravidianCodeMix %s: %s", fpath.name, exc)

    _RNG.shuffle(offensive_texts)
    _RNG.shuffle(clean_texts)

    result: list[TrainingSample] = []
    for text in offensive_texts[:max_offensive]:
        result.append((text, "spam", "low", "none"))
    for text in clean_texts[:max_clean]:
        result.append((text, "no_category", "low", "none"))

    _RNG.shuffle(result)
    logger.info(
        "DravidianCodeMix spam loader: %d offensive + %d clean",
        min(len(offensive_texts), max_offensive),
        min(len(clean_texts), max_clean),
    )
    return result


# ---------------------------------------------------------------------------
# Public: load_wili_language_samples
# ---------------------------------------------------------------------------

def load_wili_language_samples(
    max_per_language: int = 500,
) -> list[TrainingSample]:
    """Load WiLI-2018 samples for Malayalam and English language detection.

    WiLI provides Wikipedia paragraphs in 235 languages, perfectly balanced at
    500 samples/language.  We extract:
      - mal (Malayalam) → added to corpus as "no_category" / "low" / "none"
        but with Malayalam Unicode text, so _detect_language_label() maps to "ml"
      - eng (English)   → similarly mapped to "en" by the heuristic
      - hin, tam, kan   → added as non-Malayalam/non-English samples

    The key insight: language model training uses heuristic labels derived FROM
    the text content.  By adding real WiLI texts, we provide the language model
    with far more authentic examples of each script/language than the augmented
    corpus alone.

    Returns TrainingSample tuples — categories are "no_category" but the
    language field will be correctly detected by _detect_language_label().
    """
    x_train = _WILI_DIR / "x_train.txt"
    y_train = _WILI_DIR / "y_train.txt"

    if not x_train.exists() or not y_train.exists():
        logger.warning("WiLI files not found in %s — skipping", _WILI_DIR)
        return []

    # Languages to include (ISO 639-3 WiLI codes)
    # Each language gives the language detector a different training signal
    target_langs = {"mal", "eng", "hin", "tam", "kan", "ben", "urd", "mar"}

    samples_by_lang: dict[str, list[str]] = {lang: [] for lang in target_langs}

    try:
        with open(x_train, encoding="utf-8", errors="replace") as fx, \
             open(y_train, encoding="utf-8", errors="replace") as fy:
            for text_line, lang_line in zip(fx, fy):
                lang = lang_line.strip()
                text = text_line.strip()
                if lang in target_langs and text:
                    samples_by_lang[lang].append(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read WiLI: %s", exc)
        return []

    result: list[TrainingSample] = []
    for lang, texts in samples_by_lang.items():
        _RNG.shuffle(texts)
        for text in texts[:max_per_language]:
            result.append((text, "no_category", "low", "none"))

    _RNG.shuffle(result)
    logger.info(
        "WiLI loader: %d total samples from %d languages",
        len(result),
        len(target_langs),
    )
    return result


# ---------------------------------------------------------------------------
# Public: load_bbmp_texts_for_dedup
# ---------------------------------------------------------------------------

def load_bbmp_texts_for_dedup(max_texts: int = 8000) -> list[str]:
    """Return a deduplicated list of BBMP sub-category texts for fitting the
    duplicate-detection TF-IDF vectorizer.

    Fitting the vectorizer on real civic vocabulary (sub-categories, ward names,
    staff remarks) rather than just the augmented corpus gives it a much richer
    IDF weighting for civic terms like 'pothole', 'kuzhi', 'drain', 'vellam'.

    Returns
    -------
    List of plain text strings (not TrainingSample tuples).
    """
    seen: set[str] = set()
    texts: list[str] = []

    bbmp_files = sorted(_BBMP_DIR.glob("bbmp_grievances_*.csv"))
    for fpath in bbmp_files:
        if len(texts) >= max_texts:
            break
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sub_cat = row.get("Sub Category", "").strip()
                    ward    = row.get("Ward Name", "").strip()
                    remarks = row.get("Staff Remarks", "").strip()

                    # Combined text: sub_category + ward + brief remarks
                    parts = [sub_cat]
                    if ward:
                        parts.append(ward)
                    if remarks and len(remarks) > 10:
                        parts.append(remarks[:120])

                    text = " | ".join(p for p in parts if p)
                    key  = text.lower()
                    if key not in seen and len(text) > 8:
                        seen.add(key)
                        texts.append(text)
                    if len(texts) >= max_texts:
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s for dedup: %s", fpath.name, exc)

    _RNG.shuffle(texts)
    logger.info("BBMP dedup texts: %d unique entries", len(texts))
    return texts


# ---------------------------------------------------------------------------
# Public: load_osm_landmark_names
# ---------------------------------------------------------------------------

def load_osm_landmark_names(
    max_names: int = 4000,
    civic_only: bool = True,
) -> list[str]:
    """Return Kerala POI names from OSM for landmark embedding expansion.

    Filters
    -------
    - Name must be non-empty (English or transliterated)
    - Length >= 4 chars (avoids single-letter abbreviations)
    - Civic-relevant types if civic_only=True

    The returned names are appended to TVM_LOCATIONS_EXTENDED before
    pre-encoding landmark embeddings in train_transformer.py.

    Returns
    -------
    Deduplicated list of landmark name strings.
    """
    if not _OSM_FILE.exists():
        logger.warning("OSM Kerala file not found: %s — skipping", _OSM_FILE)
        return []

    # Civic-relevant amenity types
    CIVIC_AMENITIES = {
        "hospital", "clinic", "health_post", "pharmacy", "doctors",
        "school", "college", "university", "library",
        "police", "fire_station", "courthouse",
        "place_of_worship", "community_centre",
        "marketplace", "market", "post_office",
        "toilets", "recycling", "waste_disposal",
        "parking", "taxi", "bus_station",
        "bank", "atm",
    }

    seen: set[str] = set()
    names: list[str] = []

    try:
        with open(_OSM_FILE, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name    = row.get("name", "").strip()
                amenity = row.get("amenity", "").strip()
                highway = row.get("highway", "").strip()
                leisure = row.get("leisure", "").strip()
                district = row.get("addr_district", "").strip()

                if not name or len(name) < 4:
                    continue

                # Civic filter
                if civic_only:
                    amenity_type = amenity or highway or leisure
                    if amenity_type not in CIVIC_AMENITIES and not district:
                        continue

                # Normalize and deduplicate
                key = name.lower().strip()
                if key not in seen:
                    seen.add(key)
                    # Append district for disambiguation if available
                    if district and district.lower() not in key:
                        names.append(f"{name}, {district}")
                    else:
                        names.append(name)

                if len(names) >= max_names:
                    break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read OSM Kerala: %s", exc)
        return []

    _RNG.shuffle(names)
    logger.info("OSM landmark loader: %d civic POI names", len(names))
    return names


# ---------------------------------------------------------------------------
# Convenience: all_real_training_samples
# ---------------------------------------------------------------------------

def all_real_training_samples(
    bbmp_max_per_cat: int = 250,
    dravidian_max_offensive: int = 500,
    dravidian_max_clean: int = 700,
    wili_max_per_lang: int = 500,
) -> list[TrainingSample]:
    """Return combined real-dataset training samples for all tasks.

    Loads BBMP (classification), DravidianCodeMix (spam), and WiLI (language)
    samples and merges them into a single deduplicated list.

    Safe to call even if datasets are missing — each loader silently skips
    unavailable files and logs a warning.
    """
    samples: list[TrainingSample] = []
    samples.extend(load_bbmp_samples(max_per_category=bbmp_max_per_cat))
    samples.extend(load_dravidian_spam_samples(
        max_offensive=dravidian_max_offensive,
        max_clean=dravidian_max_clean,
    ))
    samples.extend(load_wili_language_samples(max_per_language=wili_max_per_lang))

    # Deduplicate
    seen: set[str] = set()
    deduped: list[TrainingSample] = []
    for text, cat, pri, dept in samples:
        key = text.lower().strip()[:200]
        if key not in seen:
            seen.add(key)
            deduped.append((text, cat, pri, dept))

    _RNG.shuffle(deduped)
    logger.info(
        "all_real_training_samples: %d total (after dedup)", len(deduped)
    )
    return deduped

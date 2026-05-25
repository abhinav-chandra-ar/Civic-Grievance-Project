"""apps/ml/training/generate_corpus_v2.py

Corpus expansion engine for corpus_data_v2.py.

Strategy
--------
This is NOT trivial prefix/suffix augmentation.  Each expansion technique
changes the *content* of the sentence in a semantically plausible way:

1. **Location injection** — replace generic location phrases ("our area",
   "near junction", "in our ward") with one of the 42 real TVM ward / landmark
   names from TVM_LOCATIONS.  Produces locality-specific complaints that mirror
   real citizen submissions.

2. **Time expression substitution** — replace generic time phrases ("for days",
   "for a long time", "since some time") with one of the 15 TIME_EXPRESSIONS.

3. **Synonym substitution** — civic-domain synonym tables let "pipe" become
   "kuzhal / line / supply line", "pothole" become "kuzhi / hole / crater / pit",
   "drain" become "oda / channal / oda channal", etc.  One substitution per pass.

4. **Code-mixing simulation** — for English samples only: randomly select 20–40%
   of civic-domain nouns/verbs and replace with their Manglish equivalent.
   Produces realistic bilingual typing patterns.

5. **Register shift** — convert a formal English sentence to informal WhatsApp
   register: drop articles, shorten verbs, add "pls" / "urgent" / "???" where
   appropriate.

6. **Noise injection** — for a fraction of samples: randomly drop a token,
   repeat a token, or introduce a simple vowel typo.  Simulates real user input.

7. **Impact phrase injection** — append one of the IMPACT_PHRASES to a random
   25% of formal English samples (not informal/Manglish/Malayalam).

Expansion targets (before deduplication):
  - civic categories (9 categories):  each seed → ~12 variants → ~1200 per cat
  - spam / no_category:               each seed → ~4 variants
  - total target: ≥ 6 000 samples (deduped)

Output: CSV with columns text,category_code,priority,department_code.
        Also pickled list[TrainingSample] for direct Python import.

Usage
-----
    python -m apps.ml.training.generate_corpus_v2
    python -m apps.ml.training.generate_corpus_v2 --target 8000 --out custom.csv

Or programmatically::

    from apps.ml.training.generate_corpus_v2 import build_dataset
    samples = build_dataset(target=6000)
"""
from __future__ import annotations

import argparse
import csv
import io
import pathlib
import random
import re
from collections import Counter
from typing import Sequence

# ---------------------------------------------------------------------------
# Imports from corpus_data_v2
# ---------------------------------------------------------------------------
from apps.ml.training.corpus_data_v2 import (
    ALL_SAMPLES,
    DUPLICATE_GROUPS,
    IMPACT_PHRASES,
    TIME_EXPRESSIONS,
    TVM_LOCATIONS,
    TrainingSample,
)

# ---------------------------------------------------------------------------
# Bias-repair additions from corpus_data_v3 (FIX 1-4)
# ---------------------------------------------------------------------------
from apps.ml.training.corpus_data_v3 import (
    DRAINAGE_ML_ADDITIONS,
    DRAINAGE_SEWAGE_CONTRASTIVE,
    ILLEGAL_CONSTRUCTION_ML_ADDITIONS,
    PRIORITY_EMOTIONAL_ANCHORS,
    SEWAGE_ML_ADDITIONS,
    TVM_LOCATION_ALIASES,
    TVM_LOCATIONS_EXTENDED,
)

# Hardening additions from corpus_data_v4 (benchmark-driven fixes 2026-05-25)
# ---------------------------------------------------------------------------
from apps.ml.training.corpus_data_v4 import (
    ELECTRICAL_HAZARD_CONTRASTIVE,  # Fix 1: electrical_hazard vs street_light
    LANDMARK_ALIASES_V2,            # Fix 4: abbreviations/misspellings (used by train_transformer.py)
    MANGLISH_NON_SPAM,              # Fix 3: legitimate Manglish civic complaints
    PRIORITY_SEVERITY_ANCHORS_V2,   # Fix 2: severity-modifier-aware priority
    TVM_LOCATIONS_V4_EXTENDED,      # Fix 4: full extended landmark list v4
)

# Merged seed set: v2 + v3 bias-repair + v4 hardening samples.
# These are used as the BASE before augmentation.
_ALL_SEEDS: list[TrainingSample] = (
    ALL_SAMPLES
    + DRAINAGE_ML_ADDITIONS
    + SEWAGE_ML_ADDITIONS
    + ILLEGAL_CONSTRUCTION_ML_ADDITIONS
    + DRAINAGE_SEWAGE_CONTRASTIVE
    + PRIORITY_EMOTIONAL_ANCHORS
    # v4 hardening fixes
    + ELECTRICAL_HAZARD_CONTRASTIVE
    + PRIORITY_SEVERITY_ANCHORS_V2
    + MANGLISH_NON_SPAM
)

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
_RNG = random.Random(42)

# ===========================================================================
# Domain synonym tables
# ===========================================================================

# (pattern, [replacements]) — first replacement is the canonical form.
# We do ONE substitution per table entry per sentence pass.
_SYNONYM_TABLE: list[tuple[re.Pattern[str], list[str]]] = [
    # water
    (re.compile(r"\bwater pipe\b", re.I),   ["water pipe", "supply line", "water line", "main pipe", "pipeline"]),
    (re.compile(r"\bpipe\b", re.I),         ["pipe", "kuzhal", "line", "supply pipe", "tube"]),
    (re.compile(r"\bwater supply\b", re.I), ["water supply", "water connection", "municipal water", "water service"]),
    (re.compile(r"\btap water\b", re.I),    ["tap water", "tap", "faucet water", "the tap"]),
    (re.compile(r"\boverhead tank\b", re.I),["overhead tank", "water tank", "OHT", "storage tank", "tank"]),
    (re.compile(r"\bcontaminated\b", re.I), ["contaminated", "polluted", "dirty", "unsafe", "impure", "unhygienic"]),
    (re.compile(r"\bleak(ing|age|s)?\b", re.I), ["leaking", "dripping", "seeping", "oozing"]),

    # drainage
    (re.compile(r"\bdrain\b", re.I),        ["drain", "oda", "channal", "drainage channel", "gutter"]),
    (re.compile(r"\bdrainage channel\b", re.I), ["drainage channel", "stormwater channel", "oda channal", "drain canal"]),
    (re.compile(r"\bblocked\b", re.I),      ["blocked", "clogged", "choked", "obstructed", "jammed"]),
    (re.compile(r"\boverflowing\b", re.I),  ["overflowing", "flooding", "spilling over", "backing up"]),
    (re.compile(r"\bmanhole\b", re.I),      ["manhole", "inspection cover", "sewer hole", "manhole cover"]),
    (re.compile(r"\bflooded\b", re.I),      ["flooded", "waterlogged", "inundated", "submerged"]),

    # road
    (re.compile(r"\bpothole\b", re.I),      ["pothole", "kuzhi", "hole", "crater", "pit", "depression"]),
    (re.compile(r"\broad damage\b", re.I),  ["road damage", "road deterioration", "broken road", "road condition"]),
    (re.compile(r"\basphalt\b", re.I),      ["asphalt", "tar", "road surface", "bitumen"]),
    (re.compile(r"\bcrater\b", re.I),       ["crater", "large pothole", "deep hole", "massive pit"]),

    # sewage
    (re.compile(r"\bsewage\b", re.I),       ["sewage", "malinjalam", "waste water", "sullage", "effluent"]),
    (re.compile(r"\bseptic tank\b", re.I),  ["septic tank", "soakpit", "waste tank"]),
    (re.compile(r"\bsewer\b", re.I),        ["sewer", "sewage line", "underground drain", "sewer pipe"]),

    # solid waste
    (re.compile(r"\bgarbage\b", re.I),      ["garbage", "trash", "waste", "litter", "refuse", "muck"]),
    (re.compile(r"\bwaste\b", re.I),        ["waste", "garbage", "rubbish", "litter", "refuse"]),
    (re.compile(r"\bbin\b", re.I),          ["bin", "dustbin", "garbage bin", "waste bin", "collection box"]),
    (re.compile(r"\bcollection\b", re.I),   ["collection", "pickup", "clearing", "removal"]),

    # electrical
    (re.compile(r"\belectric pole\b", re.I),["electric pole", "power pole", "utility pole", "KSEB pole"]),
    (re.compile(r"\bsparking\b", re.I),     ["sparking", "arcing", "throwing sparks", "sparks flying"]),
    (re.compile(r"\bwire\b", re.I),         ["wire", "cable", "power line", "conductor"]),
    (re.compile(r"\belectricity\b", re.I),  ["electricity", "power", "current", "electric supply"]),

    # street light
    (re.compile(r"\bstreet light\b", re.I), ["street light", "street lamp", "road light", "public light", "pole light"]),
    (re.compile(r"\bnot working\b", re.I),  ["not working", "broken", "dead", "non-functional", "faulty", "off"]),
    (re.compile(r"\bdark\b", re.I),         ["dark", "unlit", "pitch dark", "no light", "poorly lit"]),

    # tree
    (re.compile(r"\btree\b", re.I),         ["tree", "maram", "large tree", "old tree", "giant tree"]),
    (re.compile(r"\bfell\b", re.I),         ["fell", "collapsed", "toppled", "came down", "fallen"]),
    (re.compile(r"\bbranch\b", re.I),       ["branch", "limb", "bough", "tree branch", "hanging branch"]),

    # illegal construction
    (re.compile(r"\bconstruction\b", re.I), ["construction", "building work", "structure", "building"]),
    (re.compile(r"\bunauthorised\b", re.I), ["unauthorised", "illegal", "unlawful", "unapproved", "unpermitted"]),
    (re.compile(r"\bencroachment\b", re.I), ["encroachment", "trespass", "intrusion", "illegal occupation"]),
]

# ---------------------------------------------------------------------------
# Code-mixing table: English civic word → Manglish equivalent(s)
# Only applied to English sentences.
# ---------------------------------------------------------------------------
_CODE_MIX_TABLE: dict[str, list[str]] = {
    "water":        ["vellam", "water"],
    "pipe":         ["kuzhal", "pipe"],
    "drain":        ["oda", "drain channal"],
    "road":         ["road", "pathha"],
    "garbage":      ["maalam", "garbage"],
    "tree":         ["maram", "tree"],
    "light":        ["light", "veli"],
    "electricity":  ["current", "electricity"],
    "sewage":       ["malinjalam", "sewage"],
    "blocked":      ["block aayittu", "blocked"],
    "broken":       ["thakarnu", "broken"],
    "leaking":      ["chori aanu", "leak"],
    "overflow":     ["nirachu varunnu", "overflow"],
    "damaged":      ["nallariyayi", "damaged"],
    "complaint":    ["parayittu", "complaint"],
    "urgent":       ["urgent", "veganam"],
    "dangerous":    ["dangerous", "apathakaramanu"],
    "repair":       ["repair", "sari aakkanam"],
    "missing":      ["illa", "missing"],
    "smell":        ["mananam", "smell"],
}

# ---------------------------------------------------------------------------
# Generic location / time placeholders to replace via slot injection
# ---------------------------------------------------------------------------
_LOCATION_PATTERNS = re.compile(
    r"\b(?:our area|our ward|our locality|our street|our lane|our colony|"
    r"the area|this area|near the junction|near the bus stop|near the school|"
    r"near the market|near the temple|the junction|this junction|"
    r"the main road|main road|our road|this road|this place|here|"
    r"in this colony|in our colony|in this locality|in the ward|"
    r"near the compound)\b",
    re.I,
)

_TIME_PATTERNS = re.compile(
    r"\b(?:for days|for a long time|since some time|for a while|"
    r"for many days|for some time|for so long|for too long|"
    r"for weeks|for months|days ago|days back|recently|lately|"
    r"a long time ago|since long)\b",
    re.I,
)


# ===========================================================================
# Augmentation helpers
# ===========================================================================

def _inject_location(text: str) -> str:
    """Replace a generic location phrase with a real TVM location name."""
    match = _LOCATION_PATTERNS.search(text)
    if not match:
        return text
    loc = _RNG.choice(TVM_LOCATIONS)
    # Replace with a preposition-aware version
    original = match.group(0)
    replacement = f"near {loc}" if "near" not in original.lower() else f"near {loc}"
    return text[: match.start()] + replacement + text[match.end() :]


def _inject_time(text: str) -> str:
    """Replace a generic time phrase with a specific TIME_EXPRESSION."""
    match = _TIME_PATTERNS.search(text)
    if not match:
        return text
    return text[: match.start()] + _RNG.choice(TIME_EXPRESSIONS) + text[match.end() :]


def _inject_impact(text: str) -> str:
    """Append an IMPACT_PHRASE to a formal sentence (English only, no ML)."""
    if not text[0].isupper():
        return text  # Don't append to Manglish / Malayalam lines
    return text.rstrip(".!? ") + ". " + _RNG.choice(IMPACT_PHRASES)


def _synonym_sub(text: str) -> str:
    """Apply one random synonym substitution from _SYNONYM_TABLE."""
    candidates: list[tuple[re.Pattern[str], list[str]]] = []
    for pattern, replacements in _SYNONYM_TABLE:
        if pattern.search(text):
            candidates.append((pattern, replacements))
    if not candidates:
        return text
    pattern, replacements = _RNG.choice(candidates)
    replacement = _RNG.choice(replacements[1:]) if len(replacements) > 1 else replacements[0]
    return pattern.sub(replacement, text, count=1)


def _code_mix(text: str) -> str:
    """Randomly replace 1–2 civic-domain words with Manglish equivalents."""
    words = text.split()
    changed = 0
    for i, word in enumerate(words):
        clean = word.lower().strip(".,!?")
        if clean in _CODE_MIX_TABLE and _RNG.random() < 0.4 and changed < 2:
            words[i] = _RNG.choice(_CODE_MIX_TABLE[clean])
            changed += 1
    return " ".join(words)


def _register_shift_to_informal(text: str) -> str:
    """Convert formal English to WhatsApp-style informal text."""
    if not text[0].isupper():
        return text  # already informal / not English

    # Remove articles and verbose phrases
    t = re.sub(r"\bThe\b", "", text)
    t = re.sub(r"\bthe\b", "", t)
    t = re.sub(r"\bhas been\b", "is", t)
    t = re.sub(r"\bhave been\b", "are", t)
    t = re.sub(r"\bwas\b", "is", t)
    t = re.sub(r"\bwere\b", "are", t)
    t = re.sub(r"\bPleases? (?:send|arrange|look into|take action|inspect|check)\b", "", t, flags=re.I)
    t = re.sub(r"\bImmediate action is (?:requested|needed|required|urgently needed)\b", "please act asap", t, flags=re.I)
    t = re.sub(r"\bResidents are\b", "people are", t, flags=re.I)
    # Lowercase entire sentence (WhatsApp style)
    t = t[0].lower() + t[1:] if t else t
    # Strip multiple spaces
    t = re.sub(r" {2,}", " ", t).strip(" .")
    # Optionally add urgency suffix
    suffix = _RNG.choice(["", " pls help", " urgent", " help needed", " ??"])
    return t + suffix


def _noise_inject(text: str) -> str:
    """Simulate typing noise: drop a word, repeat a word, or vowel typo."""
    words = text.split()
    if len(words) < 4:
        return text
    op = _RNG.choice(["drop", "repeat", "typo"])
    if op == "drop":
        idx = _RNG.randint(1, len(words) - 2)  # don't drop first/last
        words.pop(idx)
    elif op == "repeat":
        idx = _RNG.randint(0, len(words) - 1)
        words.insert(idx + 1, words[idx])
    else:  # typo: swap two adjacent vowels in a word
        idx = _RNG.randint(0, len(words) - 1)
        w = list(words[idx])
        vowels = [i for i, c in enumerate(w) if c in "aeiouAEIOU"]
        if len(vowels) >= 2:
            i1, i2 = vowels[0], vowels[1]
            w[i1], w[i2] = w[i2], w[i1]
            words[idx] = "".join(w)
    return " ".join(words)


# ===========================================================================
# Expansion engine
# ===========================================================================

# Expansion plan: a list of augmentation function-name combinations to apply
# to a single seed.  Each tuple is one variant strategy.
_CIVIC_AUGMENTS: list[list[str]] = [
    ["location"],
    ["time"],
    ["synonym"],
    ["location", "synonym"],
    ["time", "synonym"],
    ["informal"],
    ["informal", "location"],
    ["code_mix"],
    ["code_mix", "location"],
    ["noise"],
    ["location", "impact"],
    ["synonym", "impact"],
]

_SPAM_AUGMENTS: list[list[str]] = [
    ["noise"],
    ["informal"],
    ["synonym"],
    ["location"],  # spam sometimes mentions locations too
]


def _apply_augments(text: str, ops: list[str]) -> str:
    """Apply a list of named augment operations to text."""
    for op in ops:
        if op == "location":
            text = _inject_location(text)
        elif op == "time":
            text = _inject_time(text)
        elif op == "synonym":
            text = _synonym_sub(text)
        elif op == "informal":
            text = _register_shift_to_informal(text)
        elif op == "code_mix":
            text = _code_mix(text)
        elif op == "noise":
            text = _noise_inject(text)
        elif op == "impact":
            text = _inject_impact(text)
    return text.strip()


def _is_civic(category_code: str) -> bool:
    return category_code not in {"spam", "no_category"}


def expand_corpus(
    seeds: list[TrainingSample],
    target: int = 6000,
    civic_factor: int = 12,
    spam_factor: int = 4,
) -> list[TrainingSample]:
    """Expand seed corpus to ~target samples using structured augmentation.

    Parameters
    ----------
    seeds:         The raw TrainingSample list (ALL_SAMPLES + v3 bias-repair additions).
    target:        Desired minimum number of output samples.
    civic_factor:  Number of augmented variants to generate per civic seed.
    spam_factor:   Number of augmented variants to generate per spam/no_category seed.

    Returns
    -------
    Deduplicated list of TrainingSample, original seeds first.
    """
    seen: set[str] = set()
    result: list[TrainingSample] = []

    def _add(text: str, cat: str, pri: str, dept: str) -> None:
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append((text.strip(), cat, pri, dept))

    # ── Pass 1: add all original seeds ──────────────────────────────────────
    for text, cat, pri, dept in seeds:
        _add(text, cat, pri, dept)

    # ── Pass 2: augmentation ─────────────────────────────────────────────────
    for text, cat, pri, dept in seeds:
        augment_list = _CIVIC_AUGMENTS if _is_civic(cat) else _SPAM_AUGMENTS
        factor = civic_factor if _is_civic(cat) else spam_factor

        for plan in augment_list[:factor]:
            variant = _apply_augments(text, plan)
            _add(variant, cat, pri, dept)

    # ── Pass 3: extra passes with random plans until target reached ──────────
    civic_seeds = [(t, c, p, d) for t, c, p, d in seeds if _is_civic(c)]
    passes = 0
    while len(result) < target and passes < 20:
        _RNG.shuffle(civic_seeds)
        for text, cat, pri, dept in civic_seeds:
            # Randomly combine 1–3 augment ops
            k = _RNG.randint(1, 3)
            ops = _RNG.sample(
                ["location", "time", "synonym", "informal", "code_mix", "noise", "impact"],
                k=min(k, 3),
            )
            variant = _apply_augments(text, ops)
            _add(variant, cat, pri, dept)
            if len(result) >= target:
                break
        passes += 1

    return result


# ===========================================================================
# CSV / pickle I/O
# ===========================================================================

def to_csv_string(samples: list[TrainingSample]) -> str:
    """Serialise samples to CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["text", "category_code", "priority", "department_code"])
    writer.writerows(samples)
    return buf.getvalue()


def save_csv(samples: list[TrainingSample], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_csv_string(samples), encoding="utf-8")
    print(f"[generate_corpus_v2] Saved {len(samples)} samples → {path}")


def print_stats(samples: list[TrainingSample]) -> None:
    print(f"\n{'─'*60}")
    print(f"  Total samples : {len(samples)}")
    cat_counts = Counter(s[1] for s in samples)
    print(f"  {'Category':<25}  Count")
    print(f"  {'─'*25}  {'─'*6}")
    for cat in sorted(cat_counts):
        print(f"  {cat:<25}  {cat_counts[cat]}")
    pri_counts = Counter(s[2] for s in samples)
    print(f"\n  {'Priority':<12}  Count")
    for pri in ["low", "medium", "high", "urgent", "critical"]:
        print(f"  {pri:<12}  {pri_counts.get(pri, 0)}")
    print(f"{'─'*60}\n")


# ===========================================================================
# build_dataset — public API
# ===========================================================================

def build_dataset(
    target: int = 6000,
    civic_factor: int = 12,
    spam_factor: int = 4,
) -> list[TrainingSample]:
    """Build and return the expanded training dataset."""
    return expand_corpus(_ALL_SEEDS, target=target,
                         civic_factor=civic_factor, spam_factor=spam_factor)


# ===========================================================================
# CLI
# ===========================================================================

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate expanded ML training corpus from corpus_data_v2 seeds."
    )
    parser.add_argument("--target", type=int, default=6000,
                        help="Minimum number of samples (default: 6000)")
    parser.add_argument(
        "--out",
        type=str,
        default="apps/ml/training/corpus_v2.csv",
        help="Output CSV path (default: apps/ml/training/corpus_v2.csv)",
    )
    parser.add_argument("--civic-factor", type=int, default=12,
                        help="Augmented variants per civic seed (default: 12)")
    parser.add_argument("--spam-factor", type=int, default=4,
                        help="Augmented variants per spam/no_category seed (default: 4)")
    parser.add_argument("--stats", action="store_true",
                        help="Print per-category statistics and exit")
    args = parser.parse_args()

    print("[generate_corpus_v2] Building dataset…")
    samples = build_dataset(
        target=args.target,
        civic_factor=args.civic_factor,
        spam_factor=args.spam_factor,
    )
    print_stats(samples)

    if not args.stats:
        out_path = pathlib.Path(args.out)
        save_csv(samples, out_path)


if __name__ == "__main__":
    _main()

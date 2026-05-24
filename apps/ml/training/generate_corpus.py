"""apps/ml/training/generate_corpus.py

Expand the seed templates in corpus_data.py into a full labeled CSV that
training scripts can consume directly.

Augmentation strategies
-----------------------
1. Case variation        — lower, title, UPPER for some samples
2. Punctuation noise     — trailing !, periods, no punctuation
3. Typo injection        — single-character swap/deletion for a small fraction
4. Prefix / suffix noise — common complaint prefixes (sir, kindly, please note)
5. Ward + area tokens    — inject generic location words so the model learns
                           to ignore them as noise (ward 5, near junction, etc.)

Output
------
A CSV with columns: text, category_code, priority, department_code

Usage
-----
    python -m apps.ml.training.generate_corpus           # prints stats
    python -m apps.ml.training.generate_corpus --output corpus.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import re
import sys
from pathlib import Path

from apps.ml.training.corpus_data import (
    ALL_SAMPLES,
    CATEGORY_CODES,
    TrainingSample,
)

# Reproducible expansion
_RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

_PREFIXES = [
    "",
    "Sir, ",
    "Kindly note that ",
    "Please take action — ",
    "This is to bring to your notice that ",
    "Respected officer, ",
    "Dear sir/madam, ",
    "Urgently, ",
    "",
    "",
    "",
]

_SUFFIXES = [
    "",
    ".",
    "!",
    " Please take action.",
    " Kindly do the needful.",
    " Immediate action required.",
    " This has been going on for weeks.",
    " Please resolve at the earliest.",
    "",
    "",
]

_AREA_TOKENS = [
    "",
    " (ward 3)",
    " near junction",
    " near bus stop",
    " near school",
    " in our area",
    " in our ward",
    "",
    "",
]

_TYPO_PAIRS: list[tuple[str, str]] = [
    ("the ", "teh "),
    ("and ", "adn "),
    ("water", "watter"),
    ("road", "raod"),
    ("drain", "drainm"),
    ("light", "ligth"),
    ("electric", "eletric"),
    ("tree", "treee"),
    ("building", "buildng"),
    ("garbage", "garbge"),
]


def _apply_typo(text: str) -> str:
    """Randomly inject one small typo into the text (10 % of calls)."""
    if _RNG.random() > 0.10:
        return text
    for src, dst in _RNG.sample(_TYPO_PAIRS, min(3, len(_TYPO_PAIRS))):
        if src in text:
            return text.replace(src, dst, 1)
    return text


def _augment(text: str, n: int = 3) -> list[str]:
    """Return *n* augmented variants of *text* (including the original)."""
    variants: list[str] = [text]
    for _ in range(n - 1):
        t = text
        t = _apply_typo(t)
        prefix = _RNG.choice(_PREFIXES)
        suffix = _RNG.choice(_SUFFIXES)
        area   = _RNG.choice(_AREA_TOKENS)
        t = f"{prefix}{t}{area}{suffix}".strip()

        # Case variation (skip for Malayalam / Manglish heavy samples)
        if _RNG.random() < 0.15 and t.isascii():
            t = t.upper()
        elif _RNG.random() < 0.25 and t.isascii():
            t = t.title()

        # Clean up excessive spaces
        t = re.sub(r" {2,}", " ", t)
        variants.append(t)
    return variants


# ---------------------------------------------------------------------------
# Corpus expansion
# ---------------------------------------------------------------------------

def expand_corpus(
    samples: list[TrainingSample],
    augment_factor: int = 4,
) -> list[TrainingSample]:
    """Expand seed templates into *augment_factor × len(samples)* rows.

    Spam and no_category samples are not augmented beyond a factor of 2 to
    keep the class balance roughly proportional.
    """
    expanded: list[TrainingSample] = []
    for text, cat, prio, dept in samples:
        factor = 2 if cat in {"spam", "no_category"} else augment_factor
        for variant in _augment(text, n=factor):
            expanded.append((variant, cat, prio, dept))

    _RNG.shuffle(expanded)
    return expanded


# ---------------------------------------------------------------------------
# CSV serialisation
# ---------------------------------------------------------------------------

def to_csv_string(samples: list[TrainingSample]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["text", "category_code", "priority", "department_code"])
    for row in samples:
        writer.writerow(row)
    return buf.getvalue()


def save_csv(samples: list[TrainingSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category_code", "priority", "department_code"])
        for row in samples:
            writer.writerow(row)
    print(f"Saved {len(samples)} rows → {path}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(samples: list[TrainingSample]) -> None:
    from collections import Counter
    cat_counts  = Counter(cat  for _, cat, _, _  in samples)
    prio_counts = Counter(prio for _, _,  prio, _ in samples)
    print(f"\nTotal samples: {len(samples)}")
    print("\nBy category:")
    for cat in CATEGORY_CODES:
        print(f"  {cat:<28} {cat_counts.get(cat, 0):>5}")
    print("\nBy priority:")
    for prio, cnt in sorted(prio_counts.items()):
        print(f"  {prio:<12} {cnt:>5}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML training corpus CSV")
    parser.add_argument("--output", type=str, default="", help="Output CSV path (prints stats if omitted)")
    parser.add_argument("--factor", type=int, default=4, help="Augmentation factor per seed (default 4)")
    args = parser.parse_args()

    expanded = expand_corpus(ALL_SAMPLES, augment_factor=args.factor)
    print_stats(expanded)

    if args.output:
        save_csv(expanded, Path(args.output))
    else:
        print("\n(pass --output <path> to write CSV)")


if __name__ == "__main__":
    main()

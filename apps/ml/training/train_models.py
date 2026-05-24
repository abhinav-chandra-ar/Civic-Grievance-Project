"""apps/ml/training/train_models.py

Train and persist all ML models for the civic grievance intelligence pipeline.

Models trained
--------------
category_pipeline      TF-IDF (char+word n-grams) → LogisticRegression
                       Predicts: category_code (11 classes incl. spam, no_category)

priority_pipeline      Same feature set → LogisticRegression
                       Predicts: priority level (low/medium/high/urgent/critical)

department_pipeline    Same feature set → LogisticRegression
                       Predicts: department_code (routing target)

spam_pipeline          Same feature set → LogisticRegression
                       Predicts: binary is_spam (True/False)

language_pipeline      Char-only features → LogisticRegression
                       Predicts: language code (en / ml / manglish / mixed)

duplicate_vectorizer   TF-IDF vectorizer (word, IDF-weighted) fitted on corpus
                       Used at inference time for cosine-similarity duplicate detection

Artifacts
---------
All models are saved to  apps/ml/models/  via joblib.
File names:
    category_pipeline.joblib
    priority_pipeline.joblib
    department_pipeline.joblib
    spam_pipeline.joblib
    language_pipeline.joblib
    duplicate_vectorizer.joblib
    label_encoders.joblib       (dict of sklearn LabelEncoders for each model)

Usage
-----
    python -m apps.ml.training.train_models          # train + save
    python -m apps.ml.training.train_models --eval   # train + save + print eval report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from apps.ml.training.corpus_data import ALL_SAMPLES, TrainingSample  # noqa: E402
from apps.ml.training.generate_corpus import expand_corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature pipeline builder
# ---------------------------------------------------------------------------

def _build_text_features(
    char_ngram: tuple[int, int] = (2, 4),
    word_ngram: tuple[int, int] = (1, 2),
    max_features: int = 30_000,
) -> FeatureUnion:
    """Combine character and word n-gram TF-IDF into a single sparse feature matrix.

    ``char_wb`` (character n-grams within word boundaries) handles:
    - Malayalam Unicode naturally (ള്ളം, ടാപ്പ്, ...)
    - Manglish fragments (vellam, kuzhal, ...)
    - English with typos (watter → still shares char n-grams with water)

    ``word`` TF-IDF captures complete terms with IDF weighting, which makes
    common words (near, the, in) low-weight and rare civic terms high-weight.
    """
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram,
        max_features=max_features // 2,
        sublinear_tf=True,
        strip_accents=None,          # preserve Unicode
        min_df=1,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram,
        max_features=max_features // 2,
        sublinear_tf=True,
        strip_accents=None,
        min_df=1,
    )
    return FeatureUnion([
        ("char", char_tfidf),
        ("word", word_tfidf),
    ])


def _build_classifier(C: float = 2.0) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        max_iter=2_000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )


def _build_pipeline(C: float = 2.0) -> Pipeline:
    return Pipeline([
        ("features", _build_text_features()),
        ("clf", _build_classifier(C=C)),
    ])


# ---------------------------------------------------------------------------
# Language labelling heuristic (used to create language training labels)
# ---------------------------------------------------------------------------

_MALAYALAM_RANGE = (0x0D00, 0x0D7F)

def _detect_language_label(text: str) -> str:
    """Assign a coarse language label to a training sample text.

    Used to produce training labels for the language detection model.
    """
    ml_char_count = sum(
        1 for ch in text if _MALAYALAM_RANGE[0] <= ord(ch) <= _MALAYALAM_RANGE[1]
    )
    lower = text.lower()
    manglish_markers = {
        "vellam", "kuzhal", "maram", "vannu", "illa", "aanu", "venam",
        "odha", "drain", "road", "ayi", "undakkum", "cheyyunilla",
        "potti", "niranju", "ozhukunu",
    }
    manglish_hits = sum(1 for w in lower.split() if w in manglish_markers)
    total = len(text)

    if ml_char_count > 3:
        ascii_count = sum(1 for ch in text if ch.isascii())
        if ascii_count / max(total, 1) > 0.3:
            return "mixed"
        return "ml"

    if manglish_hits >= 2:
        # has English mixed in?
        eng_words = {"the", "is", "are", "was", "were", "has", "have", "near", "on",
                     "in", "for", "from", "to", "of", "a", "an"}
        eng_hits = sum(1 for w in lower.split() if w in eng_words)
        if eng_hits >= 2:
            return "mixed"
        return "manglish"

    return "en"


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def _prepare_samples(
    samples: list[TrainingSample],
    augment_factor: int = 4,
) -> tuple[list[str], dict[str, list[str]]]:
    """Expand and extract parallel label lists from training samples.

    Returns
    -------
    (texts, labels_dict)
    texts            — list of raw text strings
    labels_dict      — dict with keys:
                       "category", "priority", "department",
                       "spam", "language"
    """
    expanded = expand_corpus(samples, augment_factor=augment_factor)
    texts: list[str] = []
    cat_labels: list[str] = []
    prio_labels: list[str] = []
    dept_labels: list[str] = []
    spam_labels: list[str] = []
    lang_labels: list[str] = []

    for text, cat, prio, dept in expanded:
        texts.append(text)
        cat_labels.append(cat)
        prio_labels.append(prio)
        dept_labels.append(dept)
        spam_labels.append("spam" if cat == "spam" else "not_spam")
        lang_labels.append(_detect_language_label(text))

    return texts, {
        "category":   cat_labels,
        "priority":   prio_labels,
        "department": dept_labels,
        "spam":       spam_labels,
        "language":   lang_labels,
    }


def train_category_model(
    texts: list[str],
    labels: list[str],
    *,
    evaluate: bool = False,
) -> tuple[Pipeline, LabelEncoder]:
    print("Training category model …")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    pipeline = _build_pipeline(C=2.0)

    if evaluate:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, y, cv=cv, scoring="accuracy")
        print(f"  Category CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, y)
    print(f"  Classes: {list(le.classes_)}")
    return pipeline, le


def train_priority_model(
    texts: list[str],
    labels: list[str],
    *,
    evaluate: bool = False,
) -> tuple[Pipeline, LabelEncoder]:
    print("Training priority model …")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    pipeline = _build_pipeline(C=1.0)

    if evaluate:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, y, cv=cv, scoring="accuracy")
        print(f"  Priority CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, y)
    return pipeline, le


def train_department_model(
    texts: list[str],
    labels: list[str],
    *,
    evaluate: bool = False,
) -> tuple[Pipeline, LabelEncoder]:
    print("Training department model …")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    pipeline = _build_pipeline(C=2.0)

    if evaluate:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, y, cv=cv, scoring="accuracy")
        print(f"  Department CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, y)
    return pipeline, le


def train_spam_model(
    texts: list[str],
    labels: list[str],
    *,
    evaluate: bool = False,
) -> tuple[Pipeline, LabelEncoder]:
    print("Training spam model …")
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Spam is binary — a higher C prevents over-regularisation on small class
    pipeline = _build_pipeline(C=3.0)

    if evaluate:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, y, cv=cv, scoring="f1_weighted")
        print(f"  Spam CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, y)
    return pipeline, le


def train_language_model(
    texts: list[str],
    labels: list[str],
    *,
    evaluate: bool = False,
) -> tuple[Pipeline, LabelEncoder]:
    """Language detection uses only character n-grams (word boundaries not useful)."""
    print("Training language detection model …")
    le = LabelEncoder()
    y = le.fit_transform(labels)

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=20_000,
        sublinear_tf=True,
        strip_accents=None,
        min_df=1,
    )
    clf = LogisticRegression(
        C=1.0,
        max_iter=1_000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    pipeline = Pipeline([("tfidf", char_tfidf), ("clf", clf)])

    if evaluate:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, y, cv=cv, scoring="accuracy")
        print(f"  Language CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, y)
    print(f"  Language classes: {list(le.classes_)}")
    return pipeline, le


def fit_duplicate_vectorizer(texts: list[str]) -> TfidfVectorizer:
    """Fit an IDF-weighted word TF-IDF vectorizer for cosine-similarity duplicate detection.

    Using word unigrams with IDF weighting means frequent terms like "near",
    "the", "in" become low-weight, while specific civic terms like "pothole",
    "kuzhal", "sewage" are high-weight.  This gives much better semantic
    proximity than Jaccard token overlap.
    """
    print("Fitting duplicate detection vectorizer …")
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
        strip_accents=None,
        min_df=1,
    )
    vec.fit(texts)
    print(f"  Vocabulary size: {len(vec.vocabulary_)}")
    return vec


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_all(
    *,
    evaluate: bool = False,
    augment_factor: int = 4,
    save: bool = True,
) -> dict[str, Any]:
    """Train all models and optionally save artifacts.

    Returns a dict of the trained objects for programmatic access.
    """
    print("=" * 60)
    print("TVMC Civic Grievance ML Training Pipeline")
    print("=" * 60)

    texts, labels = _prepare_samples(ALL_SAMPLES, augment_factor=augment_factor)
    print(f"Training corpus: {len(texts)} samples (after augmentation × {augment_factor})")

    cat_pipeline,  cat_le   = train_category_model(  texts, labels["category"],   evaluate=evaluate)
    prio_pipeline, prio_le  = train_priority_model(  texts, labels["priority"],   evaluate=evaluate)
    dept_pipeline, dept_le  = train_department_model(texts, labels["department"], evaluate=evaluate)
    spam_pipeline, spam_le  = train_spam_model(      texts, labels["spam"],       evaluate=evaluate)
    lang_pipeline, lang_le  = train_language_model(  texts, labels["language"],   evaluate=evaluate)
    dup_vec                  = fit_duplicate_vectorizer(texts)

    label_encoders = {
        "category":   cat_le,
        "priority":   prio_le,
        "department": dept_le,
        "spam":       spam_le,
        "language":   lang_le,
    }

    artifacts = {
        "category_pipeline":    cat_pipeline,
        "priority_pipeline":    prio_pipeline,
        "department_pipeline":  dept_pipeline,
        "spam_pipeline":        spam_pipeline,
        "language_pipeline":    lang_pipeline,
        "duplicate_vectorizer": dup_vec,
        "label_encoders":       label_encoders,
    }

    if save:
        _save_artifacts(artifacts)

    print("\nDone.")
    return artifacts


def _save_artifacts(artifacts: dict[str, Any]) -> None:
    for name, obj in artifacts.items():
        path = _MODELS_DIR / f"{name}.joblib"
        joblib.dump(obj, path, compress=3)
        print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Optional detailed evaluation after training
# ---------------------------------------------------------------------------

def evaluate_category_model(
    pipeline: Pipeline,
    le: LabelEncoder,
    texts: list[str],
    labels: list[str],
) -> None:
    y_true = le.transform(labels)
    y_pred = pipeline.predict(texts)
    print("\nCategory model report (train set — for sanity check):")
    print(classification_report(y_true, y_pred, target_names=le.classes_))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TVMC civic ML models")
    parser.add_argument("--eval",   action="store_true", help="Run cross-validation")
    parser.add_argument("--factor", type=int, default=4, help="Augmentation factor (default 4)")
    args = parser.parse_args()

    train_all(evaluate=args.eval, augment_factor=args.factor, save=True)


if __name__ == "__main__":
    main()

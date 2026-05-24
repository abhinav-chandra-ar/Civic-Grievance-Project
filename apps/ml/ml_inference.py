"""apps/ml/ml_inference.py

Lazy-loading ML inference engine for the civic grievance classification pipeline.

Architecture — three-tier priority chain
-----------------------------------------
Tier 1 — Transformer backbone  (transformer_inference.py)
    paraphrase-multilingual-MiniLM-L12-v2 sentence embeddings +
    per-task LogisticRegression heads.  Best accuracy; semantic duplicate
    detection; location intelligence.

Tier 2 — TF-IDF + LogisticRegression  (this module)
    Char + word n-gram features; no neural backbone; good multilingual
    coverage via char_wb n-grams.  Used when transformer files are absent.

Tier 3 — Rule engine  (analyzer.py)
    Keyword + regex rules; always available.

Each public function tries Tier 1, falls back to Tier 2, and raises
ModelUnavailable if both are absent.  The caller (analyzer.py) catches
ModelUnavailable and falls back to Tier 3.

Tier tracking
-------------
Each call that succeeds stores the active tier in thread-local state so
that analyzer.py can set inference_source = "transformer" | "tfidf" | "rule"
accurately.

Error visibility
----------------
Transformer failures are logged at DEBUG level (not silently dropped).
TF-IDF failures are logged at WARNING level.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent / "models"

# Thread-local storage: records which tier last produced a result.
_tls = threading.local()


def _active_tier() -> str:
    """Return the tier that handled the most recent call in this thread."""
    return getattr(_tls, "last_tier", "unknown")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionResult:
    """Generic classification result."""
    label: str
    confidence: float
    all_probs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SpamResult:
    is_spam: bool
    spam_score: float


@dataclass(frozen=True)
class LanguageResult:
    language: str
    confidence: float


# ---------------------------------------------------------------------------
# Sentinel for unavailable models
# ---------------------------------------------------------------------------

class ModelUnavailable(Exception):
    """Raised when no model tier is available for inference."""


# ---------------------------------------------------------------------------
# Transformer tier helpers
# Each helper:
#   - Tries the TransformerEngine
#   - On TransformerUnavailable: logs DEBUG, returns None  (expected fallback)
#   - On unexpected exception: logs WARNING with full message, returns None
#   - On success: sets _tls.last_tier = "transformer"
# ---------------------------------------------------------------------------

def _try_transformer_category(text: str) -> PredictionResult | None:
    try:
        from apps.ml.transformer_inference import (  # noqa: PLC0415
            TransformerUnavailable,
            get_transformer_engine,
        )
        r = get_transformer_engine().predict_category(text)
        _tls.last_tier = "transformer"
        return PredictionResult(label=r.label, confidence=r.confidence,
                                all_probs=r.all_probs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Transformer category tier skipped: %s", exc)
        return None


def _try_transformer_priority(text: str) -> PredictionResult | None:
    try:
        from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
        r = get_transformer_engine().predict_priority(text)
        _tls.last_tier = "transformer"
        return PredictionResult(label=r.label, confidence=r.confidence)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Transformer priority tier skipped: %s", exc)
        return None


def _try_transformer_department(text: str) -> PredictionResult | None:
    try:
        from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
        r = get_transformer_engine().predict_department(text)
        _tls.last_tier = "transformer"
        return PredictionResult(label=r.label, confidence=r.confidence)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Transformer department tier skipped: %s", exc)
        return None


def _try_transformer_spam(text: str) -> SpamResult | None:
    try:
        from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
        r = get_transformer_engine().predict_spam(text)
        _tls.last_tier = "transformer"
        return SpamResult(is_spam=r.is_spam, spam_score=r.spam_score)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Transformer spam tier skipped: %s", exc)
        return None


def _try_transformer_language(text: str) -> LanguageResult | None:
    try:
        from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
        r = get_transformer_engine().predict_language(text)
        _tls.last_tier = "transformer"
        return LanguageResult(language=r.language, confidence=r.confidence)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Transformer language tier skipped: %s", exc)
        return None


def _try_transformer_duplicate(text_a: str, text_b: str) -> float | None:
    try:
        from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
        sim = get_transformer_engine().compute_duplicate_similarity(text_a, text_b)
        _tls.last_tier = "transformer"
        return sim
    except Exception as exc:  # noqa: BLE001
        logger.debug("Transformer duplicate tier skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# TF-IDF tier — module-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_models: dict[str, Any] = {}
_label_encoders: dict[str, Any] = {}
_load_attempted: bool = False
_load_error: str = ""


def _load_models() -> None:
    """Load TF-IDF model artifacts from disk.  Called at most once."""
    global _load_attempted, _load_error

    with _lock:
        if _load_attempted:
            return
        _load_attempted = True

        try:
            import joblib  # noqa: PLC0415
        except ImportError:
            _load_error = "joblib not installed"
            logger.warning("TF-IDF tier disabled: %s", _load_error)
            return

        required = [
            "category_pipeline",
            "priority_pipeline",
            "department_pipeline",
            "spam_pipeline",
            "language_pipeline",
            "duplicate_vectorizer",
            "label_encoders",
        ]

        missing = [r for r in required if not (_MODELS_DIR / f"{r}.joblib").exists()]
        if missing:
            _load_error = (
                f"Missing TF-IDF model files: {missing}.  "
                "Run: python manage.py train_ml_models --no-transformer"
            )
            logger.info(
                "TF-IDF tier not loaded (files absent: %s) — "
                "transformer tier will be used if available",
                missing,
            )
            return

        try:
            for name in required:
                _models[name] = joblib.load(_MODELS_DIR / f"{name}.joblib")
            les = _models.pop("label_encoders")
            _label_encoders.update(les)
            logger.info("TF-IDF models loaded from %s", _MODELS_DIR)
        except Exception as exc:  # noqa: BLE001
            _load_error = str(exc)
            logger.error("TF-IDF load failed: %s", exc)
            _models.clear()
            _label_encoders.clear()


def _require_model(name: str) -> Any:
    _load_models()
    if name not in _models:
        raise ModelUnavailable(
            f"TF-IDF model '{name}' not available. {_load_error or 'Run train_ml_models.'}"
        )
    _tls.last_tier = "tfidf"
    return _models[name]


def _require_le(name: str) -> Any:
    _load_models()
    if name not in _label_encoders:
        raise ModelUnavailable(f"TF-IDF label encoder '{name}' not available.")
    return _label_encoders[name]


# ---------------------------------------------------------------------------
# is_models_ready — public status check
# ---------------------------------------------------------------------------

def is_models_ready() -> bool:
    """Return True if any ML tier (transformer or TF-IDF) is ready."""
    try:
        from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
        if get_transformer_engine().is_ready:
            return True
    except Exception:  # noqa: BLE001
        pass
    _load_models()
    return bool(_models) and not _load_error


def active_tier() -> str:
    """Return which tier handled the most recent call in this thread.

    Returns: "transformer" | "tfidf" | "unknown"
    """
    return _active_tier()


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

def predict_category(text: str) -> PredictionResult:
    """Predict complaint category.  Transformer tier first, TF-IDF fallback.

    Raises ModelUnavailable when both tiers are absent.
    Sets thread-local active_tier() to the tier that answered.
    """
    r = _try_transformer_category(text)
    if r is not None:
        return r

    pipeline = _require_model("category_pipeline")
    le = _require_le("category")
    probs = pipeline.predict_proba([text])[0]
    idx = int(probs.argmax())
    label = str(le.inverse_transform([idx])[0])
    all_probs = {str(le.inverse_transform([i])[0]): float(p) for i, p in enumerate(probs)}
    return PredictionResult(label=label, confidence=float(probs[idx]), all_probs=all_probs)


def predict_priority(text: str) -> PredictionResult:
    """Predict priority.  Transformer tier first, TF-IDF fallback."""
    r = _try_transformer_priority(text)
    if r is not None:
        return r
    pipeline = _require_model("priority_pipeline")
    le = _require_le("priority")
    probs = pipeline.predict_proba([text])[0]
    idx = int(probs.argmax())
    return PredictionResult(label=str(le.inverse_transform([idx])[0]),
                            confidence=float(probs[idx]))


def predict_department(text: str) -> PredictionResult:
    """Predict department.  Transformer tier first, TF-IDF fallback."""
    r = _try_transformer_department(text)
    if r is not None:
        return r
    pipeline = _require_model("department_pipeline")
    le = _require_le("department")
    probs = pipeline.predict_proba([text])[0]
    idx = int(probs.argmax())
    return PredictionResult(label=str(le.inverse_transform([idx])[0]),
                            confidence=float(probs[idx]))


def predict_spam(text: str) -> SpamResult:
    """Predict spam.  Transformer tier first, TF-IDF fallback."""
    r = _try_transformer_spam(text)
    if r is not None:
        return r
    pipeline = _require_model("spam_pipeline")
    le = _require_le("spam")
    probs = pipeline.predict_proba([text])[0]
    classes = list(le.classes_)
    spam_score = float(probs[classes.index("spam")]) if "spam" in classes else 0.0
    return SpamResult(is_spam=spam_score >= 0.5, spam_score=spam_score)


def predict_language(text: str) -> LanguageResult:
    """Predict language.  Transformer tier first, TF-IDF fallback."""
    r = _try_transformer_language(text)
    if r is not None:
        return r
    pipeline = _require_model("language_pipeline")
    le = _require_le("language")
    probs = pipeline.predict_proba([text])[0]
    idx = int(probs.argmax())
    return LanguageResult(language=str(le.inverse_transform([idx])[0]),
                          confidence=float(probs[idx]))


def compute_duplicate_similarity(text_a: str, text_b: str) -> float:
    """Semantic cosine similarity.  Transformer tier first, TF-IDF fallback.

    Transformer tier uses sentence embeddings — correctly handles paraphrases
    ("No water supply" / "Water not coming for 2 days" → ~0.60).
    TF-IDF tier uses IDF-weighted lexical similarity.
    """
    r = _try_transformer_duplicate(text_a, text_b)
    if r is not None:
        return r

    from scipy.sparse import issparse  # noqa: PLC0415
    from sklearn.preprocessing import normalize  # noqa: PLC0415

    vec = _require_model("duplicate_vectorizer")
    mat = normalize(vec.transform([text_a, text_b]), norm="l2")

    if issparse(mat):
        sim = float((mat[0] * mat[1].T).toarray()[0][0])
    else:
        import numpy as np  # noqa: PLC0415
        sim = float(np.dot(mat[0], mat[1]))

    return round(max(0.0, min(1.0, sim)), 4)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_category_batch(texts: list[str]) -> list[PredictionResult]:
    """Batch category prediction (TF-IDF tier only — more efficient for bulk)."""
    pipeline = _require_model("category_pipeline")
    le = _require_le("category")
    all_probs_matrix = pipeline.predict_proba(texts)
    return [
        PredictionResult(
            label=str(le.inverse_transform([int(p.argmax())])[0]),
            confidence=float(p.max()),
        )
        for p in all_probs_matrix
    ]


# ---------------------------------------------------------------------------
# Testing support
# ---------------------------------------------------------------------------

def _reset_for_testing() -> None:
    """Clear singleton state.  Call in test setUp / tearDown."""
    global _load_attempted, _load_error
    with _lock:
        _models.clear()
        _label_encoders.clear()
        _load_attempted = False
        _load_error = ""

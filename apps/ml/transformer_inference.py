"""apps/ml/transformer_inference.py

Transformer-backbone inference engine for the TVMC civic grievance platform.

Architecture
------------
* Frozen SentenceTransformer backbone (paraphrase-multilingual-MiniLM-L12-v2,
  118 MB, 384-dim, CPU-friendly) encodes text → 384-dim L2-normalised embedding.
* One sklearn LogisticRegression head per task, trained on frozen embeddings.
* Semantic duplicate detection via cosine similarity of sentence embeddings —
  correctly handles paraphrases across English / Malayalam / Manglish.
* Location intelligence via pre-encoded TVM landmark embeddings, nearest-
  neighbour ranked by cosine similarity.

Public API
----------
get_transformer_engine()                     -> TransformerEngine  (singleton)
TransformerEngine.is_ready                   -> bool
TransformerEngine.predict_category(text)     -> PredictionResult
TransformerEngine.predict_priority(text)     -> PredictionResult
TransformerEngine.predict_department(text)   -> PredictionResult
TransformerEngine.predict_spam(text)         -> SpamResult
TransformerEngine.predict_language(text)     -> LanguageResult
TransformerEngine.compute_duplicate_similarity(a, b) -> float
TransformerEngine.find_ward_candidates(text, top_k=5) -> LocationResult

Module-level convenience wrappers (same names, call through the singleton):
predict_category / predict_priority / predict_department /
predict_spam / predict_language / compute_duplicate_similarity /
find_ward_candidates / is_models_ready

Thread safety
-------------
Model loading is double-checked locked.  After the first load, all public
calls are read-only and require no locking.

Error visibility
----------------
Failures during _load_models() are stored in _load_error (never silently
swallowed).  TransformerUnavailable is raised with the stored message so
callers can log it rather than guessing.

Label coercion
--------------
sklearn LabelEncoder.inverse_transform() returns numpy string scalars
(np.str_).  All public result objects coerce labels to plain Python str
via explicit str() calls so downstream code never receives np.str_ types.
"""
from __future__ import annotations

import logging
import pathlib
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    from sklearn.preprocessing import LabelEncoder  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark text normalisation (Fix 4 — abbreviation/misspelling expansion)
#
# Applied in find_ward_candidates() BEFORE embedding, so the cosine search
# sees the canonical form even when users type abbreviations or misspellings.
#
# Rules applied in order (earlier rules take priority via re.sub on the
# progressively-modified string):
#   1. Multi-word abbreviations first (med clg → medical college)
#   2. Single-word abbreviations (jn → junction, clg → college, …)
#   3. Common Thiruvananthapuram misspellings
# ---------------------------------------------------------------------------
_LOC_ABBREV_RULES: list[tuple[re.Pattern[str], str]] = [
    # ── multi-word abbreviations (must come before single-word rules) ────────
    (re.compile(r"\bmed(?:ical)?\s+clg\b",         re.I), "medical college"),
    (re.compile(r"\bmed(?:ical)?\s+coll(?:ege)?\b", re.I), "medical college"),
    (re.compile(r"\bsat\s+hosp(?:ital)?\b",         re.I), "SAT hospital"),
    (re.compile(r"\bsut\s+hosp(?:ital)?\b",         re.I), "SUT hospital"),
    (re.compile(r"\bgen(?:eral)?\s+hosp(?:ital)?\b", re.I), "government hospital"),
    (re.compile(r"\bgovt\s+hosp(?:ital)?\b",         re.I), "government hospital"),

    # ── single-word abbreviations ────────────────────────────────────────────
    (re.compile(r"\bjn\b",   re.I), "junction"),
    (re.compile(r"\bjct\b",  re.I), "junction"),
    (re.compile(r"\bclg\b",  re.I), "college"),
    (re.compile(r"\bcoll\b", re.I), "college"),
    (re.compile(r"\bhosp\b", re.I), "hospital"),
    (re.compile(r"\brd\b",   re.I), "road"),
    (re.compile(r"\bst\b",   re.I), "street"),   # only in location context

    # ── common Thiruvananthapuram misspellings ────────────────────────────────
    # Strict whole-word matching to avoid corrupting unrelated words.
    (re.compile(r"\bpalaym\b",       re.I), "Palayam"),
    (re.compile(r"\bpalayam\b",      re.I), "Palayam"),          # canonical noop (belt-and-braces)
    (re.compile(r"\bkazhakootam\b",  re.I), "Kazhakkoottam"),
    (re.compile(r"\bkazhakoottam\b", re.I), "Kazhakkoottam"),
    (re.compile(r"\bkazhakkotam\b",  re.I), "Kazhakkoottam"),
    (re.compile(r"\bkowdiyar\b",     re.I), "Kowdiar"),
    (re.compile(r"\bnanthankode\b",  re.I), "Nanthancode"),
    (re.compile(r"\bnanthankod\b",   re.I), "Nanthancode"),
    (re.compile(r"\bthampanur\b",    re.I), "Thampanoor"),
    (re.compile(r"\bkariavattam\b",  re.I), "Kariavattom"),
    (re.compile(r"\btrivandum\b",    re.I), "Thiruvananthapuram"),
    (re.compile(r"\btrivandrum\b",   re.I), "Thiruvananthapuram"),
    (re.compile(r"\btrivancore\b",   re.I), "Thiruvananthapuram"),
    (re.compile(r"\btvpm\b",         re.I), "Thiruvananthapuram"),
    (re.compile(r"\bvakery\b",       re.I), "Bakery"),           # common typo
]


def _normalize_location_text(text: str) -> str:
    """Expand abbreviations and fix common misspellings in location text.

    Applied before the embedding lookup in find_ward_candidates() so that
    the cosine search sees the canonical form even when users type e.g.
    'pothole near Bakery jn' or 'issue near Palaym market'.
    """
    for pattern, replacement in _LOC_ABBREV_RULES:
        text = pattern.sub(replacement, text)
    return text

# ---------------------------------------------------------------------------
# Model file paths
# ---------------------------------------------------------------------------
_MODELS_DIR: pathlib.Path = pathlib.Path(__file__).parent / "models"
_HEADS_FILE    = "transformer_heads.joblib"
_LANDMARK_FILE = "landmark_embeddings.joblib"

_BACKBONE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Module-level singleton state (protected by _lock during load)
# ---------------------------------------------------------------------------
_backbone: "SentenceTransformer | None" = None
_heads: dict = {}
_landmark_embeddings: np.ndarray | None = None
_landmark_names: list[str] = []
_load_attempted: bool = False
_load_error: str = ""
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class TransformerUnavailable(RuntimeError):
    """Raised when the transformer backbone or heads are unavailable.

    Check ``transformer_inference.load_error()`` for the root cause.
    """


def load_error() -> str:
    """Return the error string from the last failed load attempt, or ''."""
    return _load_error


# ---------------------------------------------------------------------------
# Result dataclasses
# Labels are guaranteed to be plain Python str (never np.str_).
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    label: str
    confidence: float
    all_probs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce numpy types that leak from LabelEncoder.inverse_transform
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "confidence", float(self.confidence))
        if self.all_probs:
            object.__setattr__(
                self,
                "all_probs",
                {str(k): float(v) for k, v in self.all_probs.items()},
            )


@dataclass
class SpamResult:
    is_spam: bool
    spam_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "spam_score", float(self.spam_score))


@dataclass
class LanguageResult:
    language: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", str(self.language))
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass
class LocationResult:
    """Ward/landmark candidates ranked by descending cosine similarity."""
    candidates: list[tuple[str, float]]
    top_ward: str
    top_score: float


# ---------------------------------------------------------------------------
# Internal loader
# ---------------------------------------------------------------------------

def _load_models() -> None:
    """Load backbone + heads + landmark embeddings.  Called once, thread-safe."""
    global _backbone, _heads, _landmark_embeddings, _landmark_names
    global _load_attempted, _load_error

    _load_attempted = True

    # ── 1. sentence-transformers importable? ─────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        _load_error = f"sentence-transformers not installed: {exc}"
        logger.warning("Transformer unavailable: %s", _load_error)
        return

    # ── 2. Head file exists? ──────────────────────────────────────────────
    heads_path = _MODELS_DIR / _HEADS_FILE
    if not heads_path.exists():
        _load_error = (
            f"Missing transformer head file: {heads_path}.  "
            "Run: python manage.py train_ml_models"
        )
        logger.warning("Transformer unavailable: %s", _load_error)
        return

    # ── 3. Load backbone ──────────────────────────────────────────────────
    try:
        _backbone = SentenceTransformer(_BACKBONE_MODEL)
        logger.info("Transformer backbone loaded: %s", _BACKBONE_MODEL)
    except Exception as exc:  # noqa: BLE001
        _load_error = f"Failed to load backbone '{_BACKBONE_MODEL}': {exc}"
        logger.error("Transformer load error: %s", _load_error)
        return

    # ── 4. Load classifier heads ──────────────────────────────────────────
    try:
        import joblib  # noqa: PLC0415
        _heads = joblib.load(heads_path)
        logger.info(
            "Transformer heads loaded from %s  (keys: %s)",
            heads_path,
            list(_heads.keys()),
        )
    except Exception as exc:  # noqa: BLE001
        _load_error = f"Failed to load transformer heads: {exc}"
        logger.error("Transformer load error: %s", _load_error)
        _backbone = None  # roll back partial load
        return

    # ── 5. Load landmark embeddings (optional) ────────────────────────────
    landmark_path = _MODELS_DIR / _LANDMARK_FILE
    if landmark_path.exists():
        try:
            import joblib  # noqa: PLC0415
            data = joblib.load(landmark_path)
            _landmark_embeddings = data["embeddings"]
            _landmark_names = [str(n) for n in data["names"]]
            logger.info(
                "Landmark embeddings loaded: %d landmarks", len(_landmark_names)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Landmark embeddings failed to load: %s", exc)
            # Non-fatal — location intelligence disabled but other heads work


def _ensure_loaded() -> None:
    """Thread-safe lazy loader.  Raises TransformerUnavailable on failure."""
    global _load_attempted
    if not _load_attempted:
        with _lock:
            if not _load_attempted:
                _load_models()
    if _load_error or _backbone is None or not _heads:
        raise TransformerUnavailable(
            _load_error or "Transformer backbone not loaded"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> np.ndarray:
    """Return a (N, 384) L2-normalised embedding matrix for N texts."""
    _ensure_loaded()
    assert _backbone is not None
    return _backbone.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _predict_head(head_key: str, le_key: str, embedding: np.ndarray) -> PredictionResult:
    """Run a classifier head on a (1, D) embedding and return PredictionResult."""
    clf = _heads[head_key]
    le: "LabelEncoder" = _heads[le_key]
    proba = clf.predict_proba(embedding)[0]
    idx = int(np.argmax(proba))
    # Coerce np.str_ → str via PredictionResult.__post_init__
    label = le.inverse_transform([idx])[0]
    all_probs = {
        le.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(proba)
    }
    return PredictionResult(label=label, confidence=float(proba[idx]), all_probs=all_probs)


# ---------------------------------------------------------------------------
# TransformerEngine — the primary public API object
# ---------------------------------------------------------------------------

class TransformerEngine:
    """Singleton inference engine backed by the transformer backbone.

    Obtain the global instance via::

        engine = get_transformer_engine()

    All methods raise ``TransformerUnavailable`` when the backbone or heads
    are not loaded.  Check ``engine.is_ready`` first, or catch the exception.
    """

    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """True if backbone + all classifier heads are loaded."""
        try:
            _ensure_loaded()
            return True
        except TransformerUnavailable:
            return False

    @property
    def load_error(self) -> str:
        """Human-readable error from the last failed load attempt, or ''."""
        return _load_error

    @property
    def backbone_name(self) -> str:
        return _BACKBONE_MODEL

    # ------------------------------------------------------------------
    def predict_category(self, text: str) -> PredictionResult:
        """Predict civic grievance category (11 classes).

        Raises TransformerUnavailable if not loaded.
        """
        emb = _embed([text])
        return _predict_head("category_head", "category_le", emb)

    def predict_priority(self, text: str) -> PredictionResult:
        """Predict priority: low / medium / high / urgent / critical."""
        emb = _embed([text])
        return _predict_head("priority_head", "priority_le", emb)

    def predict_department(self, text: str) -> PredictionResult:
        """Predict routing department."""
        emb = _embed([text])
        return _predict_head("department_head", "department_le", emb)

    def predict_spam(self, text: str) -> SpamResult:
        """Return SpamResult with is_spam flag and spam_score in [0, 1]."""
        emb = _embed([text])
        result = _predict_head("spam_head", "spam_le", emb)
        spam_score = result.all_probs.get("spam", 0.0)
        return SpamResult(is_spam=spam_score >= 0.50, spam_score=spam_score)

    def predict_language(self, text: str) -> LanguageResult:
        """Predict language: en / ml / manglish / mixed."""
        emb = _embed([text])
        result = _predict_head("language_head", "language_le", emb)
        return LanguageResult(language=result.label, confidence=result.confidence)

    def compute_duplicate_similarity(self, text_a: str, text_b: str) -> float:
        """Semantic cosine similarity in [0, 1].

        Encodes both texts in one batch call (efficient).  Normalised
        embeddings → dot product = cosine similarity.

        Example: "street light dead" vs "pole light not working" → ~0.67
        Example: "No water supply" vs "Water not coming for 2 days" → ~0.60
        """
        _ensure_loaded()
        embs = _embed([text_a, text_b])
        sim = float(np.dot(embs[0], embs[1]))
        return max(0.0, min(1.0, sim))

    def find_ward_candidates(
        self, location_text: str, top_k: int = 5
    ) -> LocationResult:
        """Rank TVM ward/landmark names by semantic similarity to location_text.

        Example::
            engine.find_ward_candidates("near Pattom junction opposite medical college")
            # LocationResult(top_ward='Pattom', top_score=0.83, ...)

        Raises TransformerUnavailable if landmark embeddings are not loaded.
        """
        _ensure_loaded()
        if _landmark_embeddings is None or not _landmark_names:
            raise TransformerUnavailable(
                "Landmark embeddings not loaded.  Re-run train_ml_models."
            )
        # Expand abbreviations / fix misspellings before embedding so that
        # "Bakery jn" → "Bakery junction", "Palaym" → "Palayam", etc.
        normalized_text = _normalize_location_text(location_text)
        loc_emb = _embed([normalized_text])[0]  # (384,)
        scores = _landmark_embeddings @ loc_emb  # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        candidates = [
            (str(_landmark_names[i]), float(scores[i])) for i in top_idx
        ]
        return LocationResult(
            candidates=candidates,
            top_ward=candidates[0][0],
            top_score=candidates[0][1],
        )


# ---------------------------------------------------------------------------
# Global singleton + factory function (the required public API)
# ---------------------------------------------------------------------------

_engine: TransformerEngine | None = None


def get_transformer_engine() -> TransformerEngine:
    """Return the global TransformerEngine singleton.

    The engine is created on first call.  Model loading is lazy — the
    backbone and heads are loaded the first time an inference method is called.

    Usage::

        engine = get_transformer_engine()
        if engine.is_ready:
            result = engine.predict_category("water pipe broken near school")
            print(result.label, result.confidence)
        else:
            print("Transformer unavailable:", engine.load_error)
    """
    global _engine
    if _engine is None:
        _engine = TransformerEngine()
    return _engine


# ---------------------------------------------------------------------------
# Module-level convenience wrappers  (backwards-compatible with old API)
# ---------------------------------------------------------------------------

def is_models_ready() -> bool:
    return get_transformer_engine().is_ready


def predict_category(text: str) -> PredictionResult:
    return get_transformer_engine().predict_category(text)


def predict_priority(text: str) -> PredictionResult:
    return get_transformer_engine().predict_priority(text)


def predict_department(text: str) -> PredictionResult:
    return get_transformer_engine().predict_department(text)


def predict_spam(text: str) -> SpamResult:
    return get_transformer_engine().predict_spam(text)


def predict_language(text: str) -> LanguageResult:
    return get_transformer_engine().predict_language(text)


def compute_duplicate_similarity(text_a: str, text_b: str) -> float:
    return get_transformer_engine().compute_duplicate_similarity(text_a, text_b)


def find_ward_candidates(location_text: str, top_k: int = 5) -> LocationResult:
    return get_transformer_engine().find_ward_candidates(location_text, top_k)


# ---------------------------------------------------------------------------
# Testing support
# ---------------------------------------------------------------------------

def _reset_for_testing() -> None:
    """Reset module-level state.  Called by test fixtures for isolation."""
    global _backbone, _heads, _landmark_embeddings, _landmark_names
    global _load_attempted, _load_error, _engine
    _backbone = None
    _heads = {}
    _landmark_embeddings = None
    _landmark_names = []
    _load_attempted = False
    _load_error = ""
    _engine = None

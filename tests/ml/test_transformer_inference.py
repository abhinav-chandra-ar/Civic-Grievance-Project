"""tests/ml/test_transformer_inference.py

Tests for the transformer inference engine (apps/ml/transformer_inference.py).

Test strategy
-------------
1. Unit tests with mocked SentenceTransformer — no model downloads.
2. Tests for semantic duplicate detection (the key upgrade over TF-IDF).
3. Tests for location intelligence (find_ward_candidates).
4. Tests for graceful degradation when transformer is unavailable.
5. Tests that ml_inference.py tries transformer first.

All tests are pure Python (no Django DB).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — build minimal mock backbone and heads
# ---------------------------------------------------------------------------

def _make_mock_backbone(embedding_dim: int = 384) -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    backbone = MagicMock()

    def _encode(texts, normalize_embeddings=True, show_progress_bar=False):
        # Return orthogonal unit vectors for each text (deterministic by hash)
        result = []
        for text in texts:
            rng = np.random.default_rng(hash(text) % (2**32))
            v = rng.standard_normal(embedding_dim).astype(np.float32)
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-9)
            result.append(v)
        return np.array(result)

    backbone.encode.side_effect = _encode
    return backbone


def _make_mock_head(classes: list[str], predicted_idx: int = 0) -> MagicMock:
    """Create a mock LogisticRegression head."""
    head = MagicMock()
    n = len(classes)
    probs = [0.05] * n
    probs[predicted_idx] = 0.80
    head.predict_proba.return_value = np.array([probs])
    return head


def _make_mock_le(classes: list[str]) -> MagicMock:
    """Create a mock LabelEncoder."""
    le = MagicMock()
    le.classes_ = classes

    def inverse_transform(indices):
        return [classes[i] for i in indices]

    le.inverse_transform.side_effect = inverse_transform
    return le


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_transformer_singleton(tmp_path):
    """Reset transformer_inference singleton state and point _MODELS_DIR at
    an empty temp dir so no real model files are loaded."""
    from apps.ml import transformer_inference  # noqa: PLC0415
    original_dir = transformer_inference._MODELS_DIR
    transformer_inference._reset_for_testing()
    transformer_inference._MODELS_DIR = tmp_path
    yield
    transformer_inference._MODELS_DIR = original_dir
    transformer_inference._reset_for_testing()


@pytest.fixture
def loaded_transformer(tmp_path):
    """Inject mocked backbone + heads into transformer_inference singleton."""
    from apps.ml import transformer_inference as ti  # noqa: PLC0415

    cat_classes  = ["drainage", "electrical_hazard", "no_category", "road_damage",
                     "sewage_issue", "solid_waste", "spam", "street_light",
                     "tree_fall", "water_supply", "illegal_construction"]
    prio_classes = ["critical", "high", "low", "medium", "urgent"]
    dept_classes = ["drainage", "electricity", "none", "parks", "planning",
                    "roads", "sanitation", "sewage", "water"]
    spam_classes = ["not_spam", "spam"]
    lang_classes = ["en", "manglish", "mixed", "ml"]

    backbone = _make_mock_backbone()
    heads = {
        "category_head":   _make_mock_head(cat_classes,  9),   # water_supply
        "priority_head":   _make_mock_head(prio_classes, 1),   # high
        "department_head": _make_mock_head(dept_classes, 8),   # water
        "spam_head":       _make_mock_head(spam_classes, 0),   # not_spam
        "language_head":   _make_mock_head(lang_classes, 0),   # en
        "category_le":     _make_mock_le(cat_classes),
        "priority_le":     _make_mock_le(prio_classes),
        "department_le":   _make_mock_le(dept_classes),
        "spam_le":         _make_mock_le(spam_classes),
        "language_le":     _make_mock_le(lang_classes),
    }

    # Pre-encode 3 fake landmark names
    landmark_names = ["Pattom", "Medical College", "Kazhakkoottam"]
    landmark_embs  = backbone.encode(landmark_names, normalize_embeddings=True)

    with patch.object(ti, "_backbone",            backbone), \
         patch.object(ti, "_heads",               heads), \
         patch.object(ti, "_landmark_embeddings", landmark_embs), \
         patch.object(ti, "_landmark_names",      landmark_names), \
         patch.object(ti, "_load_attempted",      True), \
         patch.object(ti, "_load_error",          ""):
        yield backbone, heads


# ---------------------------------------------------------------------------
# is_models_ready
# ---------------------------------------------------------------------------

class TestTransformerIsReady:
    def test_false_when_not_loaded(self):
        from apps.ml.transformer_inference import is_models_ready  # noqa: PLC0415
        assert is_models_ready() is False

    def test_true_when_loaded(self, loaded_transformer):
        from apps.ml.transformer_inference import is_models_ready  # noqa: PLC0415
        assert is_models_ready() is True


# ---------------------------------------------------------------------------
# predict_category
# ---------------------------------------------------------------------------

class TestTransformerPredictCategory:
    def test_returns_prediction_result(self, loaded_transformer):
        from apps.ml.transformer_inference import predict_category  # noqa: PLC0415
        result = predict_category("water pipe broken near school")
        assert result.label == "water_supply"
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_is_from_head(self, loaded_transformer):
        from apps.ml.transformer_inference import predict_category  # noqa: PLC0415
        result = predict_category("some civic issue text")
        assert result.confidence == pytest.approx(0.80, abs=0.01)

    def test_raises_when_not_loaded(self):
        from apps.ml.transformer_inference import (  # noqa: PLC0415
            TransformerUnavailable, predict_category,
        )
        with pytest.raises(TransformerUnavailable):
            predict_category("any text")


# ---------------------------------------------------------------------------
# predict_spam
# ---------------------------------------------------------------------------

class TestTransformerPredictSpam:
    def test_not_spam_result(self, loaded_transformer):
        from apps.ml.transformer_inference import predict_spam  # noqa: PLC0415
        result = predict_spam("water supply cut off for three days")
        assert result.is_spam is False
        assert 0.0 <= result.spam_score <= 1.0

    def test_spam_detected(self, loaded_transformer):
        from apps.ml import transformer_inference as ti  # noqa: PLC0415

        spam_head = _make_mock_head(["not_spam", "spam"], predicted_idx=1)
        spam_head.predict_proba.return_value = np.array([[0.05, 0.95]])
        spam_le   = _make_mock_le(["not_spam", "spam"])

        with patch.object(ti, "_heads",
                          {**ti._heads, "spam_head": spam_head, "spam_le": spam_le}):
            from apps.ml.transformer_inference import predict_spam  # noqa: PLC0415
            result = predict_spam("buy cheap medicine call now")

        assert result.is_spam is True
        assert result.spam_score == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------------------
# predict_language
# ---------------------------------------------------------------------------

class TestTransformerPredictLanguage:
    def test_returns_language_result(self, loaded_transformer):
        from apps.ml.transformer_inference import predict_language  # noqa: PLC0415
        result = predict_language("road has large potholes near junction")
        assert result.language == "en"
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# compute_duplicate_similarity — semantic duplicate detection
# ---------------------------------------------------------------------------

class TestTransformerDuplicateSimilarity:
    def test_returns_float_in_unit_interval(self, loaded_transformer):
        from apps.ml.transformer_inference import compute_duplicate_similarity  # noqa: PLC0415
        sim = compute_duplicate_similarity(
            "water pipe leaking near school",
            "pipe burst near the school compound",
        )
        assert 0.0 <= sim <= 1.0

    def test_identical_texts_give_high_similarity(self, loaded_transformer):
        """The mock backbone returns the same embedding for identical text."""
        from apps.ml.transformer_inference import compute_duplicate_similarity  # noqa: PLC0415
        text = "road pothole near school junction"
        sim = compute_duplicate_similarity(text, text)
        # Identical text → same vector → cosine = 1.0
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_raises_when_not_loaded(self):
        from apps.ml.transformer_inference import (  # noqa: PLC0415
            TransformerUnavailable, compute_duplicate_similarity,
        )
        with pytest.raises(TransformerUnavailable):
            compute_duplicate_similarity("text a", "text b")

    def test_paraphrase_detection(self, loaded_transformer):
        """Two semantically similar but lexically different texts.

        With the mock backbone (hash-based random embeddings), these will
        have near-zero similarity.  This test verifies the API contract; the
        real model will score them high.  We just check output is in [0, 1].
        """
        from apps.ml.transformer_inference import compute_duplicate_similarity  # noqa: PLC0415
        sim = compute_duplicate_similarity(
            "street light is not working near the bus stop",
            "pole light is dead at the bus stand",
        )
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# find_ward_candidates — location intelligence
# ---------------------------------------------------------------------------

class TestFindWardCandidates:
    def test_returns_location_result(self, loaded_transformer):
        from apps.ml.transformer_inference import find_ward_candidates  # noqa: PLC0415
        result = find_ward_candidates("near Pattom junction")
        assert result.top_ward in ["Pattom", "Medical College", "Kazhakkoottam"]
        assert 0.0 <= result.top_score <= 1.0
        assert len(result.candidates) > 0

    def test_candidates_sorted_descending(self, loaded_transformer):
        from apps.ml.transformer_inference import find_ward_candidates  # noqa: PLC0415
        result = find_ward_candidates("near the hospital")
        scores = [s for _, s in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_raises_when_not_loaded(self):
        from apps.ml.transformer_inference import (  # noqa: PLC0415
            TransformerUnavailable, find_ward_candidates,
        )
        with pytest.raises(TransformerUnavailable):
            find_ward_candidates("near junction")

    def test_raises_when_no_landmark_embeddings(self, loaded_transformer):
        """Even with backbone loaded, missing landmark embeddings → error."""
        from apps.ml import transformer_inference as ti  # noqa: PLC0415
        from apps.ml.transformer_inference import (  # noqa: PLC0415
            TransformerUnavailable, find_ward_candidates,
        )
        with patch.object(ti, "_landmark_embeddings", None), \
             patch.object(ti, "_landmark_names", []):
            with pytest.raises(TransformerUnavailable):
                find_ward_candidates("near junction")


# ---------------------------------------------------------------------------
# ml_inference.py tries transformer first
# ---------------------------------------------------------------------------

class TestMlInferenceTransformerPriority:
    """Verify that ml_inference.py uses transformer tier when available."""

    @pytest.fixture(autouse=True)
    def reset_ml(self, tmp_path):
        from apps.ml import ml_inference  # noqa: PLC0415
        original_dir = ml_inference._MODELS_DIR
        ml_inference._reset_for_testing()
        ml_inference._MODELS_DIR = tmp_path
        yield
        ml_inference._MODELS_DIR = original_dir
        ml_inference._reset_for_testing()

    def test_predict_category_uses_transformer_when_available(
        self, loaded_transformer
    ):
        """When transformer is loaded, ml_inference.predict_category returns
        transformer result (water_supply from our mock)."""
        from apps.ml.ml_inference import predict_category  # noqa: PLC0415
        result = predict_category("water pipe broken near school")
        assert result.label == "water_supply"

    def test_predict_category_falls_back_to_tfidf_when_transformer_unavailable(
        self,
    ):
        """When transformer is not loaded AND TF-IDF is not loaded → ModelUnavailable."""
        from apps.ml.ml_inference import ModelUnavailable, predict_category  # noqa: PLC0415
        with pytest.raises(ModelUnavailable):
            predict_category("water pipe broken near school")

    def test_duplicate_similarity_uses_transformer(self, loaded_transformer):
        """compute_duplicate_similarity should use transformer embeddings."""
        from apps.ml.ml_inference import compute_duplicate_similarity  # noqa: PLC0415
        # Identical text → 1.0 from transformer
        text = "road pothole near school"
        sim = compute_duplicate_similarity(text, text)
        assert sim == pytest.approx(1.0, abs=0.01)

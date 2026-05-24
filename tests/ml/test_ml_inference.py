"""tests/ml/test_ml_inference.py

Tests for the ML inference engine (apps/ml/ml_inference.py).

Test strategy
-------------
1. Unit tests with mocked sklearn models — no disk I/O, no model files required.
2. Integration-style tests for the corpus + training pipeline.
3. Tests for graceful degradation when models are unavailable.
4. Tests that analyze_complaint() falls back cleanly to the rule engine.

All tests are pure Python (no Django DB) — no django_db marker needed.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build minimal sklearn-compatible mock pipeline
# ---------------------------------------------------------------------------

def _make_label_encoder(classes: list[str]) -> MagicMock:
    """Create a mock sklearn LabelEncoder with the given classes."""
    le = MagicMock()
    le.classes_ = classes

    def inverse_transform(indices):
        return [classes[i] for i in indices]

    le.inverse_transform.side_effect = inverse_transform
    return le


def _make_pipeline(classes: list[str], predicted_idx: int = 0) -> MagicMock:
    """Create a mock sklearn Pipeline that always predicts classes[predicted_idx]."""
    import numpy as np  # noqa: PLC0415

    pipeline = MagicMock()
    n = len(classes)
    # Build a probability vector with a high score on predicted_idx
    probs = [0.05] * n
    probs[predicted_idx] = 0.80
    pipeline.predict_proba.return_value = np.array([probs])
    return pipeline


def _make_vectorizer(n_features: int = 10) -> MagicMock:
    """Create a mock TF-IDF vectorizer.

    ``transform`` always returns a 2-row sparse matrix so that
    ``compute_duplicate_similarity`` can index rows 0 and 1 correctly.
    """
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    vec = MagicMock()
    # Two-row matrix: different vectors so similarity ≠ 1.0 by default
    data = np.zeros((2, n_features))
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    vec.transform.return_value = csr_matrix(data)
    vec.vocabulary_ = {f"word{i}": i for i in range(n_features)}
    return vec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ml_singleton(tmp_path):
    """Reset ml_inference AND transformer_inference singleton state before
    and after each test.

    Points both _MODELS_DIR variables at an empty temp dir so tests that do
    NOT use the ``loaded_models`` fixture will see no model files on disk (even
    if the project's apps/ml/models/ directory is populated from a prior
    training run).  Tests that need real-ish models use the ``loaded_models``
    fixture instead, which injects mocked model objects directly.

    We must also reset transformer_inference because ml_inference.py now tries
    the transformer tier first via _try_transformer_*() helpers.  Without this
    reset, real transformer model files on disk would override the TF-IDF mocks.
    """
    from apps.ml import ml_inference  # noqa: PLC0415
    original_dir = ml_inference._MODELS_DIR
    ml_inference._reset_for_testing()
    ml_inference._MODELS_DIR = tmp_path   # empty → no model files found

    # Also reset and isolate the transformer singleton
    try:
        from apps.ml import transformer_inference  # noqa: PLC0415
        original_ti_dir = transformer_inference._MODELS_DIR
        transformer_inference._reset_for_testing()
        transformer_inference._MODELS_DIR = tmp_path  # empty → no transformer files
    except Exception:  # noqa: BLE001
        original_ti_dir = None

    yield

    ml_inference._MODELS_DIR = original_dir
    ml_inference._reset_for_testing()

    try:
        from apps.ml import transformer_inference  # noqa: PLC0415
        if original_ti_dir is not None:
            transformer_inference._MODELS_DIR = original_ti_dir
        transformer_inference._reset_for_testing()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def loaded_models():
    """Patch the model loader so models appear loaded without real disk I/O."""
    from apps.ml import ml_inference  # noqa: PLC0415

    cat_classes  = ["drainage", "electrical_hazard", "no_category", "road_damage",
                     "sewage_issue", "solid_waste", "spam", "street_light",
                     "tree_fall", "water_supply", "illegal_construction"]
    prio_classes = ["critical", "high", "low", "medium", "urgent"]
    dept_classes = ["drainage", "electricity", "none", "parks", "planning",
                    "roads", "sanitation", "sewage", "water"]
    spam_classes = ["not_spam", "spam"]
    lang_classes = ["en", "manglish", "mixed", "ml"]

    models = {
        "category_pipeline":   _make_pipeline(cat_classes,  predicted_idx=9),   # water_supply
        "priority_pipeline":   _make_pipeline(prio_classes, predicted_idx=1),   # high
        "department_pipeline": _make_pipeline(dept_classes, predicted_idx=8),   # water
        "spam_pipeline":       _make_pipeline(spam_classes, predicted_idx=0),   # not_spam
        "language_pipeline":   _make_pipeline(lang_classes, predicted_idx=0),   # en
        "duplicate_vectorizer": _make_vectorizer(),
    }
    label_encoders = {
        "category":   _make_label_encoder(cat_classes),
        "priority":   _make_label_encoder(prio_classes),
        "department": _make_label_encoder(dept_classes),
        "spam":       _make_label_encoder(spam_classes),
        "language":   _make_label_encoder(lang_classes),
    }

    with patch.object(ml_inference, "_models", models), \
         patch.object(ml_inference, "_label_encoders", label_encoders), \
         patch.object(ml_inference, "_load_attempted", True), \
         patch.object(ml_inference, "_load_error", ""):
        yield models, label_encoders


# ---------------------------------------------------------------------------
# is_models_ready
# ---------------------------------------------------------------------------

class TestIsModelsReady:
    def test_returns_false_when_models_not_loaded(self):
        from apps.ml.ml_inference import is_models_ready  # noqa: PLC0415
        # Patch _load_models to do nothing so we can test the unloaded state
        with patch("apps.ml.ml_inference._load_models"):
            from apps.ml import ml_inference  # noqa: PLC0415
            ml_inference._load_attempted = True
            ml_inference._load_error = "Missing model files: ['category_pipeline']"
            assert is_models_ready() is False

    def test_returns_true_when_models_loaded(self, loaded_models):
        from apps.ml.ml_inference import is_models_ready  # noqa: PLC0415
        assert is_models_ready() is True


# ---------------------------------------------------------------------------
# predict_category
# ---------------------------------------------------------------------------

class TestPredictCategory:
    def test_returns_prediction_result(self, loaded_models):
        from apps.ml.ml_inference import predict_category  # noqa: PLC0415
        result = predict_category("water pipe leaking near school")
        assert result.label == "water_supply"
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_is_calibrated_probability(self, loaded_models):
        from apps.ml.ml_inference import predict_category  # noqa: PLC0415
        result = predict_category("some civic complaint text")
        assert result.confidence == pytest.approx(0.80, abs=0.01)

    def test_raises_model_unavailable_when_not_loaded(self):
        from apps.ml.ml_inference import ModelUnavailable, predict_category  # noqa: PLC0415
        with pytest.raises(ModelUnavailable):
            predict_category("any text")


# ---------------------------------------------------------------------------
# predict_spam
# ---------------------------------------------------------------------------

class TestPredictSpam:
    def test_not_spam_for_civic_complaint(self, loaded_models):
        from apps.ml.ml_inference import predict_spam  # noqa: PLC0415
        result = predict_spam("water supply cut off for three days")
        assert result.is_spam is False
        assert 0.0 <= result.spam_score <= 1.0

    def test_spam_result_for_promotional_text(self, loaded_models):
        """When spam class probability ≥ 0.50 → is_spam=True."""
        from apps.ml import ml_inference  # noqa: PLC0415

        # Override spam pipeline to return high spam probability
        spam_pipe = ml_inference._models["category_pipeline"]
        spam_le   = ml_inference._label_encoders["spam"]

        import numpy as np  # noqa: PLC0415
        spam_pipeline = _make_pipeline(["not_spam", "spam"], predicted_idx=1)
        spam_pipeline.predict_proba.return_value = np.array([[0.10, 0.90]])

        with patch.object(ml_inference, "_models",
                          {**ml_inference._models, "spam_pipeline": spam_pipeline}):
            from apps.ml.ml_inference import predict_spam  # noqa: PLC0415
            from apps.ml import ml_inference as mi2  # noqa: PLC0415
            # Need to also patch label_encoders.spam for "spam" class
            spam_le_mock = _make_label_encoder(["not_spam", "spam"])
            with patch.object(mi2, "_label_encoders",
                               {**mi2._label_encoders, "spam": spam_le_mock}):
                result = predict_spam("buy cheap medicines call now")
        assert result.spam_score == pytest.approx(0.90, abs=0.01)


# ---------------------------------------------------------------------------
# predict_language
# ---------------------------------------------------------------------------

class TestPredictLanguage:
    def test_returns_language_result(self, loaded_models):
        from apps.ml.ml_inference import predict_language  # noqa: PLC0415
        result = predict_language("the road has large potholes")
        assert result.language == "en"
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# compute_duplicate_similarity
# ---------------------------------------------------------------------------

class TestComputeDuplicateSimilarity:
    def test_returns_float_in_unit_interval(self, loaded_models):
        from apps.ml.ml_inference import compute_duplicate_similarity  # noqa: PLC0415
        sim = compute_duplicate_similarity(
            "water pipe leaking near school",
            "pipe burst near the school compound",
        )
        assert 0.0 <= sim <= 1.0

    def test_identical_texts_give_high_similarity(self, loaded_models):
        """Identical texts → exact same TF-IDF vector → cosine similarity = 1.0."""
        from apps.ml import ml_inference as mi  # noqa: PLC0415
        from scipy.sparse import csr_matrix  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # Return two identical rows → cosine = 1.0
        vec = MagicMock()
        vec.transform.return_value = csr_matrix(
            np.array([[0.0, 1.0, 0.5, 0.3], [0.0, 1.0, 0.5, 0.3]])
        )
        vec.vocabulary_ = {"a": 0, "b": 1, "c": 2, "d": 3}

        with patch.object(mi, "_models", {**mi._models, "duplicate_vectorizer": vec}):
            sim = mi.compute_duplicate_similarity(
                "road pothole near school", "road pothole near school"
            )

        assert sim == pytest.approx(1.0, abs=0.01)

    def test_raises_model_unavailable_when_not_loaded(self):
        from apps.ml.ml_inference import ModelUnavailable, compute_duplicate_similarity  # noqa: PLC0415
        with pytest.raises(ModelUnavailable):
            compute_duplicate_similarity("text a", "text b")


# ---------------------------------------------------------------------------
# Analyzer integration: ML + rule fallback
# ---------------------------------------------------------------------------

class TestAnalyzeComplaintMLFusion:
    """Tests for analyze_complaint() with ML models mocked."""

    def test_analyze_uses_ml_category_when_available(self, loaded_models):
        """When ML returns high-confidence category, that category is used."""
        from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
        result = analyze_complaint("water pipe broken near the school")
        # ML mock predicts water_supply with 0.80 confidence
        assert result["category_code"] == "water_supply"
        # Tier-specific source values: "tfidf" (TF-IDF mocks active),
        # "transformer" / "*_fusion" accepted when transformer is available.
        assert result["inference_source"] in {
            "tfidf", "transformer",
            "tfidf_fusion", "transformer_fusion",
        }

    def test_analyze_has_inference_source_key(self, loaded_models):
        from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
        result = analyze_complaint("large pothole on road")
        assert "inference_source" in result

    def test_analyze_returns_all_required_keys(self, loaded_models):
        from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
        result = analyze_complaint("sewage overflow near temple junction")
        required = {
            "language", "language_confidence", "normalized_text",
            "category_code", "category_confidence", "department_code",
            "landmarks", "ward_hint", "landmark_confidence",
            "priority", "spam", "duplicate",
            "needs_human_review", "review_reasons", "confidence",
            "image_analysis", "decision", "inference_source",
        }
        assert required.issubset(result.keys())

    def test_analyze_falls_back_to_rules_when_ml_unavailable(self):
        """When ML models are not loaded, rule engine result is returned.

        The autouse fixture points _MODELS_DIR at an empty temp dir so ML
        will not load, and the rule engine handles classification.
        """
        from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
        result = analyze_complaint("large pothole on road near school junction")
        # Rule engine must still return a valid category
        assert result["category_code"] == "road_damage"
        # With no ML models, _fuse_category returns rule_result which has no
        # "source" key → .get("source", "rule") == "rule".
        assert result["inference_source"] in {
            "rule", "rule_over_tfidf", "rule_over_transformer", "",
        }

    def test_spam_detected_by_ml_raises_review_flag(self, loaded_models):
        """A spam prediction causes spam_suspicion review reason."""
        from apps.ml import ml_inference as mi  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        spam_pipeline = _make_pipeline(["not_spam", "spam"], predicted_idx=1)
        spam_pipeline.predict_proba.return_value = np.array([[0.05, 0.95]])
        spam_le = _make_label_encoder(["not_spam", "spam"])

        with patch.object(mi, "_models", {**mi._models, "spam_pipeline": spam_pipeline}), \
             patch.object(mi, "_label_encoders", {**mi._label_encoders, "spam": spam_le}):
            from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
            result = analyze_complaint("buy medicine cheap call now discount")

        assert "spam_suspicion" in result["review_reasons"]


# ---------------------------------------------------------------------------
# ML category fusion logic
# ---------------------------------------------------------------------------

class TestFuseCategory:
    def test_ml_primary_wins_at_high_confidence(self):
        from apps.ml.analyzer import _fuse_category  # noqa: PLC0415
        rule = {"category_code": "road_damage", "confidence": 0.50}
        ml   = {
            "category_code": "water_supply", "confidence": 0.50,
            "ml_label": "water_supply", "ml_confidence": 0.75,
            "ml_tier": "transformer",
        }
        result = _fuse_category(rule, ml)
        assert result["category_code"] == "water_supply"
        assert result["source"] == "transformer"

    def test_rule_used_when_ml_confidence_low(self):
        from apps.ml.analyzer import _fuse_category  # noqa: PLC0415
        rule = {"category_code": "road_damage", "confidence": 0.65}
        ml   = {"category_code": "water_supply", "confidence": 0.50, "ml_label": "water_supply", "ml_confidence": 0.20}
        result = _fuse_category(rule, ml)
        assert result["category_code"] == "road_damage"

    def test_agreement_averages_confidence(self):
        from apps.ml.analyzer import _fuse_category  # noqa: PLC0415
        rule = {"category_code": "drainage", "confidence": 0.60}
        ml   = {
            "category_code": "drainage", "confidence": 0.40,
            "ml_label": "drainage", "ml_confidence": 0.45,
            "ml_tier": "tfidf",   # tier key required by updated _fuse_category
        }
        result = _fuse_category(rule, ml)
        assert result["category_code"] == "drainage"
        # source is "<tier>_fusion" where tier comes from ml["ml_tier"]
        assert result["source"] == "tfidf_fusion"
        assert result["confidence"] == pytest.approx(0.525, abs=0.01)

    def test_none_ml_returns_rule_result(self):
        from apps.ml.analyzer import _fuse_category  # noqa: PLC0415
        rule = {"category_code": "sewage_issue", "confidence": 0.70}
        result = _fuse_category(rule, None)
        assert result is rule


# ---------------------------------------------------------------------------
# ModelUnavailable propagation
# ---------------------------------------------------------------------------

class TestModelUnavailableGraceful:
    def test_predict_category_unavailable_raises(self):
        from apps.ml.ml_inference import ModelUnavailable, predict_category  # noqa: PLC0415
        with pytest.raises(ModelUnavailable):
            predict_category("test text")

    def test_analyze_complaint_does_not_raise_when_ml_unavailable(self):
        """analyze_complaint must never raise even if all ML models fail."""
        from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
        # Should not raise — falls back to rule engine
        result = analyze_complaint("test complaint about road damage near junction")
        assert isinstance(result, dict)
        assert "category_code" in result

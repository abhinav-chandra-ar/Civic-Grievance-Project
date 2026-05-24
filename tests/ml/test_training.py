"""tests/ml/test_training.py

Tests for the ML training pipeline and corpus.

Design: these tests must NOT require model files on disk and must NOT be slow.
We test:
  - corpus_data: shape, balance, content validity
  - generate_corpus: augmentation output
  - train_models: model trains and produces correct shapes / classes
    (uses a tiny toy corpus to keep it fast, < 1 second)

No Django DB access; no pytest.mark.django_db needed.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# corpus_data tests
# ---------------------------------------------------------------------------

class TestCorpusData:
    def test_all_samples_non_empty(self):
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        assert len(ALL_SAMPLES) > 0

    def test_minimum_samples_per_category(self):
        """Every required civic category must have at least 5 seed samples."""
        from collections import Counter  # noqa: PLC0415
        from apps.ml.training.corpus_data import ALL_SAMPLES, CATEGORY_CODES  # noqa: PLC0415
        counts = Counter(cat for _, cat, _, _ in ALL_SAMPLES)
        civic = [c for c in CATEGORY_CODES if c not in {"spam", "no_category"}]
        for cat in civic:
            assert counts[cat] >= 5, f"Category '{cat}' has only {counts[cat]} samples"

    def test_all_category_codes_valid(self):
        from apps.ml.training.corpus_data import ALL_SAMPLES, CATEGORY_CODES  # noqa: PLC0415
        for _, cat, _, _ in ALL_SAMPLES:
            assert cat in CATEGORY_CODES, f"Unknown category: {cat}"

    def test_all_priority_levels_valid(self):
        from apps.ml.training.corpus_data import ALL_SAMPLES, PRIORITY_LEVELS  # noqa: PLC0415
        for _, _, prio, _ in ALL_SAMPLES:
            assert prio in PRIORITY_LEVELS, f"Unknown priority: {prio}"

    def test_all_texts_non_empty(self):
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        for text, _, _, _ in ALL_SAMPLES:
            assert text.strip(), "Found empty text in corpus"

    def test_all_texts_min_length(self):
        """Each seed sample must have at least 10 characters."""
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        for text, cat, _, _ in ALL_SAMPLES:
            assert len(text) >= 10, f"Too short ({cat}): {text!r}"

    def test_all_nine_civic_categories_represented(self):
        from collections import Counter  # noqa: PLC0415
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        cats = {cat for _, cat, _, _ in ALL_SAMPLES}
        required = {
            "water_supply", "drainage", "sewage_issue", "solid_waste",
            "road_damage", "electrical_hazard", "street_light",
            "tree_fall", "illegal_construction",
        }
        missing = required - cats
        assert not missing, f"Missing categories in corpus: {missing}"

    def test_spam_and_no_category_represented(self):
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        cats = {cat for _, cat, _, _ in ALL_SAMPLES}
        assert "spam" in cats
        assert "no_category" in cats

    def test_multilingual_samples_present(self):
        """At least some samples should contain Malayalam Unicode characters."""
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        _MALAYALAM_RANGE = (0x0D00, 0x0D7F)
        has_malayalam = any(
            any(_MALAYALAM_RANGE[0] <= ord(c) <= _MALAYALAM_RANGE[1] for c in text)
            for text, _, _, _ in ALL_SAMPLES
        )
        assert has_malayalam, "No Malayalam-script samples found in corpus"

    def test_manglish_samples_present(self):
        """At least some samples should contain Manglish signal words."""
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        manglish_words = {"vellam", "kuzhal", "vannu", "aanu", "illa", "maram"}
        has_manglish = any(
            any(word in text.lower() for word in manglish_words)
            for text, _, _, _ in ALL_SAMPLES
        )
        assert has_manglish, "No Manglish samples found in corpus"


# ---------------------------------------------------------------------------
# generate_corpus tests
# ---------------------------------------------------------------------------

class TestGenerateCorpus:
    def test_expand_increases_count(self):
        from apps.ml.training.corpus_data import WATER_SUPPLY  # noqa: PLC0415
        from apps.ml.training.generate_corpus import expand_corpus  # noqa: PLC0415
        expanded = expand_corpus(WATER_SUPPLY, augment_factor=3)
        # Should be approximately 3× the seed count
        assert len(expanded) >= len(WATER_SUPPLY) * 2

    def test_labels_preserved_after_expansion(self):
        from apps.ml.training.corpus_data import ROAD_DAMAGE  # noqa: PLC0415
        from apps.ml.training.generate_corpus import expand_corpus  # noqa: PLC0415
        expanded = expand_corpus(ROAD_DAMAGE, augment_factor=2)
        for text, cat, prio, dept in expanded:
            assert cat == "road_damage"
            assert isinstance(text, str)
            assert text.strip()

    def test_spam_gets_lower_augmentation(self):
        from apps.ml.training.corpus_data import SPAM  # noqa: PLC0415
        from apps.ml.training.generate_corpus import expand_corpus  # noqa: PLC0415
        # factor=4 but spam only gets ×2
        expanded = expand_corpus(SPAM, augment_factor=4)
        # Should be ~2× not ~4×
        assert len(expanded) < len(SPAM) * 4

    def test_to_csv_string_has_header(self):
        from apps.ml.training.corpus_data import WATER_SUPPLY  # noqa: PLC0415
        from apps.ml.training.generate_corpus import expand_corpus, to_csv_string  # noqa: PLC0415
        samples = expand_corpus(WATER_SUPPLY[:3], augment_factor=2)
        csv_str = to_csv_string(samples)
        assert csv_str.startswith("text,category_code,priority,department_code")

    def test_expanded_corpus_total_size(self):
        from apps.ml.training.corpus_data import ALL_SAMPLES  # noqa: PLC0415
        from apps.ml.training.generate_corpus import expand_corpus  # noqa: PLC0415
        expanded = expand_corpus(ALL_SAMPLES, augment_factor=4)
        # 9 civic + spam + no_category, civic ×4, others ×2 — should be substantial
        assert len(expanded) >= 300


# ---------------------------------------------------------------------------
# train_models tests (uses tiny toy corpus, fast)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainModels:
    """Fast smoke tests for the training pipeline.

    Each test trains on a tiny 20-sample toy corpus to stay under 5 seconds.
    Marked @pytest.mark.slow so CI can skip in --fast mode.
    """

    def _tiny_corpus(self) -> list:
        """Minimal corpus: 2 samples per category × 4 categories."""
        return [
            ("water pipe burst leaking", "water_supply", "high", "water"),
            ("no water supply three days", "water_supply", "medium", "water"),
            ("road pothole large dangerous", "road_damage", "high", "roads"),
            ("broken asphalt potholes road", "road_damage", "medium", "roads"),
            ("sewage overflow manhole", "sewage_issue", "urgent", "sewage"),
            ("sewer blocked smell", "sewage_issue", "high", "sewage"),
            ("buy medicine cheap", "spam", "low", "none"),
            ("test test hello", "spam", "low", "none"),
            ("drain blocked overflow", "drainage", "high", "drainage"),
            ("flooded street drain", "drainage", "medium", "drainage"),
            ("electric wire fallen road", "electrical_hazard", "critical", "electricity"),
            ("live wire dangerous shock", "electrical_hazard", "urgent", "electricity"),
            ("garbage not collected", "solid_waste", "medium", "sanitation"),
            ("waste dump overflow bin", "solid_waste", "medium", "sanitation"),
            ("street light broken dark", "street_light", "medium", "electricity"),
            ("lamp post not working", "street_light", "low", "electricity"),
            ("tree fallen road blocking", "tree_fall", "urgent", "parks"),
            ("large branch fell road", "tree_fall", "high", "parks"),
            ("illegal construction permit", "illegal_construction", "high", "planning"),
            ("encroachment public land", "illegal_construction", "medium", "planning"),
        ]

    def test_category_model_trains_without_error(self):
        from apps.ml.training.train_models import train_category_model  # noqa: PLC0415
        corpus = self._tiny_corpus()
        texts  = [t for t, _, _, _ in corpus]
        labels = [c for _, c, _, _ in corpus]
        pipeline, le = train_category_model(texts, labels, evaluate=False)
        assert pipeline is not None
        assert hasattr(le, "classes_")
        assert len(le.classes_) > 0

    def test_priority_model_trains_without_error(self):
        from apps.ml.training.train_models import train_priority_model  # noqa: PLC0415
        corpus = self._tiny_corpus()
        texts  = [t for t, _, _, _ in corpus]
        labels = [p for _, _, p, _ in corpus]
        pipeline, le = train_priority_model(texts, labels, evaluate=False)
        assert pipeline is not None

    def test_spam_model_trains_without_error(self):
        from apps.ml.training.train_models import train_spam_model  # noqa: PLC0415
        corpus = self._tiny_corpus()
        texts  = [t for t, _, _, _ in corpus]
        labels = ["spam" if c == "spam" else "not_spam" for _, c, _, _ in corpus]
        pipeline, le = train_spam_model(texts, labels, evaluate=False)
        assert "spam" in list(le.classes_)
        assert "not_spam" in list(le.classes_)

    def test_duplicate_vectorizer_fits_without_error(self):
        from apps.ml.training.train_models import fit_duplicate_vectorizer  # noqa: PLC0415
        corpus = self._tiny_corpus()
        texts  = [t for t, _, _, _ in corpus]
        vec = fit_duplicate_vectorizer(texts)
        assert len(vec.vocabulary_) > 0

    def test_predict_after_training(self):
        from apps.ml.training.train_models import train_category_model  # noqa: PLC0415
        corpus = self._tiny_corpus()
        texts  = [t for t, _, _, _ in corpus]
        labels = [c for _, c, _, _ in corpus]
        pipeline, le = train_category_model(texts, labels, evaluate=False)
        # Should not raise
        preds = pipeline.predict(["water pipe leak"])
        assert len(preds) == 1
        pred_label = le.inverse_transform(preds)[0]
        assert pred_label in le.classes_


# ---------------------------------------------------------------------------
# train_all integration smoke test (saves to temp dir)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainAll:
    def test_train_all_saves_artifacts(self, tmp_path):
        """train_all() should produce 7 joblib files in models dir."""
        from unittest.mock import patch  # noqa: PLC0415
        from apps.ml.training import train_models  # noqa: PLC0415

        with patch.object(train_models, "_MODELS_DIR", tmp_path):
            artifacts = train_models.train_all(
                evaluate=False,
                augment_factor=2,
                save=True,
            )

        expected_files = {
            "category_pipeline.joblib",
            "priority_pipeline.joblib",
            "department_pipeline.joblib",
            "spam_pipeline.joblib",
            "language_pipeline.joblib",
            "duplicate_vectorizer.joblib",
            "label_encoders.joblib",
        }
        saved = {p.name for p in tmp_path.iterdir()}
        assert expected_files == saved

    def test_train_all_returns_artifact_dict(self, tmp_path):
        from unittest.mock import patch  # noqa: PLC0415
        from apps.ml.training import train_models  # noqa: PLC0415

        with patch.object(train_models, "_MODELS_DIR", tmp_path):
            artifacts = train_models.train_all(
                evaluate=False,
                augment_factor=2,
                save=False,
            )

        assert "category_pipeline" in artifacts
        assert "duplicate_vectorizer" in artifacts
        assert "label_encoders" in artifacts

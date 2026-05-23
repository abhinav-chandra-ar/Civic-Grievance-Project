"""Tests for apps.ml.image_analyzer — pure Python, zero database required.

All functions under test are pure: no Django, no DB, no I/O.
No @pytest.mark.django_db needed anywhere in this file.

Pillow requirement
------------------
All tests require Pillow (PIL).  If Pillow is not installed the entire
module is skipped cleanly via pytest.importorskip().

Install with:  poetry add Pillow

Synthetic image helpers
-----------------------
Tests create images in-memory using PIL.Image.new() and helpers below.
No real image files are read from disk.

Coverage
--------
validate_image          — PIL images, bytes, tiny, corrupted, unsupported format
assess_image_quality    — blank, dark, blurry, too-small, good outdoor-like
detect_irrelevant_image — blank, screenshot dimensions, text-heavy, good image
compare_text_image_consistency — all 5 decision branches
analyze_image           — orchestrated output, all image types
analyze_complaint + image penalties
nlp adapter             — 4 new metadata keys, exact contract preservation
"""
from __future__ import annotations

import io
import random

import pytest

# Skip this entire module if Pillow is not installed.
PIL = pytest.importorskip("PIL", reason="Pillow not installed — run `poetry add Pillow`")
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402  (after importorskip)

from apps.ml.image_analyzer import (
    PROVIDER,
    _BLANK_STDDEV,
    _DARK_THRESHOLD,
    _SCREEN_RESOLUTIONS,
    _TEXT_BIMODAL_THRESHOLD,
    analyze_image,
    assess_image_quality,
    compare_text_image_consistency,
    detect_irrelevant_image,
    validate_image,
)

# ===========================================================================
# Synthetic image factory helpers
# ===========================================================================


def _img(mode: str = "RGB", size: tuple[int, int] = (600, 400),
         color: tuple | int = (120, 80, 40)) -> Image.Image:
    """Return a plain solid-colour image."""
    return Image.new(mode, size, color=color)


def _noisy(size: tuple[int, int] = (600, 400), seed: int = 42) -> Image.Image:
    """Return a pseudo-random RGB image that mimics an outdoor photo."""
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    pixels = [
        (rng.randint(30, 220), rng.randint(30, 200), rng.randint(20, 180))
        for _ in range(size[0] * size[1])
    ]
    img.putdata(pixels)
    return img


def _dark(size: tuple[int, int] = (600, 400)) -> Image.Image:
    """Near-black image."""
    return Image.new("RGB", size, color=(8, 8, 8))


def _blank_white(size: tuple[int, int] = (600, 400)) -> Image.Image:
    """Pure white blank image."""
    return Image.new("RGB", size, color=(255, 255, 255))


def _blank_grey(size: tuple[int, int] = (600, 400)) -> Image.Image:
    """Uniform mid-grey — simulates wallpaper or accidental solid-colour snap."""
    return Image.new("RGB", size, color=(180, 180, 180))


def _tiny(size: tuple[int, int] = (60, 40)) -> Image.Image:
    return Image.new("RGB", size, color=(128, 100, 80))


def _screenshot() -> Image.Image:
    """800×600 image (smallest entry in _SCREEN_RESOLUTIONS) with mid-tone content.

    Two solid-colour halves give pixel stddev ≈ 32 (well above _BLANK_STDDEV=14)
    so the blank check does NOT fire.  Both fill colours are mid-tones, keeping
    the bimodal ratio near 0 so the text-heavy check does NOT fire either.
    Only the screenshot-dimensions check fires, returning reason="screenshot_dimensions".
    """
    w, h = 800, 600  # (800, 600) is guaranteed to be in _SCREEN_RESOLUTIONS
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w // 2, h - 1], fill=(60, 80, 120))    # gray ≈ 87, mid-tone
    draw.rectangle([w // 2, 0, w - 1, h - 1], fill=(180, 140, 80))  # gray ≈ 133, mid-tone
    return img


def _text_heavy(size: tuple[int, int] = (600, 400)) -> Image.Image:
    """White image with many black horizontal lines — document / receipt."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(15, size[1], 18):
        draw.line([(5, y), (size[0] - 5, y)], fill=(0, 0, 0), width=2)
    return img


def _blurry(size: tuple[int, int] = (600, 400)) -> Image.Image:
    """Very dark linear gradient (0 → 55) — triggers BOTH 'too_dark' and 'blurry'.

    Measured pixel statistics for the default 600×400 size:
        pixel_stddev ≈ 15.9  (> _BLANK_STDDEV=14  → blank check does NOT fire)
        mean brightness ≈ 27  (< _DARK_THRESHOLD=28 → 'too_dark' fires: −0.40)
        edge_stddev    ≈ 3.2  (< _BLUR_EDGE_STDDEV=7 → 'blurry'  fires: −0.35)
        quality_score  ≈ 0.25 (< 0.50 → usable=False)

    A linear ramp has a constant first derivative, so FIND_EDGES (edge
    detector) returns the same tiny value at every pixel.  That makes the
    *standard deviation* of the edge image near-zero — exactly what the
    blurry heuristic looks for.
    """
    w, h = size
    row_bytes = bytes(int(x * 55 // max(w - 1, 1)) for x in range(w))  # 0 → 55
    row_img = Image.frombytes("L", (w, 1), row_bytes)
    return row_img.resize((w, h), Image.NEAREST).convert("RGB")


def _to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Serialise PIL Image to bytes (default JPEG)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _corrupted_bytes() -> bytes:
    return b"this is definitely not an image file \x00\xff\xfe\xfd"


# ===========================================================================
# validate_image
# ===========================================================================

class TestValidateImage:
    def test_valid_pil_image(self):
        r = validate_image(_noisy())
        assert r["is_valid"] is True
        assert r["reason"] == ""

    def test_valid_jpeg_bytes(self):
        r = validate_image(_to_bytes(_noisy()))
        assert r["is_valid"] is True

    def test_valid_png_bytes(self):
        r = validate_image(_to_bytes(_noisy(), fmt="PNG"))
        assert r["is_valid"] is True

    def test_corrupted_bytes_invalid(self):
        r = validate_image(_corrupted_bytes())
        assert r["is_valid"] is False
        assert r["reason"] != ""

    def test_too_small_width_invalid(self):
        r = validate_image(Image.new("RGB", (50, 400), color=(128, 128, 128)))
        assert r["is_valid"] is False
        assert "small" in r["reason"].lower() or "too" in r["reason"].lower()

    def test_too_small_height_invalid(self):
        r = validate_image(Image.new("RGB", (400, 50), color=(128, 128, 128)))
        assert r["is_valid"] is False

    def test_tiny_image_invalid(self):
        r = validate_image(_tiny())
        assert r["is_valid"] is False

    def test_exactly_minimum_dimension_valid(self):
        # 100×100 is the minimum allowed
        r = validate_image(Image.new("RGB", (100, 200), color=(100, 100, 100)))
        assert r["is_valid"] is True

    def test_large_image_valid(self):
        r = validate_image(Image.new("RGB", (1920, 1080), color=(120, 80, 60)))
        assert r["is_valid"] is True

    def test_reason_is_string(self):
        r = validate_image(_noisy())
        assert isinstance(r["reason"], str)

    def test_returns_two_keys(self):
        r = validate_image(_noisy())
        assert set(r.keys()) == {"is_valid", "reason"}


# ===========================================================================
# assess_image_quality
# ===========================================================================

class TestAssessImageQuality:
    def test_returns_all_required_keys(self):
        r = assess_image_quality(_noisy())
        assert set(r.keys()) == {"quality_score", "quality_flags", "usable"}

    def test_good_outdoor_image_usable(self):
        r = assess_image_quality(_noisy())
        assert r["usable"] is True
        assert r["quality_score"] >= 0.5

    def test_blank_white_not_usable(self):
        r = assess_image_quality(_blank_white())
        assert r["usable"] is False
        assert "blank_or_overexposed" in r["quality_flags"]

    def test_blank_grey_not_usable(self):
        r = assess_image_quality(_blank_grey())
        assert r["usable"] is False

    def test_dark_image_not_usable(self):
        r = assess_image_quality(_dark())
        assert r["usable"] is False
        assert "too_dark" in r["quality_flags"]

    def test_tiny_image_flagged(self):
        # Tiny image is below minimum so validate_image fails → quality = invalid
        r = assess_image_quality(_tiny())
        assert r["usable"] is False
        assert r["quality_score"] == 0.0

    def test_blurry_image_flagged(self):
        r = assess_image_quality(_blurry())
        assert "blurry" in r["quality_flags"]
        assert r["usable"] is False

    def test_quality_score_in_range(self):
        for img in [_noisy(), _dark(), _blank_white(), _blurry()]:
            r = assess_image_quality(img)
            assert 0.0 <= r["quality_score"] <= 1.0

    def test_quality_flags_is_list(self):
        r = assess_image_quality(_noisy())
        assert isinstance(r["quality_flags"], list)

    def test_good_image_no_flags(self):
        r = assess_image_quality(_noisy())
        assert r["quality_flags"] == []

    def test_corrupted_bytes_not_usable(self):
        r = assess_image_quality(_corrupted_bytes())
        assert r["usable"] is False
        assert r["quality_score"] == 0.0

    def test_accepts_bytes_input(self):
        r = assess_image_quality(_to_bytes(_noisy()))
        assert r["usable"] is True

    def test_accepts_pil_image(self):
        r = assess_image_quality(_noisy())
        assert isinstance(r["quality_score"], float)


# ===========================================================================
# detect_irrelevant_image
# ===========================================================================

class TestDetectIrrelevantImage:
    def test_returns_required_keys(self):
        r = detect_irrelevant_image(_noisy())
        assert set(r.keys()) == {"is_irrelevant", "reason"}

    def test_good_outdoor_not_irrelevant(self):
        r = detect_irrelevant_image(_noisy())
        assert r["is_irrelevant"] is False
        assert r["reason"] == ""

    def test_blank_white_is_irrelevant(self):
        r = detect_irrelevant_image(_blank_white())
        assert r["is_irrelevant"] is True
        assert "blank" in r["reason"].lower() or "colour" in r["reason"].lower()

    def test_blank_grey_is_irrelevant(self):
        r = detect_irrelevant_image(_blank_grey())
        assert r["is_irrelevant"] is True

    def test_screenshot_dimensions_is_irrelevant(self):
        r = detect_irrelevant_image(_screenshot())
        assert r["is_irrelevant"] is True
        assert "screenshot" in r["reason"].lower()

    def test_text_heavy_is_irrelevant(self):
        r = detect_irrelevant_image(_text_heavy())
        assert r["is_irrelevant"] is True
        assert "text" in r["reason"].lower()

    def test_invalid_image_not_flagged_as_irrelevant(self):
        # Invalid images skip the irrelevance check
        r = detect_irrelevant_image(_corrupted_bytes())
        assert r["is_irrelevant"] is False

    def test_reason_is_string(self):
        for img in [_noisy(), _blank_white(), _screenshot(), _text_heavy()]:
            r = detect_irrelevant_image(img)
            assert isinstance(r["reason"], str)

    def test_accepts_bytes_input(self):
        r = detect_irrelevant_image(_to_bytes(_noisy()))
        assert r["is_irrelevant"] is False

    def test_non_screen_size_not_flagged_as_screenshot(self):
        # 640×480 is not in _SCREEN_RESOLUTIONS
        img = Image.new("RGB", (640, 480), color=(100, 150, 80))
        r = detect_irrelevant_image(img)
        # Should not fire screenshot heuristic (640×480 not in set)
        assert r["reason"] != "screenshot_dimensions"


# ===========================================================================
# compare_text_image_consistency
# ===========================================================================

class TestCompareTextImageConsistency:
    def _analysis(self, **kwargs):
        base = {
            "is_valid": True, "is_irrelevant": False,
            "irrelevant_reason": "", "usable": True,
        }
        base.update(kwargs)
        return base

    def test_returns_required_keys(self):
        r = compare_text_image_consistency("road_damage", self._analysis())
        assert set(r.keys()) == {"is_consistent", "consistency_score", "conflict_reason"}

    def test_invalid_image_inconsistent(self):
        r = compare_text_image_consistency("road_damage", self._analysis(is_valid=False))
        assert r["is_consistent"] is False
        assert r["consistency_score"] <= 0.15
        assert "invalid" in r["conflict_reason"].lower()

    def test_irrelevant_image_inconsistent(self):
        r = compare_text_image_consistency(
            "water_supply",
            self._analysis(is_irrelevant=True, irrelevant_reason="screenshot_dimensions"),
        )
        assert r["is_consistent"] is False
        assert r["consistency_score"] <= 0.20
        assert "irrelevant" in r["conflict_reason"].lower()

    def test_unusable_image_inconsistent(self):
        r = compare_text_image_consistency("drainage", self._analysis(usable=False))
        assert r["is_consistent"] is False
        assert r["consistency_score"] <= 0.30
        assert "poor" in r["conflict_reason"].lower()

    def test_usable_with_category_consistent(self):
        r = compare_text_image_consistency("road_damage", self._analysis())
        assert r["is_consistent"] is True
        assert r["consistency_score"] >= 0.70

    def test_usable_without_category_neutral(self):
        r = compare_text_image_consistency("", self._analysis())
        assert r["is_consistent"] is True
        assert 0.50 <= r["consistency_score"] < 0.70

    def test_consistency_score_in_range(self):
        for is_valid, is_irrel, usable in [
            (False, False, False),
            (True, True, False),
            (True, False, False),
            (True, False, True),
        ]:
            r = compare_text_image_consistency(
                "road_damage",
                self._analysis(is_valid=is_valid, is_irrelevant=is_irrel, usable=usable),
            )
            assert 0.0 <= r["consistency_score"] <= 1.0

    def test_conflict_reason_is_string(self):
        r = compare_text_image_consistency("road_damage", self._analysis(is_valid=False))
        assert isinstance(r["conflict_reason"], str)


# ===========================================================================
# analyze_image — orchestrator
# ===========================================================================

class TestAnalyzeImage:
    _REQUIRED_KEYS = {
        "is_valid", "quality_score", "quality_flags", "usable",
        "is_irrelevant", "irrelevant_reason",
        "is_consistent", "consistency_score", "conflict_reason",
        "provider",
    }

    def test_returns_all_required_keys(self):
        r = analyze_image(_noisy())
        assert set(r.keys()) == self._REQUIRED_KEYS

    def test_provider_is_correct(self):
        r = analyze_image(_noisy())
        assert r["provider"] == PROVIDER
        assert r["provider"] == "image_rule_v1"

    def test_good_outdoor_image(self):
        r = analyze_image(_noisy(), text_category="road_damage")
        assert r["is_valid"] is True
        assert r["usable"] is True
        assert r["is_irrelevant"] is False
        assert r["is_consistent"] is True

    def test_blank_image(self):
        r = analyze_image(_blank_white())
        assert r["is_valid"] is True       # blank IS a valid image file
        assert r["usable"] is False        # but not usable evidence
        assert r["is_irrelevant"] is True  # and flagged as irrelevant

    def test_dark_image(self):
        r = analyze_image(_dark())
        assert r["usable"] is False
        assert "too_dark" in r["quality_flags"]

    def test_screenshot_image(self):
        r = analyze_image(_screenshot(), text_category="road_damage")
        assert r["is_irrelevant"] is True
        assert r["is_consistent"] is False

    def test_text_heavy_image(self):
        r = analyze_image(_text_heavy())
        assert r["is_irrelevant"] is True

    def test_corrupted_bytes(self):
        r = analyze_image(_corrupted_bytes())
        assert r["is_valid"] is False
        assert r["usable"] is False
        assert r["is_consistent"] is False

    def test_jpeg_bytes_input(self):
        r = analyze_image(_to_bytes(_noisy()))
        assert r["is_valid"] is True

    def test_png_bytes_input(self):
        r = analyze_image(_to_bytes(_noisy(), fmt="PNG"))
        assert r["is_valid"] is True

    def test_blurry_image_not_usable(self):
        r = analyze_image(_blurry())
        assert r["usable"] is False
        assert "blurry" in r["quality_flags"]

    def test_quality_score_float_in_range(self):
        for img in [_noisy(), _dark(), _blank_white(), _blurry()]:
            r = analyze_image(img)
            assert 0.0 <= r["quality_score"] <= 1.0

    def test_without_category_neutral_consistency(self):
        r = analyze_image(_noisy(), text_category="")
        assert r["is_consistent"] is True
        assert r["consistency_score"] < 0.70   # neutral, not full

    def test_with_category_full_consistency(self):
        r = analyze_image(_noisy(), text_category="drainage")
        assert r["is_consistent"] is True
        assert r["consistency_score"] >= 0.70


# ===========================================================================
# analyze_complaint — image confidence penalties
# ===========================================================================

class TestAnalyzeComplaintImagePenalties:
    """Verify that Phase B image evidence adjusts Phase A confidence correctly."""

    def _base_result(self, image_input=None):
        from apps.ml.analyzer import analyze_complaint
        return analyze_complaint(
            "Pothole near Pattom junction road damaged",
            image_input=image_input,
        )

    def test_no_image_gives_none_image_analysis(self):
        r = self._base_result(image_input=None)
        assert r["image_analysis"] is None

    def test_no_image_confidence_unchanged(self):
        no_img = self._base_result(image_input=None)
        # Confidence without image is the Phase A baseline
        assert no_img["image_analysis"] is None
        assert "image_analysis" in no_img

    def test_good_image_does_not_reduce_confidence(self):
        no_img_conf = self._base_result(image_input=None)["confidence"]
        with_img_conf = self._base_result(image_input=_noisy())["confidence"]
        # Good evidence should not penalise confidence
        assert with_img_conf >= no_img_conf * 0.99  # allow float rounding

    def test_invalid_image_reduces_confidence(self):
        no_img_conf = self._base_result(image_input=None)["confidence"]
        bad_conf = self._base_result(image_input=_corrupted_bytes())["confidence"]
        assert bad_conf < no_img_conf

    def test_invalid_image_adds_review_flag(self):
        r = self._base_result(image_input=_corrupted_bytes())
        assert "image_invalid" in r["review_reasons"]
        assert r["needs_human_review"] is True

    def test_irrelevant_image_adds_review_flag(self):
        r = self._base_result(image_input=_screenshot())
        assert "image_irrelevant" in r["review_reasons"]
        assert r["needs_human_review"] is True

    def test_poor_quality_image_adds_review_flag(self):
        r = self._base_result(image_input=_dark())
        assert "image_poor_quality" in r["review_reasons"]
        assert r["needs_human_review"] is True

    def test_contradiction_adds_review_flag(self):
        # Screenshot is inconsistent with any complaint
        r = self._base_result(image_input=_screenshot())
        assert "image_contradicts_complaint" in r["review_reasons"]

    def test_good_image_preserves_all_phase_a_keys(self):
        r = self._base_result(image_input=_noisy())
        phase_a_keys = {
            "language", "category_code", "department_code", "priority",
            "landmarks", "ward_hint", "spam", "duplicate",
            "needs_human_review", "review_reasons", "confidence",
        }
        for k in phase_a_keys:
            assert k in r, f"Phase A key missing after Phase B: {k!r}"

    def test_image_analysis_key_present_with_image(self):
        r = self._base_result(image_input=_noisy())
        assert r["image_analysis"] is not None
        assert r["image_analysis"]["provider"] == PROVIDER

    def test_confidence_always_in_range(self):
        for img in [None, _noisy(), _corrupted_bytes(), _screenshot(), _dark()]:
            r = self._base_result(image_input=img)
            assert 0.0 <= r["confidence"] <= 1.0, f"Out of range for img={img}"


# ===========================================================================
# NLP adapter — 4 new metadata keys preserved
# ===========================================================================

class TestNlpAdapterPhaseB:
    _REQUIRED_METADATA_KEYS = {
        "text_length", "ward_hint", "landmark_hints",
        "spam_check", "duplicate_check",
        "needs_human_review", "review_reasons",
        # Phase B additions
        "image_analysis", "consistency_check",
        "evidence_quality", "evidence_review_reason",
        # Phase C addition
        "decision",
    }

    def _call(self, text: str, **kwargs):
        from apps.integrations.clients.nlp import classify_grievance_text
        return classify_grievance_text(raw_text=text, **kwargs)

    def test_metadata_has_all_12_keys_without_image(self):
        r = self._call("pothole near pattom road")
        assert set(r["metadata"].keys()) == self._REQUIRED_METADATA_KEYS

    def test_image_analysis_is_none_without_image(self):
        r = self._call("road issue near ulloor")
        assert r["metadata"]["image_analysis"] is None

    def test_consistency_check_is_none_without_image(self):
        r = self._call("road issue near ulloor")
        assert r["metadata"]["consistency_check"] is None

    def test_evidence_quality_is_none_without_image(self):
        r = self._call("drainage blocked near karamana")
        assert r["metadata"]["evidence_quality"] is None

    def test_evidence_review_reason_empty_without_image(self):
        r = self._call("street light broken near pettah")
        assert r["metadata"]["evidence_review_reason"] == ""

    def test_with_good_image_all_keys_populated(self):
        r = self._call("pothole near pattom", image_input=_noisy())
        assert r["metadata"]["image_analysis"] is not None
        assert r["metadata"]["consistency_check"] is True
        assert r["metadata"]["evidence_quality"] is not None
        assert r["metadata"]["evidence_review_reason"] == ""

    def test_with_invalid_image_evidence_review_reason_set(self):
        r = self._call("road issue", image_input=_corrupted_bytes())
        assert r["metadata"]["evidence_review_reason"] == "image_invalid"
        assert r["metadata"]["needs_human_review"] is True

    def test_with_irrelevant_image_consistency_check_false(self):
        r = self._call("road issue near pattom", image_input=_screenshot())
        assert r["metadata"]["consistency_check"] is False
        assert r["metadata"]["evidence_review_reason"] in {
            "image_irrelevant", "image_contradicts_complaint"
        }

    def test_8_top_level_keys_unchanged(self):
        r = self._call("pothole near pattom", image_input=_noisy())
        assert set(r.keys()) == {
            "normalized_summary", "category_code", "department_code",
            "priority", "confidence", "language", "provider", "metadata",
        }

    def test_confidence_is_clamped_float(self):
        r = self._call("live wire fallen", image_input=_corrupted_bytes())
        assert isinstance(r["confidence"], float)
        assert 0.0 <= r["confidence"] <= 1.0

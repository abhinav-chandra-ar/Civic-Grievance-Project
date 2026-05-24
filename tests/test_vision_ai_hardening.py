"""Hardening tests for Vision AI: image_analyzer + CLIP inference integration.

Scope
-----
Tests the pure-function layer in ``apps/ml/image_analyzer.py`` and the
CLIP integration path from ``apps/ml/vision_inference.py``.

Strategy
--------
* All heuristic tests use synthetic PIL images created with NumPy/PIL
  directly in memory — no disk files, no network calls.
* CLIP tests patch ``get_clip_engine`` so they run without a GPU or
  HuggingFace download.
* analyze_image() CLIP path is tested by injecting a mock engine return value.

Coverage map
------------
1. validate_image() — valid / corrupted / tiny inputs
2. assess_image_quality() — usable / too-dark / too-blurry / blank
3. detect_irrelevant_image() — blank, screenshot dimensions, text-heavy
4. compare_text_image_consistency() — decision table rows
5. analyze_image() — heuristic orchestration (end-to-end without CLIP)
6. analyze_image() — CLIP path mocked (consistency_verdict populated)
7. analyze_image() — CLIP crash fallback (heuristic provider preserved)
8. classify_civic_issue_from_image() — CLIP unavailable path
"""
from __future__ import annotations

import io
import struct
import zlib
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pillow availability guard
# ---------------------------------------------------------------------------

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PIL_AVAILABLE,
    reason="Pillow not installed — skipping vision tests",
)

from apps.ml.image_analyzer import (
    analyze_image,
    assess_image_quality,
    classify_civic_issue_from_image,
    compare_text_image_consistency,
    detect_irrelevant_image,
    validate_image,
)

# ---------------------------------------------------------------------------
# Synthetic image factories
# ---------------------------------------------------------------------------

def _solid_image(width=200, height=200, color=(200, 200, 200)) -> bytes:
    """Return PNG bytes for a solid-colour image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _noisy_image(width=300, height=300) -> bytes:
    """Return PNG bytes for a noisy natural-looking image with high std-dev."""
    import random
    img = Image.new("RGB", (width, height))
    pixels = [(random.randint(30, 220), random.randint(30, 220), random.randint(30, 220))
              for _ in range(width * height)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _dark_image(width=200, height=200) -> bytes:
    """Return PNG bytes for an almost-black image (brightness < 28)."""
    img = Image.new("RGB", (width, height), color=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _screenshot_image() -> bytes:
    """Return PNG bytes at 800×600 (a screen resolution in _SCREEN_RESOLUTIONS).

    Uses ±60 variation around mid-grey so grayscale stddev ≈ 35 > _BLANK_STDDEV=14,
    bypassing the blank check and letting the screenshot-dimensions check fire.
    800×600 is chosen over 1920×1080 to keep pixel count manageable in CI.
    """
    import random
    img = Image.new("RGB", (800, 600))
    pixels = [(
        128 + random.randint(-60, 60),
        128 + random.randint(-60, 60),
        128 + random.randint(-60, 60),
    ) for _ in range(800 * 600)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tiny_image() -> bytes:
    """Return PNG bytes too small to be valid (50×50)."""
    img = Image.new("RGB", (50, 50), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _corrupt_bytes() -> bytes:
    """Return bytes that are not a valid image."""
    return b"not_an_image_at_all_xyz"


# ---------------------------------------------------------------------------
# 1. validate_image()
# ---------------------------------------------------------------------------

def test_validate_image_valid_noisy():
    result = validate_image(_noisy_image())
    assert result["is_valid"] is True


def test_validate_image_corrupt():
    result = validate_image(_corrupt_bytes())
    assert result["is_valid"] is False
    assert "reason" in result


def test_validate_image_too_small():
    result = validate_image(_tiny_image())
    assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# 2. assess_image_quality()
# ---------------------------------------------------------------------------

def test_assess_quality_noisy_image_is_usable():
    result = assess_image_quality(_noisy_image())
    assert result["usable"] is True
    assert isinstance(result["quality_score"], float)
    assert 0.0 <= result["quality_score"] <= 1.0


def test_assess_quality_dark_image_not_usable():
    result = assess_image_quality(_dark_image())
    assert result["usable"] is False
    assert "too_dark" in result["quality_flags"]


def test_assess_quality_blank_image_not_usable():
    # Solid colour → near-zero std-dev → blank flag
    result = assess_image_quality(_solid_image(color=(180, 180, 180)))
    assert result["usable"] is False
    assert any("blank" in flag or "uniform" in flag for flag in result["quality_flags"]), (
        f"Expected blank/uniform flag, got: {result['quality_flags']}"
    )


# ---------------------------------------------------------------------------
# 3. detect_irrelevant_image()
# ---------------------------------------------------------------------------

def test_detect_irrelevant_blank_image():
    result = detect_irrelevant_image(_solid_image(color=(200, 200, 200)))
    assert result["is_irrelevant"] is True
    assert "blank" in result["reason"].lower() or "solid" in result["reason"].lower()


def test_detect_irrelevant_screenshot_dimensions():
    result = detect_irrelevant_image(_screenshot_image())
    assert result["is_irrelevant"] is True
    assert "screenshot" in result["reason"].lower()


def test_detect_irrelevant_noisy_image_is_relevant():
    result = detect_irrelevant_image(_noisy_image())
    assert result["is_irrelevant"] is False


# ---------------------------------------------------------------------------
# 4. compare_text_image_consistency() — decision table
# ---------------------------------------------------------------------------

def test_consistency_invalid_image():
    result = compare_text_image_consistency(
        "road_damage",
        {"is_valid": False, "is_irrelevant": False, "usable": False, "irrelevant_reason": ""},
    )
    assert result["is_consistent"] is False
    assert result["consistency_score"] <= 0.15


def test_consistency_irrelevant_image():
    result = compare_text_image_consistency(
        "road_damage",
        {"is_valid": True, "is_irrelevant": True, "usable": False, "irrelevant_reason": "blank_or_solid_colour"},
    )
    assert result["is_consistent"] is False
    assert "irrelevant" in result["conflict_reason"].lower()


def test_consistency_unusable_image():
    result = compare_text_image_consistency(
        "road_damage",
        {"is_valid": True, "is_irrelevant": False, "usable": False, "irrelevant_reason": ""},
    )
    assert result["is_consistent"] is False
    assert result["consistency_score"] <= 0.30


def test_consistency_valid_usable_with_category():
    result = compare_text_image_consistency(
        "road_damage",
        {"is_valid": True, "is_irrelevant": False, "usable": True, "irrelevant_reason": ""},
    )
    assert result["is_consistent"] is True
    assert result["consistency_score"] >= 0.70


def test_consistency_valid_usable_no_category():
    result = compare_text_image_consistency(
        "",  # no category
        {"is_valid": True, "is_irrelevant": False, "usable": True, "irrelevant_reason": ""},
    )
    assert result["is_consistent"] is True
    assert 0.50 <= result["consistency_score"] <= 0.65


# ---------------------------------------------------------------------------
# 5. analyze_image() — heuristic path (no CLIP)
# ---------------------------------------------------------------------------

# CLIP is a lazy import inside analyze_image() via:
#   from apps.ml.vision_inference import get_clip_engine
# Patching must target the source module, not the importer.
_PATCH_CLIP_ENGINE = "apps.ml.vision_inference.get_clip_engine"


def test_analyze_image_returns_all_required_keys():
    # Suppress live CLIP so the test only verifies key presence, not CLIP output
    mock_engine = MagicMock()
    mock_engine.is_ready = False
    with patch(_PATCH_CLIP_ENGINE, return_value=mock_engine):
        result = analyze_image(_noisy_image(), text_category="road_damage")
    required_keys = {
        "is_valid", "quality_score", "quality_flags", "usable",
        "is_irrelevant", "irrelevant_reason",
        "is_consistent", "consistency_score", "conflict_reason",
        "provider",
        "vision_class", "vision_confidence", "vision_all_scores",
        "consistency_verdict", "fraud_flags", "vision_provider",
    }
    missing = required_keys - set(result)
    assert not missing, f"analyze_image() missing keys: {missing}"


def test_analyze_image_blank_flagged_as_irrelevant():
    result = analyze_image(_solid_image(), text_category="road_damage")
    assert result["is_valid"] is True
    assert result["is_irrelevant"] is True


def test_analyze_image_valid_noisy_is_consistent():
    """Heuristic path: a valid, usable image with a known category → is_consistent=True.

    CLIP is running live in this environment and would return 'contradicts' for
    random noise against 'road_damage'.  We suppress CLIP to isolate the
    heuristic consistency check (which only considers validity + usability).
    """
    mock_engine = MagicMock()
    mock_engine.is_ready = False  # disabled → heuristic path used
    with patch(_PATCH_CLIP_ENGINE, return_value=mock_engine):
        result = analyze_image(_noisy_image(), text_category="road_damage")
    assert result["is_valid"] is True
    assert result["is_consistent"] is True


# ---------------------------------------------------------------------------
# 6. analyze_image() — CLIP path (mocked engine)
# ---------------------------------------------------------------------------

def _make_mock_engine():
    engine = MagicMock()
    engine.is_ready = True

    vis = MagicMock()
    vis.predicted_class = "road_damage"
    vis.confidence = 0.85
    vis.all_scores = {"road_damage": 0.85, "garbage_dump": 0.05, "water_leak": 0.10}
    vis.provider = "clip_vit_b32"
    engine.classify_image.return_value = vis

    cons = MagicMock()
    cons.verdict = "supports"
    engine.check_consistency.return_value = cons

    fraud = MagicMock()
    fraud.flags = []
    engine.detect_fraud_signals.return_value = fraud

    return engine


def test_analyze_image_clip_path_sets_vision_class():
    mock_engine = _make_mock_engine()
    with patch(_PATCH_CLIP_ENGINE, return_value=mock_engine):
        result = analyze_image(_noisy_image(), text_category="road_damage")

    assert result["vision_class"] == "road_damage"
    assert result["vision_confidence"] == 0.85
    assert result["vision_provider"] == "clip_vit_b32"


def test_analyze_image_clip_path_sets_consistency_verdict():
    mock_engine = _make_mock_engine()
    with patch(_PATCH_CLIP_ENGINE, return_value=mock_engine):
        result = analyze_image(_noisy_image(), text_category="road_damage")

    assert result["consistency_verdict"] == "supports"


def test_analyze_image_clip_contradiction_sets_fraud_flag():
    mock_engine = _make_mock_engine()
    # Engine says image is garbage but category is road_damage
    fraud = MagicMock()
    fraud.flags = ["category_mismatch"]
    mock_engine.detect_fraud_signals.return_value = fraud

    cons = MagicMock()
    cons.verdict = "contradicts"
    mock_engine.check_consistency.return_value = cons

    with patch(_PATCH_CLIP_ENGINE, return_value=mock_engine):
        result = analyze_image(_noisy_image(), text_category="road_damage")

    assert result["consistency_verdict"] == "contradicts"
    assert "category_mismatch" in result["fraud_flags"]


# ---------------------------------------------------------------------------
# 7. analyze_image() — CLIP crash fallback
# ---------------------------------------------------------------------------

def test_analyze_image_clip_crash_falls_back_to_heuristic():
    """If CLIP raises, vision_provider must indicate heuristic fallback."""
    with patch(_PATCH_CLIP_ENGINE, side_effect=RuntimeError("CLIP is down")):
        result = analyze_image(_noisy_image(), text_category="road_damage")

    # Must still have all keys
    assert "vision_class" in result
    # Provider must indicate fallback
    assert "heuristic" in str(result["vision_provider"]).lower()


# ---------------------------------------------------------------------------
# 8. classify_civic_issue_from_image() — CLIP unavailable path
# ---------------------------------------------------------------------------

def test_classify_civic_issue_unavailable():
    # classify_civic_issue_from_image() catches ALL exceptions from get_clip_engine
    # and returns the unavailable payload.
    with patch(_PATCH_CLIP_ENGINE, side_effect=RuntimeError("no clip")):
        result = classify_civic_issue_from_image(_noisy_image())

    assert result["vision_provider"] == "unavailable"
    assert result["vision_class"] is None
    assert result["is_civic_image"] is False


def test_classify_civic_issue_clip_path():
    mock_engine = MagicMock()
    mock_engine.is_ready = True
    vis = MagicMock()
    vis.predicted_class = "garbage_dump"
    vis.confidence = 0.78
    vis.all_scores = {"garbage_dump": 0.78}
    vis.mapped_category = "waste_management"
    vis.is_civic = True
    vis.provider = "clip_vit_b32"
    mock_engine.classify_image.return_value = vis

    with patch(_PATCH_CLIP_ENGINE, return_value=mock_engine):
        result = classify_civic_issue_from_image(_noisy_image())

    assert result["vision_class"] == "garbage_dump"
    assert result["mapped_category"] == "waste_management"
    assert result["is_civic_image"] is True

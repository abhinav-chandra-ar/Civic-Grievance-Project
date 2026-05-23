"""apps/ml/image_analyzer.py

Evidence intelligence layer for complaint images.

Design constraints
------------------
* Pure functions only — no Django imports, no database access, no side effects.
* Requires Pillow (PIL) for pixel-level analysis.  When Pillow is not installed,
  every function returns a safe degraded response rather than raising ImportError.
* Uses only PIL + Python stdlib (io, pathlib).  No NumPy, no OpenCV, no ML models.
* Does NOT classify image content (pothole / garbage / drainage etc.) — that
  requires real CV models.  Instead it validates evidence quality and flags
  obviously junk submissions.

Image input contract
--------------------
All public functions accept ``image_input`` which may be:
  • str | pathlib.Path → absolute or relative path to an image file
  • bytes             → raw image bytes (e.g. from an uploaded file read)
  • PIL.Image.Image   → an already-opened Pillow image object

Function inventory
------------------
validate_image()                 → format / dimension sanity check
assess_image_quality()           → blur, darkness, blank detection
detect_irrelevant_image()        → screenshot / blank / text-heavy heuristics
compare_text_image_consistency() → logical consistency between complaint category
                                   and image analysis result
analyze_image()                  → orchestrator — calls all of the above
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Union

# ---------------------------------------------------------------------------
# Optional Pillow import — graceful degradation when not installed
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

PROVIDER = "image_rule_v1"

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

_MIN_DIMENSION = 100        # pixels — either axis
_MIN_PIXELS    = 20_000     # total pixel count (approx 142 × 142)
_SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "MPO"})

# Quality thresholds
_DARK_THRESHOLD       = 28.0   # mean brightness (0–255); below = too dark
_BLANK_STDDEV         = 14.0   # pixel std-dev; below = blank / uniform
_BLUR_EDGE_STDDEV     = 7.0    # std-dev of edge-filtered image; below = blurry
_SMALL_PIXEL_COUNT    = 40_000 # below = low-resolution (≈200 × 200)

# Screenshot: exact pixel dimensions that match common screen resolutions
_SCREEN_RESOLUTIONS: frozenset[tuple[int, int]] = frozenset({
    (1920, 1080), (1280, 720),  (1366, 768),  (2560, 1440),
    (3840, 2160), (1024, 768),  (800,  600),  (1600, 900),
    (1440, 900),  (1280, 800),  (1920, 1200), (2560, 1600),
    (3840, 2400), (1280, 1024), (1680, 1050),
    # Portrait equivalents
    (1080, 1920), (720, 1280),  (768, 1366),  (1440, 2560),
    # Common mobile screenshots
    (1080, 2340), (1080, 2400), (1170, 2532), (1284, 2778),
})

# Text-heavy / document: bimodal (near-black or near-white) pixel ratio
_TEXT_BIMODAL_THRESHOLD = 0.65   # fraction of pixels near 0 or 255

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NOT_AVAILABLE = {
    "is_valid":          False,
    "quality_score":     0.0,
    "quality_flags":     ["pillow_not_installed"],
    "usable":            False,
    "is_irrelevant":     False,
    "irrelevant_reason": "",
    "is_consistent":     False,
    "consistency_score": 0.0,
    "conflict_reason":   "Pillow (PIL) is not installed",
    "provider":          PROVIDER,
}


def _open_image(image_input: Any) -> "Image.Image":
    """Open an image from path, bytes, or pass through a PIL Image."""
    # Already a PIL Image (duck-type check avoids importing Image at top-level)
    if hasattr(image_input, "mode") and hasattr(image_input, "size"):
        return image_input  # type: ignore[return-value]
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))
    # str or Path
    return Image.open(Path(image_input))


def _mean_brightness(img: "Image.Image") -> float:
    """Return mean pixel value of grayscale image (0–255)."""
    gray = img.convert("L")
    return ImageStat.Stat(gray).mean[0]


def _pixel_stddev(img: "Image.Image") -> float:
    """Return std-dev of grayscale pixel values — measures overall variation."""
    gray = img.convert("L")
    return ImageStat.Stat(gray).stddev[0]


def _edge_stddev(img: "Image.Image") -> float:
    """Return std-dev of pixel values after edge detection.

    Higher value → sharper image (more edges detected).
    Lower value  → blurry image (few or no edges).
    """
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges).stddev[0]


def _bimodal_ratio(img: "Image.Image") -> float:
    """Return fraction of pixels that are near-black (< 30) or near-white (> 225).

    A high ratio indicates a text-heavy document or high-contrast screenshot.
    Natural outdoor photos have many mid-tone pixels → low ratio.
    """
    gray = img.convert("L")
    hist = gray.histogram()   # list of 256 counts
    total = sum(hist)
    if total == 0:
        return 0.0
    dark  = sum(hist[:30])    # near-black
    light = sum(hist[226:])   # near-white
    return (dark + light) / total


# ===========================================================================
# Public pure functions
# ===========================================================================


def validate_image(image_input: Any) -> dict[str, object]:
    """Validate that the image input is readable and meets minimum requirements.

    Checks
    ------
    * Pillow is available
    * Image bytes / file are not corrupted
    * Format is in the supported set (JPEG, PNG, WEBP, GIF, BMP, TIFF)
    * Both dimensions ≥ 100 px
    * Total pixel count ≥ 20 000 (≈ 142 × 142)

    Returns
    -------
    ``{"is_valid": bool, "reason": str}``
    """
    if not _PIL_AVAILABLE:
        return {"is_valid": False, "reason": "Pillow (PIL) is not installed"}

    try:
        img = _open_image(image_input)
        img.verify()   # detects truncated / corrupted files
    except UnidentifiedImageError:
        return {"is_valid": False, "reason": "unrecognised or corrupted image format"}
    except (OSError, SyntaxError, Exception) as exc:
        return {"is_valid": False, "reason": f"cannot open image: {exc}"}

    # Re-open after verify() (verify() leaves the file pointer exhausted)
    try:
        img = _open_image(image_input)
        img.load()
    except Exception as exc:
        return {"is_valid": False, "reason": f"image data corrupted: {exc}"}

    fmt = (img.format or "").upper()
    if fmt and fmt not in _SUPPORTED_FORMATS:
        return {"is_valid": False, "reason": f"unsupported format: {fmt}"}

    w, h = img.size
    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        return {
            "is_valid": False,
            "reason": f"image too small: {w}×{h} (minimum {_MIN_DIMENSION}px on each side)",
        }

    if w * h < _MIN_PIXELS:
        return {
            "is_valid": False,
            "reason": f"image resolution too low: {w * h} pixels (minimum {_MIN_PIXELS})",
        }

    return {"is_valid": True, "reason": ""}


def assess_image_quality(image_input: Any) -> dict[str, object]:
    """Assess whether the image is usable as complaint evidence.

    Quality flags
    -------------
    ``"too_small"``             — image below low-resolution threshold
    ``"too_dark"``              — mean brightness below threshold
    ``"blank_or_overexposed"``  — near-uniform pixel values (solid colour)
    ``"blurry"``                — very few edges detected
    ``"low_resolution"``        — total pixels below evidence-grade threshold

    Quality score
    -------------
    Starts at 1.0; each flag deducts a penalty; result clamped to [0.0, 1.0].
    Score ≥ 0.50 → ``"usable": True``

    Returns
    -------
    ``{"quality_score": float, "quality_flags": list[str], "usable": bool}``
    """
    if not _PIL_AVAILABLE:
        return {"quality_score": 0.0, "quality_flags": ["pillow_not_installed"], "usable": False}

    val = validate_image(image_input)
    if not val["is_valid"]:
        return {
            "quality_score": 0.0,
            "quality_flags":  ["invalid_image"],
            "usable":         False,
        }

    try:
        img = _open_image(image_input)
        img.load()
    except Exception:
        return {"quality_score": 0.0, "quality_flags": ["load_error"], "usable": False}

    flags: list[str] = []
    score = 1.0
    w, h = img.size

    # ── Pixel count / resolution ─────────────────────────────────────────
    if w * h < _SMALL_PIXEL_COUNT:
        flags.append("low_resolution")
        score -= 0.25
    if w < _MIN_DIMENSION * 2 or h < _MIN_DIMENSION * 2:
        flags.append("too_small")
        score -= 0.20

    # ── Brightness ───────────────────────────────────────────────────────
    brightness = _mean_brightness(img)
    if brightness < _DARK_THRESHOLD:
        flags.append("too_dark")
        score -= 0.40

    # ── Uniform / blank / overexposed ────────────────────────────────────
    stddev = _pixel_stddev(img)
    if stddev < _BLANK_STDDEV:
        flags.append("blank_or_overexposed")
        score -= 0.70

    # ── Blurriness (only when image has enough contrast to measure) ──────
    elif _edge_stddev(img) < _BLUR_EDGE_STDDEV:
        flags.append("blurry")
        score -= 0.35

    quality_score = round(max(0.0, min(1.0, score)), 3)
    return {
        "quality_score": quality_score,
        "quality_flags":  flags,
        "usable":         quality_score >= 0.50,
    }


def detect_irrelevant_image(image_input: Any) -> dict[str, object]:
    """Detect obviously junk evidence using conservative pixel heuristics.

    Detectable cases
    ----------------
    ``"blank_or_solid_colour"``  — near-uniform image (wallpaper, accidental snap)
    ``"screenshot_dimensions"``  — exact match to common screen resolutions
    ``"text_heavy_document"``    — high bimodal (near-black/white) pixel ratio
                                   suggests a photograph of a document or receipt

    Conservative policy
    -------------------
    If the image does not trigger any of the above rules it is reported as
    *not irrelevant*.  The function never guesses at content (no selfie / meme /
    cartoon detection from pixels alone — that requires real CV models).

    Returns
    -------
    ``{"is_irrelevant": bool, "reason": str}``
    """
    if not _PIL_AVAILABLE:
        return {"is_irrelevant": False, "reason": "Pillow not available — cannot assess"}

    val = validate_image(image_input)
    if not val["is_valid"]:
        return {"is_irrelevant": False, "reason": "image invalid — irrelevance check skipped"}

    try:
        img = _open_image(image_input)
        img.load()
    except Exception:
        return {"is_irrelevant": False, "reason": "image load error"}

    w, h = img.size

    # ── Blank / solid colour ─────────────────────────────────────────────
    if _pixel_stddev(img) < _BLANK_STDDEV:
        return {"is_irrelevant": True, "reason": "blank_or_solid_colour"}

    # ── Screenshot dimensions ────────────────────────────────────────────
    if (w, h) in _SCREEN_RESOLUTIONS:
        return {"is_irrelevant": True, "reason": "screenshot_dimensions"}

    # ── Text-heavy document ──────────────────────────────────────────────
    if _bimodal_ratio(img) >= _TEXT_BIMODAL_THRESHOLD:
        return {"is_irrelevant": True, "reason": "text_heavy_document"}

    return {"is_irrelevant": False, "reason": ""}


def compare_text_image_consistency(
    text_category: str,
    image_analysis: dict[str, object],
) -> dict[str, object]:
    """Assess logical consistency between the complaint category and image evidence.

    This is a *logical* function — it does not re-examine pixels.  It combines
    the results of prior image analysis steps to decide whether the uploaded
    evidence supports or undermines the complaint claim.

    Decision table
    --------------
    Invalid image            → inconsistent (score 0.10) — no usable evidence
    Irrelevant image         → inconsistent (score 0.15) — junk evidence
    Valid but unusable       → inconsistent (score 0.25) — too dark / blurry / small
    Usable, no category      → neutral (score 0.55) — cannot compare without category
    Usable + category exists → consistent (score 0.75) — evidence supports claim

    Returns
    -------
    ``{"is_consistent": bool, "consistency_score": float, "conflict_reason": str}``
    """
    is_valid   = bool(image_analysis.get("is_valid", False))
    is_irrel   = bool(image_analysis.get("is_irrelevant", False))
    is_usable  = bool(image_analysis.get("usable", False))
    irrel_rsn  = str(image_analysis.get("irrelevant_reason", ""))

    if not is_valid:
        return {
            "is_consistent":    False,
            "consistency_score": 0.10,
            "conflict_reason":  "image is invalid or unreadable",
        }

    if is_irrel:
        return {
            "is_consistent":    False,
            "consistency_score": 0.15,
            "conflict_reason":  f"image appears irrelevant: {irrel_rsn}",
        }

    if not is_usable:
        return {
            "is_consistent":    False,
            "consistency_score": 0.25,
            "conflict_reason":  "image quality too poor to serve as evidence",
        }

    # Image is valid, relevant, and usable.
    if not text_category:
        return {
            "is_consistent":    True,
            "consistency_score": 0.55,
            "conflict_reason":  "",
        }

    return {
        "is_consistent":    True,
        "consistency_score": 0.75,
        "conflict_reason":  "",
    }


def analyze_image(
    image_input: Any,
    text_category: str = "",
) -> dict[str, object]:
    """Orchestrate all image intelligence steps and return a unified payload.

    Parameters
    ----------
    image_input
        Path, bytes, or PIL Image — see module docstring.
    text_category
        Optional complaint category code (e.g. ``"road_damage"``) used for
        the consistency check.

    Returns
    -------
    dict with keys:
        is_valid, quality_score, quality_flags, usable,
        is_irrelevant, irrelevant_reason,
        is_consistent, consistency_score, conflict_reason,
        provider
    """
    if not _PIL_AVAILABLE:
        return dict(_NOT_AVAILABLE)

    val_result  = validate_image(image_input)
    qual_result = assess_image_quality(image_input)
    irrel_result = detect_irrelevant_image(image_input)

    # Build the partial analysis dict that compare_text_image_consistency needs.
    partial = {
        "is_valid":          val_result["is_valid"],
        "is_irrelevant":     irrel_result["is_irrelevant"],
        "irrelevant_reason": irrel_result["reason"],
        "usable":            qual_result["usable"],
    }
    cons_result = compare_text_image_consistency(text_category, partial)

    return {
        "is_valid":          val_result["is_valid"],
        "quality_score":     qual_result["quality_score"],
        "quality_flags":     qual_result["quality_flags"],
        "usable":            qual_result["usable"],
        "is_irrelevant":     irrel_result["is_irrelevant"],
        "irrelevant_reason": irrel_result["reason"],
        "is_consistent":     cons_result["is_consistent"],
        "consistency_score": cons_result["consistency_score"],
        "conflict_reason":   cons_result["conflict_reason"],
        "provider":          PROVIDER,
    }

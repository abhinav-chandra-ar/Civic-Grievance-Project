"""apps/ml/vision_inference.py

Zero-shot CLIP vision intelligence for civic grievance image analysis.

Architecture
------------
* Uses ``openai/clip-vit-base-patch32`` via HuggingFace ``transformers``.
* Singleton engine — loaded once on first call, reused for all subsequent
  requests.  No Django imports.  No side effects.  CPU-safe.
* Zero-shot classification: no custom training needed.  The model compares
  image embeddings to a set of civic-domain text prompts and returns the
  best-matching class + confidence.
* Graceful degradation: when the model is unavailable (not downloaded,
  import error, OOM), every method returns a safe degraded result.

Civic issue classes
-------------------
road_damage, pothole, garbage_dump, sewage_overflow, drainage_issue,
water_leak, fallen_tree, street_light_damage, illegal_dumping,
irrelevant_image, indoor_irrelevant, screenshot, poor_quality

Public API
----------
get_clip_engine()                → CLIPEngine singleton
CLIPEngine.classify_image()      → VisionResult(predicted_class, confidence, all_scores)
CLIPEngine.text_image_similarity() → float (cosine similarity)
CLIPEngine.check_consistency()   → ConsistencyResult(verdict, score, reason)
CLIPEngine.detect_fraud_signals() → FraudResult(is_suspicious, flags, reason)
"""
from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifier
# ---------------------------------------------------------------------------
_BACKBONE = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------------------
# Civic issue class prompts
# One descriptive sentence per class — the more specific the better.
# ---------------------------------------------------------------------------
CIVIC_PROMPTS: dict[str, str] = {
    "road_damage":          "a photo of a damaged road with cracks, crumbling edges or broken asphalt surface",
    "pothole":              "a photo of a deep pothole or road crater on a paved road or street",
    "garbage_dump":         "a photo of a large pile of garbage or trash bags dumped outdoors on a road or field",
    "sewage_overflow":      "a photo of sewage or dirty black water overflowing from a manhole onto a road",
    "drainage_issue":       "a photo of a blocked stormwater drain channel or a road flooded with rainwater",
    "water_leak":           "a photo of water bursting or leaking from a broken pipe or water gushing on a street",
    "fallen_tree":          "a photo of a large tree or heavy branch that has fallen across a road or path",
    "street_light_damage":  "a photo of a broken, damaged or missing street light or lamp post on a road",
    "illegal_dumping":      "a photo of construction debris, rubble or waste illegally dumped on a road or footpath",
    "irrelevant_image":     "a selfie, portrait photo, or personal image not showing any civic infrastructure issue",
    "indoor_irrelevant":    "a photo taken inside a house, living room, office or building interior",
    "screenshot":           "a screenshot of a mobile phone screen, computer display or digital app interface",
    "poor_quality":         "a very blurry, completely dark, overexposed or featureless image with no identifiable content",
}

# Mapping from CLIP civic class → grievance category codes (None = not a civic issue)
CLIP_CLASS_TO_CATEGORY: dict[str, str | None] = {
    "road_damage":         "road_damage",
    "pothole":             "road_damage",
    "garbage_dump":        "solid_waste",
    "sewage_overflow":     "sewage_issue",
    "drainage_issue":      "drainage",
    "water_leak":          "water_supply",
    "fallen_tree":         "tree_fall",
    "street_light_damage": "street_light",
    "illegal_dumping":     "solid_waste",
    "irrelevant_image":    None,
    "indoor_irrelevant":   None,
    "screenshot":          None,
    "poor_quality":        None,
}

# Classes that indicate non-genuine civic evidence
_SUSPICIOUS_CLASSES: frozenset[str] = frozenset({
    "irrelevant_image",
    "indoor_irrelevant",
    "screenshot",
    "poor_quality",
})

# Category family groups: when the predicted CLIP class maps to any category
# in the same family as the complaint text category, the evidence is consistent.
_CATEGORY_FAMILIES: dict[str, frozenset[str]] = {
    "road_damage":    frozenset({"road_damage", "pothole", "road_repair", "road_broken"}),
    "solid_waste":    frozenset({"solid_waste", "garbage_dump", "illegal_dumping", "waste_management"}),
    "sewage_issue":   frozenset({"sewage_issue", "sewage_overflow"}),
    "drainage":       frozenset({"drainage", "drainage_issue"}),
    "water_supply":   frozenset({"water_supply", "water_leak"}),
    "tree_fall":      frozenset({"tree_fall", "fallen_tree"}),
    "street_light":   frozenset({"street_light", "street_light_damage"}),
    "electrical_hazard": frozenset({"electrical_hazard"}),
    "illegal_construction": frozenset({"illegal_construction"}),
}


def _categories_related(clip_category: str | None, complaint_category: str) -> bool:
    """Return True when the CLIP-predicted category is in the same family
    as the complaint text category."""
    if clip_category is None:
        return False
    for fam_key, fam_set in _CATEGORY_FAMILIES.items():
        if complaint_category in fam_set and clip_category in fam_set:
            return True
        if complaint_category == fam_key:
            if clip_category in fam_set:
                return True
    # Direct match as fallback
    return clip_category == complaint_category


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VisionResult:
    """Result of zero-shot civic issue classification."""
    predicted_class:   str                  # top CLIP class
    confidence:        float                # 0.0–1.0 (softmax probability)
    all_scores:        dict[str, float]     # all class → probability mappings
    mapped_category:   str | None           # grievance category code or None
    is_civic:          bool                 # False when predicted class is a fraud indicator
    provider:          str = "clip_vit_b32"


@dataclass
class ConsistencyResult:
    """Result of text-image consistency check."""
    verdict:       str    # "supports" | "contradicts" | "uncertain"
    score:         float  # 0.0–1.0  (raw cosine similarity)
    reason:        str    # human-readable explanation


@dataclass
class FraudResult:
    """Result of suspicious evidence detection."""
    is_suspicious: bool
    flags:         list[str] = field(default_factory=list)
    reason:        str = ""


# ---------------------------------------------------------------------------
# CLIPEngine
# ---------------------------------------------------------------------------

class CLIPEngine:
    """Singleton zero-shot image classification engine using CLIP.

    Do not instantiate directly — use ``get_clip_engine()``.
    """

    def __init__(self) -> None:
        self.backbone_name: str = _BACKBONE
        self.is_ready: bool = False
        self.load_error: str = ""
        self._model: Any = None
        self._processor: Any = None
        self._class_names: list[str] = list(CIVIC_PROMPTS.keys())
        self._class_prompts: list[str] = list(CIVIC_PROMPTS.values())
        self._load()

    def _load(self) -> None:
        try:
            import torch  # noqa: PLC0415
            from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

            logger.info("Loading CLIP backbone: %s", self.backbone_name)
            self._processor = CLIPProcessor.from_pretrained(self.backbone_name)
            self._model = CLIPModel.from_pretrained(self.backbone_name)
            self._model.eval()

            # Pre-encode all civic text prompts once at load time.
            # Cached as a (N, 512) tensor — avoids re-encoding on every call.
            logger.info("Pre-encoding %d civic text prompts...", len(self._class_prompts))
            self._prompt_embeddings = self._encode_texts_raw(self._class_prompts)

            self.is_ready = True
            logger.info("CLIP backbone ready. %d classes cached.", len(self._class_names))
        except Exception as exc:  # noqa: BLE001
            self.is_ready = False
            self.load_error = str(exc)
            self._prompt_embeddings = None
            logger.warning("CLIP backbone failed to load: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_pil(self, image_input: Any) -> "Any":
        """Convert path / bytes / PIL Image to PIL Image."""
        from PIL import Image  # noqa: PLC0415
        if hasattr(image_input, "mode") and hasattr(image_input, "size"):
            return image_input
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        return Image.open(Path(image_input)).convert("RGB")

    def _encode_image(self, image_input: Any) -> "Any":
        """Return L2-normalised image embedding tensor, shape (1, 512).

        Uses vision_model + visual_projection sub-models directly, which is
        compatible with transformers >= 4.x including 5.x where
        get_image_features() no longer returns a raw tensor.
        """
        import torch  # noqa: PLC0415
        img = self._open_pil(image_input)
        inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vis_out = self._model.vision_model(**inputs)
            emb = self._model.visual_projection(vis_out.pooler_output)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb  # (1, 512)

    def _encode_texts_raw(self, texts: list[str]) -> "Any":
        """Return L2-normalised text embedding tensor, shape (N, 512).

        Uses text_model + text_projection sub-models directly for
        transformers 5.x compatibility.
        """
        import torch  # noqa: PLC0415
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            text_out = self._model.text_model(**inputs)
            emb = self._model.text_projection(text_out.pooler_output)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb  # (N, 512)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def classify_image(self, image_input: Any) -> VisionResult:
        """Zero-shot classify the image against all civic issue prompts.

        Returns a ``VisionResult`` with the predicted class, confidence
        (softmax probability), and per-class score dict.

        When the engine is not ready, returns a degraded result with
        predicted_class="unknown" and confidence=0.0.
        """
        if not self.is_ready:
            return VisionResult(
                predicted_class="unknown",
                confidence=0.0,
                all_scores={k: 0.0 for k in self._class_names},
                mapped_category=None,
                is_civic=False,
                provider="unavailable",
            )
        try:
            import torch  # noqa: PLC0415

            img_emb  = self._encode_image(image_input)      # (1, 512)
            text_emb = self._prompt_embeddings              # (N, 512) — cached at load

            # Cosine similarities — both already L2-normalised
            logits = (img_emb @ text_emb.T).squeeze(0)     # (N,)
            # Scale by 100 for sharper softmax distribution
            probs = torch.softmax(logits * 100.0, dim=0)

            all_scores: dict[str, float] = {
                name: round(float(p), 4)
                for name, p in zip(self._class_names, probs)
            }

            top_idx = int(probs.argmax())
            top_class = self._class_names[top_idx]
            top_conf  = round(float(probs[top_idx]), 4)

            return VisionResult(
                predicted_class=top_class,
                confidence=top_conf,
                all_scores=all_scores,
                mapped_category=CLIP_CLASS_TO_CATEGORY.get(top_class),
                is_civic=top_class not in _SUSPICIOUS_CLASSES,
                provider="clip_vit_b32",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP classify_image failed: %s", exc)
            return VisionResult(
                predicted_class="unknown",
                confidence=0.0,
                all_scores={k: 0.0 for k in self._class_names},
                mapped_category=None,
                is_civic=False,
                provider="error",
            )

    def text_image_similarity(self, text: str, image_input: Any) -> float:
        """Return cosine similarity between the complaint text and image.

        Uses raw CLIP embedding similarity (not softmax over prompts).
        Higher = more semantically related.  Typical range: -0.3 to +0.4.
        Threshold for "matching": ≥ 0.20.
        """
        if not self.is_ready or not text.strip():
            return 0.0
        try:
            import torch  # noqa: PLC0415
            img_emb  = self._encode_image(image_input)                    # (1, 512)
            text_emb = self._encode_texts_raw([text.strip()[:300]])       # (1, 512)
            sim = float((img_emb @ text_emb.T).squeeze())
            return round(sim, 4)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP text_image_similarity failed: %s", exc)
            return 0.0

    def check_consistency(
        self,
        text: str,
        image_input: Any,
        expected_category: str = "",
    ) -> ConsistencyResult:
        """Assess whether the image is consistent with the complaint text.

        Decision logic
        --------------
        1.  Classify the image → top CLIP class.
        2.  If top class is a fraud indicator (screenshot / indoor / irrelevant)
            → verdict = "contradicts" regardless of text.
        3.  If expected_category given:
                If CLIP class maps to same category family → "supports"
                If CLIP class maps to a different civic category → "contradicts"
        4.  Fall back to raw text-image cosine similarity:
                ≥ 0.22 → "supports"
                ≤ 0.05 → "contradicts"
                else   → "uncertain"
        """
        if not self.is_ready:
            return ConsistencyResult(
                verdict="uncertain",
                score=0.0,
                reason="clip_unavailable",
            )

        try:
            vis = self.classify_image(image_input)
            raw_sim = self.text_image_similarity(text, image_input)

            # Rule 1: fraud / junk image always contradicts
            if vis.predicted_class in _SUSPICIOUS_CLASSES and vis.confidence >= 0.30:
                return ConsistencyResult(
                    verdict="contradicts",
                    score=round(1.0 - vis.confidence, 3),
                    reason=f"image_classified_as_{vis.predicted_class}",
                )

            # Rule 2: category family alignment
            if expected_category and vis.mapped_category is not None:
                if _categories_related(vis.mapped_category, expected_category):
                    return ConsistencyResult(
                        verdict="supports",
                        score=round(max(raw_sim, 0.0) + vis.confidence * 0.3, 3),
                        reason=(
                            f"image_shows_{vis.predicted_class}"
                            f"_matches_{expected_category}"
                        ),
                    )
                # CLIP is confident about a different civic category
                if vis.confidence >= 0.35:
                    return ConsistencyResult(
                        verdict="contradicts",
                        score=round(1.0 - vis.confidence, 3),
                        reason=(
                            f"image_shows_{vis.predicted_class}"
                            f"_not_{expected_category}"
                        ),
                    )

            # Rule 3: raw text-image similarity
            if raw_sim >= 0.22:
                return ConsistencyResult(
                    verdict="supports",
                    score=round(raw_sim, 3),
                    reason="text_image_cosine_similarity_high",
                )
            if raw_sim <= 0.05:
                return ConsistencyResult(
                    verdict="contradicts",
                    score=round(raw_sim, 3),
                    reason="text_image_cosine_similarity_very_low",
                )

            return ConsistencyResult(
                verdict="uncertain",
                score=round(raw_sim, 3),
                reason="insufficient_signal",
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP check_consistency failed: %s", exc)
            return ConsistencyResult(
                verdict="uncertain",
                score=0.0,
                reason=f"error: {exc}",
            )

    def detect_fraud_signals(
        self,
        image_input: Any,
        heuristic_flags: list[str] | None = None,
    ) -> FraudResult:
        """Detect suspicious or non-genuine evidence using CLIP + heuristics.

        Checks
        ------
        * CLIP classifies as screenshot / indoor / irrelevant / poor_quality
        * Heuristic flags already detected (blank, text_heavy, screen_resolution)

        Returns a ``FraudResult`` with a consolidated flag list.
        """
        flags: list[str] = []

        # Pass through existing heuristic flags
        if heuristic_flags:
            flags.extend(heuristic_flags)

        if not self.is_ready:
            is_suspicious = bool(flags)
            return FraudResult(
                is_suspicious=is_suspicious,
                flags=flags,
                reason="clip_unavailable; heuristics_only",
            )

        try:
            vis = self.classify_image(image_input)

            if vis.predicted_class in _SUSPICIOUS_CLASSES:
                flag_name = f"clip_{vis.predicted_class}"
                if flag_name not in flags:
                    flags.append(flag_name)

            # High-confidence secondary suspicious class
            for cls in _SUSPICIOUS_CLASSES:
                if cls != vis.predicted_class:
                    score = vis.all_scores.get(cls, 0.0)
                    if score >= 0.25:
                        flag_name = f"clip_{cls}_secondary"
                        if flag_name not in flags:
                            flags.append(flag_name)

            is_suspicious = bool(flags)
            reason = "; ".join(flags) if flags else ""
            return FraudResult(is_suspicious=is_suspicious, flags=flags, reason=reason)

        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP detect_fraud_signals failed: %s", exc)
            is_suspicious = bool(flags)
            return FraudResult(
                is_suspicious=is_suspicious,
                flags=flags,
                reason=f"clip_error: {exc}",
            )


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_engine: CLIPEngine | None = None
_lock = threading.Lock()


def get_clip_engine() -> CLIPEngine:
    """Return the process-wide CLIP engine singleton.

    Thread-safe double-checked locking.  The first call triggers model load
    (may take 5–30 s depending on download + hardware).  All subsequent calls
    return the cached instance immediately.
    """
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = CLIPEngine()
    return _engine


def _reset_for_testing() -> None:
    """Reset the singleton (test isolation only)."""
    global _engine
    _engine = None

"""Text/NLP classification client hook.

Adapter layer between apps.integrations.services and apps.ml.analyzer.
Calls the pure analyzer and maps its rich output to the exact 8-key contract
that apps.integrations.services.analyze_grievance_submission() expects.

Contract (must not change without updating services.py):
    normalized_summary  str
    category_code       str
    department_code     str
    priority            str   — one of GrievancePriority values
    confidence          float — 0.0–1.0
    language            str
    provider            str
    metadata            dict  — extended with ML + image + decision fields

Metadata keys (Phase A + Phase B + Phase C):
    text_length             int
    ward_hint               str | None
    landmark_hints          list[str]
    spam_check              dict  — {is_spam, spam_score, spam_reason}
    duplicate_check         dict  — {is_duplicate, similarity_score, matching_text}
    needs_human_review      bool
    review_reasons          list[str]
    image_analysis          dict | None  — full Phase B payload or None
    consistency_check       bool | None  — is_consistent from image analysis
    evidence_quality        float | None — quality_score from image analysis
    evidence_review_reason  str         — primary image-related review reason
    decision                dict        — Phase C final automation verdict
                                          {automation_action, routing_confidence,
                                           needs_review, review_reasons,
                                           duplicate_risk, escalation,
                                           decision_metadata}
"""
from __future__ import annotations

from apps.ml.analyzer import analyze_complaint

from .base import confidence as clamp_confidence

ML_PROVIDER = "local_ml_v1"


def classify_grievance_text(
    *,
    raw_text: str,
    language_hint: str | None = None,
    image_input: object = None,
    recent_texts: list[str] | None = None,
) -> dict[str, object]:
    """Return ML-enriched classification hints without mutating domain state.

    Parameters
    ----------
    raw_text
        Raw complaint text in any language / script.
    language_hint
        Optional caller-supplied language override.
    image_input
        Optional image evidence (str path, bytes, or PIL Image).
        When provided, Phase B image intelligence is applied and
        ``metadata`` is enriched with image analysis fields.
    recent_texts
        Optional list of recent normalised summaries from the same ward
        (or a global sample).  Passed directly to ``analyze_complaint()``
        so the Phase-A Jaccard duplicate detector can compare against
        real DB content rather than an empty list.
    """
    text = raw_text.strip()
    result = analyze_complaint(
        text,
        language_hint=language_hint,
        image_input=image_input,
        recent_texts=recent_texts or [],
    )

    img = result.get("image_analysis")  # dict or None

    # Derive the primary evidence review reason (first image-related flag).
    _image_flags = {"image_invalid", "image_poor_quality", "image_irrelevant",
                    "image_contradicts_complaint"}
    evidence_review_reason = next(
        (r for r in result["review_reasons"] if r in _image_flags),  # type: ignore[arg-type]
        "",
    )

    return {
        "normalized_summary": result["normalized_text"],
        "category_code":      result["category_code"],
        "department_code":    result["department_code"],
        "priority":           result["priority"],
        "confidence":         clamp_confidence(float(result["confidence"])),
        "language":           result["language"],
        "provider":           ML_PROVIDER,
        "metadata": {
            "text_length":            len(text),
            "ward_hint":              result["ward_hint"],
            "landmark_hints":         [lm["name"] for lm in result["landmarks"]],  # type: ignore[index]
            "spam_check":             result["spam"],
            "duplicate_check":        result["duplicate"],
            "needs_human_review":     result["needs_human_review"],
            "review_reasons":         result["review_reasons"],
            # Phase B image fields (heuristic)
            "image_analysis":         img,
            "consistency_check":      img["is_consistent"] if img else None,
            "evidence_quality":       img["quality_score"] if img else None,
            "evidence_review_reason": evidence_review_reason,
            # Phase B+ vision AI fields (CLIP — None when CLIP unavailable)
            "vision_class":           img["vision_class"]        if img else None,
            "vision_confidence":      img["vision_confidence"]   if img else None,
            "vision_all_scores":      img["vision_all_scores"]   if img else None,
            "consistency_verdict":    img["consistency_verdict"] if img else None,
            "fraud_flags":            img["fraud_flags"]         if img else [],
            "vision_provider":        img["vision_provider"]     if img else None,
            # Phase C decision intelligence
            "decision":               result["decision"],
        },
    }

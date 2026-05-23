"""apps/ml/decision_engine.py

Final automation decision layer for the TVMC civic grievance system.

Design constraints
------------------
* Pure functions only — no Django imports, no database access, no side effects.
* Accepts only the output dict of analyze_complaint() as input.
* Builds on Phase A (text intelligence) and Phase B (image evidence) results.
* Does NOT re-run any prior analysis — only interprets existing results.

Decision vocabulary
-------------------
auto_route        Sufficient confidence, no hard-blocking flags.
review_required   One or more hard-blocking flags present.
escalate          Life-safety priority or urgent/critical severity.
reject            Confirmed high-confidence spam (spam_score > 0.85).

Confidence thresholds
---------------------
AUTO_ROUTE_MIN  ≥ 0.65  — routing_confidence must clear this for auto-routing
LOW_CONFIDENCE  < 0.40  — triggers a low_confidence review flag

Hard-blocking review flags
--------------------------
These always prevent AUTO_ROUTE regardless of routing_confidence:
    spam_suspicion, no_category_detected, duplicate_risk_high,
    image_contradicts_complaint, image_invalid
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_AUTO_ROUTE_MIN_CONFIDENCE    = 0.65   # routing_confidence needed for AUTO_ROUTE
_REVIEW_CONFIDENCE_THRESHOLD  = 0.40   # below → low_confidence flag

_DUPLICATE_LOW    = 0.40               # similarity < 0.40 → low risk
_DUPLICATE_MEDIUM = 0.55               # 0.40 ≤ similarity < 0.55 → medium risk
                                       # Phase A duplicate threshold is also 0.55

# Life-safety categories that escalate unconditionally at urgent/critical
_LIFE_SAFETY_CATEGORIES: frozenset[str] = frozenset({
    "electrical_hazard",
    "tree_fall",
    "sewage_issue",
})

_ESCALATION_PRIORITIES: frozenset[str] = frozenset({"urgent", "critical"})

# Review reasons that hard-block AUTO_ROUTE even when routing_confidence is high
_HARD_BLOCKING_REVIEW_FLAGS: frozenset[str] = frozenset({
    "spam_suspicion",
    "no_category_detected",
    "duplicate_risk_high",
    "image_contradicts_complaint",
    "image_invalid",
})

# ---------------------------------------------------------------------------
# Public pure functions
# ---------------------------------------------------------------------------


def calculate_routing_confidence(
    *,
    category_confidence: float,
    department_present: bool,
    ward_hint_present: bool,
    image_quality: float | None,
    duplicate_similarity: float,
    spam_score: float,
    language_confidence: float,
) -> dict[str, float]:
    """Compute a single routing confidence score from all available signals.

    Weight table
    ------------
    category match      40 %  — core: can we identify the issue?
    ward / location     20 %  — can we route to a specific ward?
    language clarity    15 %  — is the text interpretable?
    department present  15 %  — do we have a department to route to?
    base floor          10 %  — non-empty submission baseline

    Adjustments
    -----------
    image quality bonus   usable evidence (quality_score ≥ 0.5) adds up to +0.08
    duplicate penalty     confirmed duplicate → ×0.75; medium risk → ×0.90
    spam hard penalty     score × (1 − spam_score × 0.40)

    Returns
    -------
    ``{"routing_confidence": float}``  — clamped to [0.0, 1.0]
    """
    base      = 0.10
    cat_part  = float(category_confidence)  * 0.40
    ward_part = 0.20 if ward_hint_present   else 0.0
    lang_part = float(language_confidence)  * 0.15
    dept_part = 0.15 if department_present  else 0.0

    score = base + cat_part + ward_part + lang_part + dept_part

    # Image quality bonus (only when usable evidence is present)
    if image_quality is not None:
        iq = float(image_quality)
        if iq >= 0.50:
            score += iq * 0.08   # maximum +0.08 for quality_score=1.0

    # Duplicate penalty
    dup = float(duplicate_similarity)
    if dup >= _DUPLICATE_MEDIUM:      # confirmed duplicate
        score *= 0.75
    elif dup >= _DUPLICATE_LOW:       # medium risk
        score *= 0.90

    # Spam penalty (spam_score=1.0 removes 40 % of remaining score)
    score *= max(0.0, 1.0 - float(spam_score) * 0.40)

    return {"routing_confidence": round(max(0.0, min(1.0, score)), 3)}


def detect_duplicate_risk(
    *,
    similarity_score: float,
    is_duplicate: bool,
) -> dict[str, object]:
    """Classify duplicate risk from Phase A Jaccard similarity output.

    Phase A already runs the Jaccard computation and flags confirmed duplicates
    at threshold 0.55.  This function adds a three-level risk classification
    the decision layer uses to pick between AUTO_ROUTE and REVIEW_REQUIRED.

    Risk levels
    -----------
    low    similarity < 0.40   — safe to auto-route
    medium 0.40 ≤ sim < 0.55  — notable overlap; queue for review
    high   sim ≥ 0.55 or is_duplicate=True  — Phase A confirmed; hard-block

    Returns
    -------
    ``{"risk_level": str, "risk_score": float, "is_confirmed": bool}``
    """
    sim = round(float(similarity_score), 3)
    confirmed = bool(is_duplicate)

    if sim >= _DUPLICATE_MEDIUM or confirmed:
        level = "high"
    elif sim >= _DUPLICATE_LOW:
        level = "medium"
    else:
        level = "low"

    return {"risk_level": level, "risk_score": sim, "is_confirmed": confirmed}


def decide_review_requirement(
    *,
    routing_confidence: float,
    spam_score: float,
    duplicate_risk_level: str,
    image_analysis: dict[str, Any] | None,
    category_confidence: float,
    department_code: str,
    ward_hint: str | None,
    language_confidence: float,
    existing_review_reasons: list[str] | None = None,
) -> dict[str, object]:
    """Determine whether the complaint needs human review before routing.

    Conditions checked (each appends a distinct reason string)
    ----------------------------------------------------------
    spam_suspicion           spam_score > 0.40
    no_category_detected     category_confidence == 0.0
    no_department_hint       department_code is empty
    no_ward_hint             ward_hint is None
    low_confidence           routing_confidence < 0.40
    duplicate_risk_high      duplicate_risk_level == "high"
    duplicate_risk_medium    duplicate_risk_level == "medium"
    language_uncertain       language_confidence < 0.25
    image_invalid            image present but unreadable
    image_poor_quality       image present but not usable as evidence
    image_irrelevant         image detected as junk/screenshot/blank
    image_contradicts_complaint  image inconsistent with complaint

    Phase A / B reasons passed in via ``existing_review_reasons`` are merged
    without duplication so the final list is deduplicated.

    Returns
    -------
    ``{"needs_review": bool, "review_reasons": list[str]}``
    """
    reasons: list[str] = []

    if float(spam_score) > 0.40:
        reasons.append("spam_suspicion")

    if float(category_confidence) == 0.0:
        reasons.append("no_category_detected")

    if not department_code:
        reasons.append("no_department_hint")

    if ward_hint is None:
        reasons.append("no_ward_hint")

    if float(routing_confidence) < _REVIEW_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")

    if duplicate_risk_level == "high":
        reasons.append("duplicate_risk_high")
    elif duplicate_risk_level == "medium":
        reasons.append("duplicate_risk_medium")

    if float(language_confidence) < 0.25:
        reasons.append("language_uncertain")

    # Image-specific triggers (Phase B evidence)
    if image_analysis is not None:
        if not image_analysis.get("is_valid", True):
            reasons.append("image_invalid")
        elif not image_analysis.get("usable", True):
            reasons.append("image_poor_quality")
        elif image_analysis.get("is_irrelevant", False):
            reasons.append("image_irrelevant")
        if not image_analysis.get("is_consistent", True):
            if "image_contradicts_complaint" not in reasons:
                reasons.append("image_contradicts_complaint")

    # Merge Phase A / B reasons (deduplicate, preserve order)
    if existing_review_reasons:
        seen = set(reasons)
        for r in existing_review_reasons:
            if r not in seen:
                reasons.append(r)
                seen.add(r)

    return {"needs_review": bool(reasons), "review_reasons": reasons}


def decide_escalation(
    *,
    priority: str,
    category_code: str,
    routing_confidence: float,
    needs_review: bool,
    review_reasons: list[str],
) -> dict[str, object]:
    """Determine whether the complaint warrants immediate escalation.

    Escalation rules (evaluated in priority order)
    -----------------------------------------------
    Rule 1  Life-safety category (electrical_hazard / tree_fall / sewage_issue)
            AND priority is urgent or critical → always escalate.
    Rule 2  Any complaint with priority urgent or critical → escalate.
    Rule 3  image_contradicts_complaint present AND priority is high or above
            → evidence contradiction on a serious complaint → escalate.

    Returns
    -------
    ``{"should_escalate": bool, "escalation_reason": str}``
    """
    prio = str(priority).lower()
    cat  = str(category_code).lower()

    if cat in _LIFE_SAFETY_CATEGORIES and prio in _ESCALATION_PRIORITIES:
        return {
            "should_escalate":   True,
            "escalation_reason": f"life_safety_category_urgent: {cat}",
        }

    if prio in _ESCALATION_PRIORITIES:
        return {
            "should_escalate":   True,
            "escalation_reason": f"urgent_priority: {prio}",
        }

    if "image_contradicts_complaint" in review_reasons and prio in ("high", "urgent", "critical"):
        return {
            "should_escalate":   True,
            "escalation_reason": "evidence_contradiction_on_serious_complaint",
        }

    return {"should_escalate": False, "escalation_reason": ""}


def make_final_decision(analyzer_output: dict[str, Any]) -> dict[str, object]:
    """Produce the final automation verdict from the complete analyzer output.

    This is the main entry point for Phase C.  It accepts the dict returned
    by ``analyze_complaint()`` and returns a structured decision dict.

    Decision actions
    ----------------
    reject            spam_score > 0.85 (high-confidence spam)
    escalate          urgent / critical priority or life-safety category
    review_required   one or more hard-blocking review flags, or
                      routing_confidence < AUTO_ROUTE_MIN
    auto_route        routing_confidence ≥ 0.65 and no hard-blocking flags

    Hard-blocking flags (prevent auto_route even at high confidence):
        spam_suspicion, no_category_detected, duplicate_risk_high,
        image_contradicts_complaint, image_invalid

    Returns
    -------
    dict with keys:
        automation_action   str    — "auto_route" | "review_required"
                                     | "escalate" | "reject"
        routing_confidence  float
        needs_review        bool
        review_reasons      list[str]
        duplicate_risk      dict   — {risk_level, risk_score, is_confirmed}
        escalation          dict   — {should_escalate, escalation_reason}
        decision_metadata   dict   — raw signal values used for the decision
    """
    # ── Extract Phase A / B fields ────────────────────────────────────────
    spam_result    = analyzer_output.get("spam", {})
    dup_result     = analyzer_output.get("duplicate", {})
    image_analysis = analyzer_output.get("image_analysis")

    spam_score          = float(spam_result.get("spam_score", 0.0))
    dup_sim             = float(dup_result.get("similarity_score", 0.0))
    is_duplicate        = bool(dup_result.get("is_duplicate", False))
    category_confidence = float(analyzer_output.get("category_confidence", 0.0))
    language_confidence = float(analyzer_output.get("language_confidence", 0.0))
    department_code     = str(analyzer_output.get("department_code", ""))
    ward_hint           = analyzer_output.get("ward_hint")
    priority            = str(analyzer_output.get("priority", "medium"))
    category_code       = str(analyzer_output.get("category_code", ""))

    # Phase B: image quality scalar (None when no image was provided)
    image_quality: float | None = None
    if image_analysis is not None:
        image_quality = float(image_analysis.get("quality_score", 0.0))

    # ── Step 1: routing confidence ────────────────────────────────────────
    rc = calculate_routing_confidence(
        category_confidence=category_confidence,
        department_present=bool(department_code),
        ward_hint_present=ward_hint is not None,
        image_quality=image_quality,
        duplicate_similarity=dup_sim,
        spam_score=spam_score,
        language_confidence=language_confidence,
    )
    routing_confidence: float = rc["routing_confidence"]

    # ── Step 2: duplicate risk classification ────────────────────────────
    dup_risk = detect_duplicate_risk(
        similarity_score=dup_sim,
        is_duplicate=is_duplicate,
    )

    # ── Step 3: review requirement ────────────────────────────────────────
    review_result = decide_review_requirement(
        routing_confidence=routing_confidence,
        spam_score=spam_score,
        duplicate_risk_level=str(dup_risk["risk_level"]),
        image_analysis=image_analysis,
        category_confidence=category_confidence,
        department_code=department_code,
        ward_hint=ward_hint,
        language_confidence=language_confidence,
        existing_review_reasons=list(analyzer_output.get("review_reasons", [])),
    )
    needs_review:   bool      = bool(review_result["needs_review"])
    review_reasons: list[str] = list(review_result["review_reasons"])

    # ── Step 4: escalation check ──────────────────────────────────────────
    escalation = decide_escalation(
        priority=priority,
        category_code=category_code,
        routing_confidence=routing_confidence,
        needs_review=needs_review,
        review_reasons=review_reasons,
    )

    # ── Step 5: final action ──────────────────────────────────────────────
    # Order matters: reject > escalate > review_required > auto_route
    hard_blocked = bool(set(review_reasons) & _HARD_BLOCKING_REVIEW_FLAGS)

    if spam_score > 0.85:
        action = "reject"
    elif escalation["should_escalate"]:
        action = "escalate"
    elif hard_blocked or routing_confidence < _AUTO_ROUTE_MIN_CONFIDENCE:
        action = "review_required"
    else:
        action = "auto_route"

    return {
        "automation_action":  action,
        "routing_confidence": routing_confidence,
        "needs_review":       needs_review,
        "review_reasons":     review_reasons,
        "duplicate_risk":     dup_risk,
        "escalation":         escalation,
        "decision_metadata": {
            "spam_score":           spam_score,
            "duplicate_similarity": dup_sim,
            "category_confidence":  category_confidence,
            "language_confidence":  language_confidence,
            "department_present":   bool(department_code),
            "ward_hint_present":    ward_hint is not None,
            "image_quality":        image_quality,
            "priority":             priority,
        },
    }

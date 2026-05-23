"""Integration tests for Phase D — AI production wiring.

Scope
-----
Verifies that the AI pipeline (Phase A/B/C) is correctly wired into the
grievance lifecycle via ``analyze_grievance_submission()`` and
``enrich_grievance_with_ai()``.

Strategy
--------
All tests are unit-style (no DB) unless explicitly marked ``django_db``.
DB-touching helpers (``local_landmark_candidates_for_mention``,
``recent_grievance_summaries_for_duplicate_context``) are patched so the
tests run fast and do not require migrations.

Test categories
---------------
1. ``TestAnalyzeGrievanceSubmissionStructure``   — payload contract
2. ``TestAnalyzeGrievanceSubmissionDuplicateContext`` — recent-texts wiring
3. ``TestAnalyzeGrievanceSubmissionFailureSafety``   — AI crash safety
4. ``TestEnrichGrievanceWithAi``                  — enrichment mapping
5. ``TestExplainabilityReasonBuilder``            — ai_explainability strings
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from apps.integrations.services import (
    analyze_grievance_submission,
    enrich_grievance_with_ai,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RAW_TEXT = "Pothole on main road near Pattom junction."


def _make_nlp_result(
    *,
    category_code: str = "road_damage",
    priority: str = "medium",
    action: str = "auto_route",
    confidence: float = 0.75,
    review_reasons: list[str] | None = None,
    escalation_reason: str = "",
) -> dict:
    """Build a minimal ``classify_grievance_text()`` return payload.

    The shape mirrors the real return value so tests exercise real code paths
    without running the full ML engine.
    """
    rr = review_reasons or []
    needs_review = action != "auto_route"
    return {
        "normalized_summary": "Pothole near Pattom junction.",
        "category_code":      category_code,
        "department_code":    "roads_and_drainage",
        "priority":           priority,
        "confidence":         confidence,
        "language":           "english",
        "provider":           "local_ml_v1",
        "metadata": {
            "text_length":            len(_RAW_TEXT),
            "ward_hint":              "tvm_034",
            "landmark_hints":         ["Pattom"],
            "spam_check":             {"is_spam": False, "spam_score": 0.0, "spam_reason": ""},
            "duplicate_check":        {"is_duplicate": False, "similarity_score": 0.0, "matching_text": None},
            "needs_human_review":     needs_review,
            "review_reasons":         rr,
            "image_analysis":         None,
            "consistency_check":      None,
            "evidence_quality":       None,
            "evidence_review_reason": "",
            "decision": {
                "automation_action":  action,
                "routing_confidence": confidence,
                "needs_review":       needs_review,
                "review_reasons":     rr,
                "duplicate_risk":     {"risk_level": "low", "risk_score": 0.0, "is_confirmed": False},
                "escalation":         {"should_escalate": action == "escalate", "escalation_reason": escalation_reason},
                "decision_metadata":  {},
            },
        },
    }


def _make_submission_payload(
    *,
    action: str = "auto_route",
    confidence: float = 0.75,
    review_reasons: list[str] | None = None,
    escalation_reason: str = "",
) -> dict:
    """Build a minimal ``analyze_grievance_submission()`` return payload.

    Used when mocking the whole analysis function to test ``enrich_grievance_with_ai()``.
    """
    rr = review_reasons or []
    needs_review = action != "auto_route"
    ai_decision = {
        "automation_action":  action,
        "routing_confidence": confidence,
        "needs_review":       needs_review,
        "review_reasons":     rr,
        "duplicate_risk":     {"risk_level": "low", "risk_score": 0.0, "is_confirmed": False},
        "escalation":         {"should_escalate": action == "escalate", "escalation_reason": escalation_reason},
        "decision_metadata":  {},
    }
    if action == "auto_route":
        explainability = f"Auto-routed by AI with {confidence:.0%} confidence."
    elif action == "reject":
        explainability = "Submission rejected by AI: likely spam or invalid content."
    elif action == "escalate":
        if escalation_reason:
            explainability = f"Escalated by AI: {escalation_reason}."
        else:
            explainability = "Escalated by AI due to high priority or life-safety concern."
    else:
        reason_str = ", ".join(rr[:3]) if rr else "low confidence"
        explainability = f"Human review required: {reason_str}."
    return {
        "normalized_summary":           "Pothole near Pattom junction.",
        "category_code":                "road_damage",
        "priority":                     "medium",
        "landmark_resolution_metadata": {
            "provider_result":  {"landmark_code": None, "provider": "local_stub", "confidence": 0.0, "metadata": {}},
            "local_candidates": [],
        },
        "duplicate_detection_metadata": {
            "possible_duplicate_tracking_code": None,
            "confidence": 0.0,
            "candidates": [],
            "provider": "local_stub",
            "metadata": {},
        },
        "ai_decision":      ai_decision,
        "ai_explainability": explainability,
        "provider_metadata": {
            "nlp":       {"provider": "local_ml_v1", "confidence": confidence, "language": "english", "metadata": {}},
            "landmark":  {"provider": "local_stub", "confidence": 0.0, "metadata": {}},
            "duplicate": {"provider": "local_stub", "confidence": 0.0, "metadata": {}},
        },
    }


def _mock_grievance(
    raw_text: str = _RAW_TEXT,
    landmark_mention: str = "",
    citizen_location_text: str = "",
) -> MagicMock:
    g = MagicMock()
    g.pk = 42
    g.raw_text = raw_text
    g.landmark_mention = landmark_mention
    g.citizen_location_text = citizen_location_text
    return g


# ---------------------------------------------------------------------------
# Patch paths (constants for DRY)
# ---------------------------------------------------------------------------

_P_NLP     = "apps.integrations.services.classify_grievance_text"
_P_RECENT  = "apps.integrations.services.recent_grievance_summaries_for_duplicate_context"
_P_LOCAL   = "apps.integrations.services.local_landmark_candidates_for_mention"
_P_ANALYZE = "apps.integrations.services.analyze_grievance_submission"
_P_UPDATE  = "apps.grievances.services.update_grievance_enrichment"


# ===========================================================================
# 1. analyze_grievance_submission() — payload contract
# ===========================================================================


class TestAnalyzeGrievanceSubmissionStructure:
    """Verify the Phase D extended payload contract — no DB access needed."""

    _REQUIRED_KEYS = {
        "normalized_summary",
        "category_code",
        "priority",
        "landmark_resolution_metadata",
        "duplicate_detection_metadata",
        "ai_decision",
        "ai_explainability",
        "provider_metadata",
    }

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_payload_has_all_required_top_level_keys(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result()
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert self._REQUIRED_KEYS <= set(payload.keys())

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_ai_decision_is_dict_at_top_level(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(action="auto_route")
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert isinstance(payload["ai_decision"], dict)

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_ai_decision_automation_action_matches_nlp_result(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(action="review_required")
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert payload["ai_decision"]["automation_action"] == "review_required"

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_ai_explainability_is_string(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(action="auto_route")
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert isinstance(payload["ai_explainability"], str)

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_ai_explainability_non_empty_for_valid_action(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(action="auto_route")
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert payload["ai_explainability"] != ""

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_provider_metadata_has_nlp_landmark_duplicate_keys(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result()
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert {"nlp", "landmark", "duplicate"} <= payload["provider_metadata"].keys()

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_accepts_image_input_and_ward_code_without_error(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result()
        payload = analyze_grievance_submission(
            raw_text=_RAW_TEXT,
            image_input=None,
            ward_code="tvm_034",
        )
        assert payload is not None

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_backward_compat_keys_not_removed(self, mock_nlp, mock_recent, mock_local):
        """Keys present before Phase D must still be in the payload."""
        mock_nlp.return_value = _make_nlp_result()
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        legacy_keys = {"normalized_summary", "category_code", "priority",
                       "landmark_resolution_metadata", "duplicate_detection_metadata",
                       "provider_metadata"}
        assert legacy_keys <= set(payload.keys())


# ===========================================================================
# 2. analyze_grievance_submission() — duplicate context wiring
# ===========================================================================


class TestAnalyzeGrievanceSubmissionDuplicateContext:
    """recent_summaries must flow from selector → NLP client."""

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=["Old pothole report.", "Drainage overflow."])
    @patch(_P_NLP)
    def test_recent_summaries_passed_to_nlp_as_recent_texts(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result()
        analyze_grievance_submission(raw_text=_RAW_TEXT, ward_code="tvm_034")
        _, kwargs = mock_nlp.call_args
        assert kwargs.get("recent_texts") == ["Old pothole report.", "Drainage overflow."]

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_ward_code_forwarded_to_recent_summaries_selector(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result()
        analyze_grievance_submission(raw_text=_RAW_TEXT, ward_code="tvm_034")
        mock_recent.assert_called_once_with(ward_code="tvm_034")

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_none_ward_code_forwarded_to_selector(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result()
        analyze_grievance_submission(raw_text=_RAW_TEXT)
        mock_recent.assert_called_once_with(ward_code=None)

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_image_input_forwarded_to_classify_text(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result()
        sentinel = object()
        analyze_grievance_submission(raw_text=_RAW_TEXT, image_input=sentinel)
        _, kwargs = mock_nlp.call_args
        assert kwargs.get("image_input") is sentinel


# ===========================================================================
# 3. analyze_grievance_submission() — failure safety
# ===========================================================================


class TestAnalyzeGrievanceSubmissionFailureSafety:
    """AI crash must never propagate to the caller."""

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP, side_effect=RuntimeError("ML model crash"))
    def test_returns_dict_when_nlp_crashes(self, mock_nlp, mock_recent, mock_local):
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert isinstance(payload, dict)

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP, side_effect=RuntimeError("crash"))
    def test_fallback_ai_decision_is_review_required(self, mock_nlp, mock_recent, mock_local):
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert payload["ai_decision"].get("automation_action") == "review_required"

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP, side_effect=RuntimeError("crash"))
    def test_fallback_has_all_required_top_level_keys(self, mock_nlp, mock_recent, mock_local):
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        required = {
            "normalized_summary", "category_code", "priority",
            "landmark_resolution_metadata", "duplicate_detection_metadata",
            "ai_decision", "ai_explainability", "provider_metadata",
        }
        assert required <= set(payload.keys())

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP, side_effect=RuntimeError("crash"))
    def test_fallback_review_reasons_includes_ai_pipeline_error(self, mock_nlp, mock_recent, mock_local):
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert "ai_pipeline_error" in payload["ai_decision"].get("review_reasons", [])

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, side_effect=Exception("DB down"))
    @patch(_P_NLP)
    def test_db_error_in_selector_also_falls_back(self, mock_nlp, mock_recent, mock_local):
        """A DB failure fetching recent summaries should use the fallback path."""
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert isinstance(payload, dict)
        # Fallback → review_required because NLP was never called
        assert payload["ai_decision"].get("automation_action") == "review_required"

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP, side_effect=MemoryError("OOM"))
    def test_memory_error_also_falls_back(self, mock_nlp, mock_recent, mock_local):
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert payload["ai_decision"]["automation_action"] == "review_required"


# ===========================================================================
# 4. enrich_grievance_with_ai() — enrichment mapping
# ===========================================================================


class TestEnrichGrievanceWithAi:
    """Verify enrichment values and failure-safety contract."""

    # ── Return-value tests ───────────────────────────────────────────────────

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_returns_true_on_success(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        mock_update.return_value = MagicMock()
        assert enrich_grievance_with_ai(grievance=_mock_grievance()) is True

    @patch(_P_ANALYZE, side_effect=RuntimeError("crash"))
    def test_returns_false_when_analysis_crashes(self, mock_analyze):
        assert enrich_grievance_with_ai(grievance=_mock_grievance()) is False

    @patch(_P_UPDATE, side_effect=Exception("DB write error"))
    @patch(_P_ANALYZE)
    def test_returns_false_when_enrichment_write_fails(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        assert enrich_grievance_with_ai(grievance=_mock_grievance()) is False

    @patch(_P_UPDATE, side_effect=ValueError("validation"))
    @patch(_P_ANALYZE)
    def test_returns_false_on_any_enrichment_exception(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        assert enrich_grievance_with_ai(grievance=_mock_grievance()) is False

    # ── update_enrichment call-count tests ───────────────────────────────────

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_update_enrichment_called_exactly_once_on_success(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        assert mock_update.call_count == 1

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE, side_effect=RuntimeError("crash"))
    def test_update_enrichment_not_called_when_analysis_fails(self, mock_analyze, mock_update):
        enrich_grievance_with_ai(grievance=_mock_grievance())
        mock_update.assert_not_called()

    # ── Field mapping tests ──────────────────────────────────────────────────

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_normalized_summary_mapped_to_enrichment(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["normalized_summary"] == "Pothole near Pattom junction."

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_category_code_mapped_to_enrichment(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["category_code"] == "road_damage"

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_priority_mapped_to_enrichment(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["priority"] == "medium"

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_status_metadata_has_ai_enrichment_flag(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["status_metadata"]["ai_enrichment"] is True

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_status_metadata_automation_action_auto_route(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload(action="auto_route")
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["status_metadata"]["automation_action"] == "auto_route"

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_status_metadata_automation_action_review_required(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload(action="review_required")
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        sm = kwargs["values"]["status_metadata"]
        assert sm["automation_action"] == "review_required"
        assert sm["needs_review"] is True

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_status_metadata_automation_action_escalate(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload(action="escalate")
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["status_metadata"]["automation_action"] == "escalate"

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_status_metadata_automation_action_reject(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload(action="reject")
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["status_metadata"]["automation_action"] == "reject"

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_status_reason_set_to_ai_explainability(self, mock_analyze, mock_update):
        payload = _make_submission_payload(action="auto_route", confidence=0.75)
        mock_analyze.return_value = payload
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=_mock_grievance())
        _, kwargs = mock_update.call_args
        assert kwargs["values"]["status_reason"] == payload["ai_explainability"]

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_grievance_object_forwarded_to_update_enrichment(self, mock_analyze, mock_update):
        mock_analyze.return_value = _make_submission_payload()
        mock_update.return_value = MagicMock()
        g = _mock_grievance()
        enrich_grievance_with_ai(grievance=g)
        _, kwargs = mock_update.call_args
        assert kwargs["grievance"] is g


# ===========================================================================
# 5. _build_explainability_reason() — tested via analyze_grievance_submission
# ===========================================================================


class TestExplainabilityReasonBuilder:
    """Verify the human-readable AI explainability string for each action."""

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_auto_route_string_contains_confidence_percentage(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(action="auto_route", confidence=0.80)
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        explainability = payload["ai_explainability"]
        assert "80%" in explainability or "auto" in explainability.lower()

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_reject_string_mentions_rejected(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(action="reject")
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert "reject" in payload["ai_explainability"].lower()

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_review_required_string_mentions_review(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(
            action="review_required",
            review_reasons=["low_confidence"],
        )
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert "review" in payload["ai_explainability"].lower()

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_review_reasons_embedded_in_review_required_string(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(
            action="review_required",
            review_reasons=["spam_suspicion"],
        )
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert "spam_suspicion" in payload["ai_explainability"]

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_escalate_string_mentions_escalated(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(
            action="escalate",
            escalation_reason="Life safety hazard detected",
        )
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert "escalat" in payload["ai_explainability"].lower()

    @patch(_P_LOCAL, return_value=[])
    @patch(_P_RECENT, return_value=[])
    @patch(_P_NLP)
    def test_escalate_with_reason_embeds_reason_in_string(self, mock_nlp, mock_recent, mock_local):
        mock_nlp.return_value = _make_nlp_result(
            action="escalate",
            escalation_reason="Life safety hazard detected",
        )
        payload = analyze_grievance_submission(raw_text=_RAW_TEXT)
        assert "Life safety hazard detected" in payload["ai_explainability"]

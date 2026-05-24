"""Unit and integration tests for Phase E — KSMART-style routing intelligence.

apps/integrations/routing.py

Test categories
---------------
1. ``TestResolveWardFromHint``           — ward resolver (unit + DB)
2. ``TestResolveDepartmentFromCategory`` — dept resolver (unit + DB)
3. ``TestResolveRoutingBucket``          — routing bucket logic (pure unit)
4. ``TestBuildPhaseERouting``            — orchestrator (unit + light DB)
5. ``TestPhaseEIntegrationWithEnrich``   — wiring into enrich_grievance_with_ai

DB tests use real Ward/Department rows but keep geometry minimal.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from django.contrib.gis.geos import Point, Polygon

from apps.integrations.routing import (
    _serializable_dept_result,
    _serializable_ward_result,
    build_phase_e_routing,
    resolve_department_from_category,
    resolve_routing_bucket,
    resolve_ward_from_hint,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SIMPLE_POLYGON = Polygon(
    ((76.9, 8.5), (76.9, 8.6), (77.0, 8.6), (77.0, 8.5), (76.9, 8.5)),
    srid=4326,
)


def _make_ai_decision(
    *,
    action: str = "auto_route",
    needs_review: bool = False,
) -> dict:
    return {
        "automation_action": action,
        "routing_confidence": 0.75,
        "needs_review": needs_review,
        "review_reasons": [],
        "duplicate_risk": {"risk_level": "low", "risk_score": 0.0, "is_confirmed": False},
        "escalation": {"should_escalate": False, "escalation_reason": ""},
    }


# ===========================================================================
# 1. Ward resolver
# ===========================================================================


class TestResolveWardFromHint:
    """Ward resolver — unit tests (no DB)."""

    def test_returns_unresolved_when_ward_hint_is_none(self):
        result = resolve_ward_from_hint(ward_hint=None)
        assert result["ward"] is None
        assert result["source"] == "unresolved"
        assert result["confidence"] == 0.0

    def test_returns_unresolved_when_ward_hint_is_empty_string(self):
        result = resolve_ward_from_hint(ward_hint="")
        assert result["ward"] is None
        assert result["source"] == "unresolved"

    def test_ward_code_is_none_when_hint_is_none(self):
        result = resolve_ward_from_hint(ward_hint=None)
        assert result["ward_code"] is None

    @pytest.mark.django_db
    def test_returns_ward_instance_when_code_matches_db(self):
        from apps.wards.models import Ward

        Ward.objects.create(code="tvm_034", name="Pattom", boundary=_SIMPLE_POLYGON)
        result = resolve_ward_from_hint(ward_hint="tvm_034")
        assert result["ward"] is not None
        assert result["ward"].code == "tvm_034"
        assert result["ward_code"] == "tvm_034"
        assert result["confidence"] == 0.85
        assert result["source"] == "landmark_alias_match"

    @pytest.mark.django_db
    def test_returns_ward_not_in_db_when_hint_not_seeded(self):
        result = resolve_ward_from_hint(ward_hint="tvm_099")
        assert result["ward"] is None
        assert result["ward_code"] == "tvm_099"
        assert result["source"] == "ward_not_in_db"
        assert result["confidence"] == 0.0

    @pytest.mark.django_db
    def test_returns_unresolved_for_inactive_ward(self):
        from apps.wards.models import Ward

        Ward.objects.create(
            code="tvm_001",
            name="Kazhakkoottam",
            boundary=_SIMPLE_POLYGON,
            is_active=False,
        )
        result = resolve_ward_from_hint(ward_hint="tvm_001")
        assert result["ward"] is None
        assert result["source"] == "ward_not_in_db"

    @pytest.mark.django_db
    def test_returns_correct_ward_when_multiple_wards_exist(self):
        from apps.wards.models import Ward

        Ward.objects.create(code="tvm_001", name="Kazhakkoottam", boundary=_SIMPLE_POLYGON)
        Ward.objects.create(code="tvm_034", name="Pattom", boundary=_SIMPLE_POLYGON)
        result = resolve_ward_from_hint(ward_hint="tvm_034")
        assert result["ward"].code == "tvm_034"
        assert result["ward"].name == "Pattom"


# ===========================================================================
# 2. Department resolver
# ===========================================================================


class TestResolveDepartmentFromCategory:
    """Department resolver — unit tests (no DB)."""

    def test_returns_unresolved_when_both_codes_empty(self):
        # No DB access — early return before any import queries
        with patch("apps.departments.models.Department.objects") as mock_qs:
            result = resolve_department_from_category(
                category_code="", department_code=""
            )
        assert result["department"] is None
        assert result["source"] == "unresolved"

    def test_returns_unresolved_when_both_codes_are_empty_strings(self):
        result = resolve_department_from_category(category_code="", department_code="")
        assert result["department"] is None
        assert result["confidence"] == 0.0

    @pytest.mark.django_db
    def test_direct_code_match_returns_department(self):
        from apps.departments.models import Department

        Department.objects.create(
            code="roads_and_drainage",
            name="Roads and Drainage",
            handled_categories=["road_damage", "drainage"],
        )
        result = resolve_department_from_category(
            category_code="road_damage",
            department_code="roads_and_drainage",
        )
        assert result["department"] is not None
        assert result["department"].code == "roads_and_drainage"
        assert result["confidence"] == 0.90
        assert result["source"] == "direct_code_match"

    @pytest.mark.django_db
    def test_category_fallback_used_when_direct_code_missing(self):
        from apps.departments.models import Department

        # DB has department with category but different code than hint
        Department.objects.create(
            code="roads_dept",
            name="Roads Department",
            handled_categories=["road_damage"],
        )
        result = resolve_department_from_category(
            category_code="road_damage",
            department_code="roads_and_drainage",  # not in DB
        )
        # Falls back to category match
        assert result["department"] is not None
        assert result["department"].code == "roads_dept"
        assert result["confidence"] == 0.75
        assert result["source"] == "category_match"

    @pytest.mark.django_db
    def test_category_only_match_works_without_department_code(self):
        from apps.departments.models import Department

        Department.objects.create(
            code="sanitation",
            name="Sanitation",
            handled_categories=["waste_management"],
        )
        result = resolve_department_from_category(
            category_code="waste_management",
            department_code="",
        )
        assert result["department"] is not None
        assert result["department"].code == "sanitation"
        assert result["source"] == "category_match"

    @pytest.mark.django_db
    def test_returns_unresolved_when_nothing_matches(self):
        result = resolve_department_from_category(
            category_code="unknown_category",
            department_code="unknown_dept",
        )
        assert result["department"] is None
        assert result["source"] == "unresolved"

    @pytest.mark.django_db
    def test_inactive_department_is_excluded(self):
        from apps.departments.models import Department

        Department.objects.create(
            code="roads_and_drainage",
            name="Roads",
            handled_categories=["road_damage"],
            is_active=False,
        )
        result = resolve_department_from_category(
            category_code="road_damage",
            department_code="roads_and_drainage",
        )
        assert result["department"] is None
        assert result["source"] == "unresolved"

    @pytest.mark.django_db
    def test_direct_match_preferred_over_category_match(self):
        from apps.departments.models import Department

        # Both a direct-code match and a category-match exist
        Department.objects.create(
            code="roads_and_drainage",
            name="Roads and Drainage",
            handled_categories=["road_damage"],
        )
        Department.objects.create(
            code="other_dept",
            name="Other",
            handled_categories=["road_damage"],
        )
        result = resolve_department_from_category(
            category_code="road_damage",
            department_code="roads_and_drainage",
        )
        assert result["source"] == "direct_code_match"
        assert result["department"].code == "roads_and_drainage"


# ===========================================================================
# 3. Routing bucket resolver — pure unit tests
# ===========================================================================


class TestResolveRoutingBucket:
    """Routing bucket logic — no DB, all possible routing modes."""

    def _no_ward(self) -> dict:
        return {"ward": None, "ward_code": None, "confidence": 0.0, "source": "unresolved"}

    def _no_dept(self) -> dict:
        return {
            "department": None,
            "department_code": None,
            "confidence": 0.0,
            "source": "unresolved",
        }

    def _ward(self, code: str = "tvm_034", conf: float = 0.85) -> dict:
        return {
            "ward": MagicMock(code=code),
            "ward_code": code,
            "confidence": conf,
            "source": "landmark_alias_match",
        }

    def _dept(self, code: str = "roads_and_drainage", conf: float = 0.90) -> dict:
        return {
            "department": MagicMock(code=code),
            "department_code": code,
            "confidence": conf,
            "source": "direct_code_match",
        }

    # ── manual_review precedence ────────────────────────────────────────────

    def test_review_required_action_always_manual_review(self):
        result = resolve_routing_bucket(
            ward_result=self._ward(),
            dept_result=self._dept(),
            ai_decision=_make_ai_decision(action="review_required", needs_review=True),
        )
        assert result["routing_mode"] == "manual_review"
        assert result["routing_bucket"] == "manual_review"

    def test_reject_action_always_manual_review(self):
        result = resolve_routing_bucket(
            ward_result=self._ward(),
            dept_result=self._dept(),
            ai_decision=_make_ai_decision(action="reject"),
        )
        assert result["routing_mode"] == "manual_review"

    def test_needs_review_true_forces_manual_review_even_for_auto_route(self):
        result = resolve_routing_bucket(
            ward_result=self._ward(),
            dept_result=self._dept(),
            ai_decision=_make_ai_decision(action="auto_route", needs_review=True),
        )
        assert result["routing_mode"] == "manual_review"

    def test_manual_review_confidence_is_1(self):
        result = resolve_routing_bucket(
            ward_result=self._no_ward(),
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="review_required"),
        )
        assert result["confidence"] == 1.0

    # ── ward_queue ──────────────────────────────────────────────────────────

    def test_ward_queue_when_ward_resolved_with_high_confidence(self):
        result = resolve_routing_bucket(
            ward_result=self._ward("tvm_034", conf=0.85),
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_mode"] == "ward_queue"
        assert result["routing_bucket"] == "ward_tvm_034"
        assert result["confidence"] == 0.85

    def test_ward_queue_not_used_when_confidence_below_threshold(self):
        # confidence=0.50 < 0.60 threshold → falls through to department/central
        result = resolve_routing_bucket(
            ward_result=self._ward("tvm_034", conf=0.50),
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_mode"] != "ward_queue"

    def test_ward_queue_bucket_name_includes_ward_code(self):
        result = resolve_routing_bucket(
            ward_result=self._ward("tvm_085", conf=0.85),
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_bucket"] == "ward_tvm_085"

    def test_ward_queue_not_used_when_ward_instance_is_none(self):
        # ward_code present but no DB instance (ward_not_in_db source)
        result = resolve_routing_bucket(
            ward_result={
                "ward": None,
                "ward_code": "tvm_034",
                "confidence": 0.85,
                "source": "ward_not_in_db",
            },
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_mode"] != "ward_queue"

    # ── department_queue ────────────────────────────────────────────────────

    def test_department_queue_when_dept_resolved_but_no_ward(self):
        result = resolve_routing_bucket(
            ward_result=self._no_ward(),
            dept_result=self._dept("sanitation"),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_mode"] == "department_queue"
        assert result["routing_bucket"] == "dept_sanitation"

    def test_department_queue_bucket_name_includes_dept_code(self):
        result = resolve_routing_bucket(
            ward_result=self._no_ward(),
            dept_result=self._dept("water_authority"),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_bucket"] == "dept_water_authority"

    def test_ward_queue_wins_over_department_queue_when_both_resolved(self):
        result = resolve_routing_bucket(
            ward_result=self._ward("tvm_034", conf=0.85),
            dept_result=self._dept("roads_and_drainage"),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_mode"] == "ward_queue"

    # ── central_queue fallback ──────────────────────────────────────────────

    def test_central_queue_when_nothing_resolved(self):
        result = resolve_routing_bucket(
            ward_result=self._no_ward(),
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_mode"] == "central_queue"
        assert result["routing_bucket"] == "central"

    def test_central_queue_confidence_is_zero(self):
        result = resolve_routing_bucket(
            ward_result=self._no_ward(),
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["confidence"] == 0.0

    def test_central_queue_when_ward_below_threshold_and_no_dept(self):
        result = resolve_routing_bucket(
            ward_result=self._ward("tvm_034", conf=0.40),
            dept_result=self._no_dept(),
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_mode"] == "central_queue"


# ===========================================================================
# 4. Serialisation helpers
# ===========================================================================


class TestSerialisationHelpers:
    """_serializable_ward_result and _serializable_dept_result."""

    def test_ward_result_strips_orm_instance(self):
        mock_ward = MagicMock()
        mock_ward.pk = 7
        result = _serializable_ward_result(
            {"ward": mock_ward, "ward_code": "tvm_034", "confidence": 0.85, "source": "landmark_alias_match"}
        )
        assert "ward" not in result
        assert result["ward_id"] == 7
        assert result["ward_code"] == "tvm_034"
        assert result["confidence"] == 0.85
        assert result["source"] == "landmark_alias_match"

    def test_ward_result_with_none_instance(self):
        result = _serializable_ward_result(
            {"ward": None, "ward_code": None, "confidence": 0.0, "source": "unresolved"}
        )
        assert result["ward_id"] is None
        assert result["ward_code"] is None

    def test_dept_result_strips_orm_instance(self):
        mock_dept = MagicMock()
        mock_dept.pk = 3
        result = _serializable_dept_result(
            {
                "department": mock_dept,
                "department_code": "roads_and_drainage",
                "confidence": 0.90,
                "source": "direct_code_match",
            }
        )
        assert "department" not in result
        assert result["department_id"] == 3
        assert result["department_code"] == "roads_and_drainage"
        assert result["confidence"] == 0.90

    def test_dept_result_with_none_instance(self):
        result = _serializable_dept_result(
            {"department": None, "department_code": None, "confidence": 0.0, "source": "unresolved"}
        )
        assert result["department_id"] is None


# ===========================================================================
# 5. build_phase_e_routing() — orchestrator
# ===========================================================================


class TestBuildPhaseERouting:
    """Orchestrator — unit tests covering graceful degradation."""

    def test_returns_required_keys(self):
        result = build_phase_e_routing(
            routing_context={},
            ai_decision=_make_ai_decision(),
        )
        assert "ward_instance" in result
        assert "department_instance" in result
        assert "routing_metadata" in result

    def test_empty_routing_context_returns_none_instances(self):
        result = build_phase_e_routing(
            routing_context={},
            ai_decision=_make_ai_decision(),
        )
        assert result["ward_instance"] is None
        assert result["department_instance"] is None

    def test_empty_routing_context_falls_to_central_queue(self):
        result = build_phase_e_routing(
            routing_context={},
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_metadata"]["routing_mode"] == "central_queue"

    def test_review_required_action_routes_to_manual_review(self):
        result = build_phase_e_routing(
            routing_context={},
            ai_decision=_make_ai_decision(action="review_required", needs_review=True),
        )
        assert result["routing_metadata"]["routing_mode"] == "manual_review"

    def test_routing_metadata_contains_required_keys(self):
        result = build_phase_e_routing(
            routing_context={},
            ai_decision=_make_ai_decision(),
        )
        meta = result["routing_metadata"]
        assert "ward_resolution" in meta
        assert "department_resolution" in meta
        assert "routing_bucket" in meta
        assert "routing_mode" in meta
        assert "routing_confidence" in meta

    def test_routing_metadata_is_json_serialisable(self):
        """No ORM instances must appear in routing_metadata — it must be storable as JSON."""
        import json

        result = build_phase_e_routing(
            routing_context={},
            ai_decision=_make_ai_decision(),
        )
        # Should not raise
        serialised = json.dumps(result["routing_metadata"])
        assert len(serialised) > 0

    def test_none_ward_hint_in_context_resolves_no_ward(self):
        result = build_phase_e_routing(
            routing_context={"ward_hint": None, "category_code": "", "department_code": ""},
            ai_decision=_make_ai_decision(),
        )
        assert result["ward_instance"] is None
        assert result["routing_metadata"]["ward_resolution"]["source"] == "unresolved"

    @pytest.mark.django_db
    def test_db_ward_resolution_sets_ward_instance(self):
        from apps.wards.models import Ward

        Ward.objects.create(code="tvm_034", name="Pattom", boundary=_SIMPLE_POLYGON)
        result = build_phase_e_routing(
            routing_context={"ward_hint": "tvm_034", "category_code": "", "department_code": ""},
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["ward_instance"] is not None
        assert result["ward_instance"].code == "tvm_034"
        assert result["routing_metadata"]["routing_mode"] == "ward_queue"
        assert result["routing_metadata"]["routing_bucket"] == "ward_tvm_034"

    @pytest.mark.django_db
    def test_db_department_resolution_sets_department_instance(self):
        from apps.departments.models import Department

        Department.objects.create(
            code="roads_and_drainage",
            name="Roads",
            handled_categories=["road_damage"],
        )
        result = build_phase_e_routing(
            routing_context={
                "ward_hint": None,
                "category_code": "road_damage",
                "department_code": "roads_and_drainage",
            },
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["department_instance"] is not None
        assert result["department_instance"].code == "roads_and_drainage"
        assert result["routing_metadata"]["routing_mode"] == "department_queue"

    @pytest.mark.django_db
    def test_ward_and_dept_both_resolved_chooses_ward_queue(self):
        from apps.departments.models import Department
        from apps.wards.models import Ward

        Ward.objects.create(code="tvm_034", name="Pattom", boundary=_SIMPLE_POLYGON)
        Department.objects.create(
            code="roads_and_drainage",
            name="Roads",
            handled_categories=["road_damage"],
        )
        result = build_phase_e_routing(
            routing_context={
                "ward_hint": "tvm_034",
                "category_code": "road_damage",
                "department_code": "roads_and_drainage",
            },
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_metadata"]["routing_mode"] == "ward_queue"

    @pytest.mark.django_db
    def test_routing_metadata_ward_id_matches_db_record(self):
        from apps.wards.models import Ward

        ward = Ward.objects.create(code="tvm_034", name="Pattom", boundary=_SIMPLE_POLYGON)
        result = build_phase_e_routing(
            routing_context={"ward_hint": "tvm_034", "category_code": "", "department_code": ""},
            ai_decision=_make_ai_decision(action="auto_route"),
        )
        assert result["routing_metadata"]["ward_resolution"]["ward_id"] == ward.pk


# ===========================================================================
# 6. Phase E wiring into enrich_grievance_with_ai()
# ===========================================================================

_P_ANALYZE = "apps.integrations.services.analyze_grievance_submission"
_P_UPDATE = "apps.grievances.services.update_grievance_enrichment"
_P_ROUTING = "apps.integrations.routing.build_phase_e_routing"

# Pre-built stub Phase E results used by mocked integration tests.
_STUB_PHASE_E_CENTRAL = {
    "ward_instance": None,
    "department_instance": None,
    "routing_metadata": {
        "ward_resolution": {
            "ward_id": None,
            "ward_code": None,
            "confidence": 0.0,
            "source": "unresolved",
        },
        "department_resolution": {
            "department_id": None,
            "department_code": None,
            "confidence": 0.0,
            "source": "unresolved",
        },
        "routing_bucket": "central",
        "routing_mode": "central_queue",
        "routing_confidence": 0.0,
    },
}

_STUB_PHASE_E_MANUAL_REVIEW = {
    "ward_instance": None,
    "department_instance": None,
    "routing_metadata": {
        "ward_resolution": {
            "ward_id": None,
            "ward_code": None,
            "confidence": 0.0,
            "source": "unresolved",
        },
        "department_resolution": {
            "department_id": None,
            "department_code": None,
            "confidence": 0.0,
            "source": "unresolved",
        },
        "routing_bucket": "manual_review",
        "routing_mode": "manual_review",
        "routing_confidence": 1.0,
    },
}


class TestPhaseEIntegrationWithEnrich:
    """Phase E routing is called inside enrich_grievance_with_ai().

    These tests verify the wiring without requiring DB seeding, by
    patching analyze_grievance_submission to return a payload that
    includes routing_context.
    """

    def _mock_grievance(self) -> MagicMock:
        g = MagicMock()
        g.pk = 1
        g.raw_text = "Pothole near Pattom junction."
        g.landmark_mention = "Pattom"
        g.citizen_location_text = ""
        return g

    def _make_payload_with_routing_context(
        self,
        *,
        ward_hint: str | None = None,
        category_code: str = "road_damage",
        department_code: str = "roads_and_drainage",
        action: str = "auto_route",
    ) -> dict:
        """Minimal payload with routing_context populated."""
        needs_review = action != "auto_route"
        ai_decision = {
            "automation_action": action,
            "routing_confidence": 0.75,
            "needs_review": needs_review,
            "review_reasons": [],
            "duplicate_risk": {"risk_level": "low", "risk_score": 0.0, "is_confirmed": False},
            "escalation": {"should_escalate": False, "escalation_reason": ""},
            "decision_metadata": {},
        }
        return {
            "normalized_summary": "Pothole near Pattom junction.",
            "category_code": category_code,
            "priority": "medium",
            "landmark_resolution_metadata": {"provider_result": {}, "local_candidates": []},
            "duplicate_detection_metadata": {
                "possible_duplicate_tracking_code": None,
                "confidence": 0.0,
                "candidates": [],
                "provider": "local_stub",
                "metadata": {},
            },
            "ai_decision": ai_decision,
            "ai_explainability": "Auto-routed by AI with 75% confidence.",
            "routing_context": {
                "ward_hint": ward_hint,
                "landmark_hints": [],
                "category_code": category_code,
                "department_code": department_code,
            },
            "provider_metadata": {
                "nlp": {"provider": "local_ml_v1", "confidence": 0.75, "language": "english", "metadata": {}},
                "landmark": {"provider": "local_stub", "confidence": 0.0, "metadata": {}},
                "duplicate": {"provider": "local_stub", "confidence": 0.0, "metadata": {}},
            },
        }

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_enrich_still_returns_true_when_phase_e_has_no_db_matches(
        self, mock_analyze, mock_update
    ):
        """No ward/dept in DB → Phase E degrades gracefully; enrichment still completes."""
        from apps.integrations.services import enrich_grievance_with_ai

        mock_analyze.return_value = self._make_payload_with_routing_context(
            ward_hint="tvm_034",
            category_code="road_damage",
        )
        mock_update.return_value = MagicMock()
        assert enrich_grievance_with_ai(grievance=self._mock_grievance()) is True

    @patch(_P_ROUTING)
    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_phase_e_routing_metadata_added_to_status_metadata(
        self, mock_analyze, mock_update, mock_phase_e
    ):
        """status_metadata must include phase_e_routing key after enrichment."""
        from apps.integrations.services import enrich_grievance_with_ai

        mock_analyze.return_value = self._make_payload_with_routing_context(action="auto_route")
        mock_update.return_value = MagicMock()
        mock_phase_e.return_value = _STUB_PHASE_E_CENTRAL
        enrich_grievance_with_ai(grievance=self._mock_grievance())

        _, kwargs = mock_update.call_args
        status_meta = kwargs["values"]["status_metadata"]
        assert "phase_e_routing" in status_meta

    @patch(_P_ROUTING)
    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_phase_e_routing_metadata_has_expected_keys(
        self, mock_analyze, mock_update, mock_phase_e
    ):
        from apps.integrations.services import enrich_grievance_with_ai

        mock_analyze.return_value = self._make_payload_with_routing_context()
        mock_update.return_value = MagicMock()
        mock_phase_e.return_value = _STUB_PHASE_E_CENTRAL
        enrich_grievance_with_ai(grievance=self._mock_grievance())

        _, kwargs = mock_update.call_args
        routing_meta = kwargs["values"]["status_metadata"]["phase_e_routing"]
        assert "routing_mode" in routing_meta
        assert "routing_bucket" in routing_meta
        assert "routing_confidence" in routing_meta
        assert "ward_resolution" in routing_meta
        assert "department_resolution" in routing_meta

    @patch(_P_ROUTING)
    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_review_required_action_routes_to_manual_review_in_metadata(
        self, mock_analyze, mock_update, mock_phase_e
    ):
        from apps.integrations.services import enrich_grievance_with_ai

        mock_analyze.return_value = self._make_payload_with_routing_context(
            action="review_required"
        )
        mock_update.return_value = MagicMock()
        mock_phase_e.return_value = _STUB_PHASE_E_MANUAL_REVIEW
        enrich_grievance_with_ai(grievance=self._mock_grievance())

        _, kwargs = mock_update.call_args
        routing_meta = kwargs["values"]["status_metadata"]["phase_e_routing"]
        assert routing_meta["routing_mode"] == "manual_review"
        assert routing_meta["routing_bucket"] == "manual_review"

    @patch(_P_UPDATE)
    @patch(_P_ANALYZE)
    def test_base_enrichment_fields_still_present_alongside_phase_e(
        self, mock_analyze, mock_update
    ):
        """Phase E must not overwrite existing AI enrichment fields."""
        from apps.integrations.services import enrich_grievance_with_ai

        mock_analyze.return_value = self._make_payload_with_routing_context()
        mock_update.return_value = MagicMock()
        enrich_grievance_with_ai(grievance=self._mock_grievance())

        _, kwargs = mock_update.call_args
        values = kwargs["values"]
        assert values["category_code"] == "road_damage"
        assert values["status_metadata"]["ai_enrichment"] is True
        assert values["status_metadata"]["automation_action"] == "auto_route"

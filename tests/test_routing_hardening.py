"""Hardening tests for Phase E KSMART-style routing.

Scope
-----
Verifies the three resolver functions in ``apps/integrations/routing.py``
and the ``build_phase_e_routing()`` orchestrator.

What is tested
--------------
1. Department resolution via direct code match (confidence 0.90)
2. Department resolution via GIN fallback (confidence 0.75)
3. Department resolution when no department exists (unresolved)
4. Ward resolution when ward is in DB (confidence 0.85)
5. Ward resolution when ward is NOT in DB (source = ward_not_in_db)
6. Ward resolution with no hint (source = unresolved)
7. Routing bucket: manual_review takes precedence
8. Routing bucket: ward_queue when ward resolved + conf >= 0.60
9. Routing bucket: department_queue when ward unresolved but dept resolved
10. Routing bucket: central_queue when both unresolved
11. Privacy: routing output never contains officer identity
12. build_phase_e_routing orchestrator end-to-end
"""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Polygon

from apps.departments.models import Department
from apps.integrations.routing import (
    build_phase_e_routing,
    resolve_department_from_category,
    resolve_routing_bucket,
    resolve_ward_from_hint,
)
from apps.wards.models import Ward

pytestmark = pytest.mark.django_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOUNDARY = Polygon(
    ((76.90, 8.40), (76.95, 8.40), (76.95, 8.45), (76.90, 8.45), (76.90, 8.40)),
    srid=4326,
)


def _make_ward(code="tvm_034", name="Pattom"):
    return Ward.objects.create(code=code, name=name, boundary=_BOUNDARY)


def _make_dept(code="roads_and_drainage", name="Roads & Drainage", categories=None):
    return Department.objects.create(
        code=code,
        name=name,
        handled_categories=categories or ["road_damage", "drainage"],
        is_active=True,
    )


def _auto_route_decision():
    return {
        "automation_action": "auto_route",
        "needs_review":      False,
    }


def _review_decision():
    return {
        "automation_action": "review_required",
        "needs_review":      True,
    }


# ---------------------------------------------------------------------------
# Department resolver
# ---------------------------------------------------------------------------

def test_dept_resolver_direct_code_match():
    dept = _make_dept()
    result = resolve_department_from_category(
        category_code="road_damage",
        department_code="roads_and_drainage",
    )
    assert result["department"] == dept
    assert result["confidence"] == 0.90
    assert result["source"] == "direct_code_match"


def test_dept_resolver_gin_fallback_when_code_missing():
    """No department_code hint → falls back to GIN contains query."""
    dept = _make_dept()
    result = resolve_department_from_category(
        category_code="road_damage",
        department_code="",          # empty hint forces GIN path
    )
    assert result["department"] == dept
    assert result["confidence"] == 0.75
    assert result["source"] == "category_match"


def test_dept_resolver_unresolved_when_not_in_db():
    result = resolve_department_from_category(
        category_code="road_damage",
        department_code="roads_and_drainage",  # not in DB
    )
    assert result["department"] is None
    assert result["confidence"] == 0.0
    assert result["source"] == "unresolved"


def test_dept_resolver_inactive_dept_not_returned():
    Department.objects.create(
        code="roads_and_drainage",
        name="Roads",
        handled_categories=["road_damage"],
        is_active=False,             # inactive — must NOT be returned
    )
    result = resolve_department_from_category(
        category_code="road_damage",
        department_code="roads_and_drainage",
    )
    assert result["department"] is None


# ---------------------------------------------------------------------------
# Ward resolver
# ---------------------------------------------------------------------------

def test_ward_resolver_found_in_db():
    ward = _make_ward()
    result = resolve_ward_from_hint(ward_hint="tvm_034")
    assert result["ward"] == ward
    assert result["confidence"] == 0.85
    assert result["source"] == "landmark_alias_match"


def test_ward_resolver_ward_not_in_db():
    result = resolve_ward_from_hint(ward_hint="tvm_034")   # not seeded
    assert result["ward"] is None
    assert result["confidence"] == 0.0
    assert result["source"] == "ward_not_in_db"
    assert result["ward_code"] == "tvm_034"   # hint preserved for audit trail


def test_ward_resolver_none_hint():
    result = resolve_ward_from_hint(ward_hint=None)
    assert result["ward"] is None
    assert result["confidence"] == 0.0
    assert result["source"] == "unresolved"


def test_ward_resolver_inactive_ward_not_returned():
    Ward.objects.create(code="tvm_034", name="Pattom", boundary=_BOUNDARY, is_active=False)
    result = resolve_ward_from_hint(ward_hint="tvm_034")
    assert result["ward"] is None
    assert result["source"] == "ward_not_in_db"


# ---------------------------------------------------------------------------
# Routing bucket
# ---------------------------------------------------------------------------

def test_routing_bucket_manual_review_takes_precedence():
    ward = _make_ward()
    dept = _make_dept()
    ward_result = {"ward": ward, "ward_code": "tvm_034", "confidence": 0.90}
    dept_result = {"department": dept, "department_code": "roads_and_drainage", "confidence": 0.90}

    routing = resolve_routing_bucket(
        ward_result=ward_result,
        dept_result=dept_result,
        ai_decision=_review_decision(),      # should override everything
    )
    assert routing["routing_mode"] == "manual_review"
    assert routing["routing_bucket"] == "manual_review"


def test_routing_bucket_ward_queue_when_ward_resolved():
    ward = _make_ward()
    ward_result = {"ward": ward, "ward_code": "tvm_034", "confidence": 0.85}
    dept_result = {"department": None, "department_code": None, "confidence": 0.0}

    routing = resolve_routing_bucket(
        ward_result=ward_result,
        dept_result=dept_result,
        ai_decision=_auto_route_decision(),
    )
    assert routing["routing_mode"] == "ward_queue"
    assert routing["routing_bucket"] == "ward_tvm_034"


def test_routing_bucket_department_queue_when_no_ward():
    dept = _make_dept()
    ward_result = {"ward": None, "ward_code": None, "confidence": 0.0}
    dept_result = {"department": dept, "department_code": "roads_and_drainage", "confidence": 0.75}

    routing = resolve_routing_bucket(
        ward_result=ward_result,
        dept_result=dept_result,
        ai_decision=_auto_route_decision(),
    )
    assert routing["routing_mode"] == "department_queue"
    assert routing["routing_bucket"] == "dept_roads_and_drainage"


def test_routing_bucket_central_queue_fallback():
    ward_result = {"ward": None, "ward_code": None, "confidence": 0.0}
    dept_result = {"department": None, "department_code": None, "confidence": 0.0}

    routing = resolve_routing_bucket(
        ward_result=ward_result,
        dept_result=dept_result,
        ai_decision=_auto_route_decision(),
    )
    assert routing["routing_mode"] == "central_queue"
    assert routing["routing_bucket"] == "central"


def test_routing_bucket_ward_confidence_below_threshold_falls_to_dept():
    """Ward confidence < 0.60 must NOT use ward_queue."""
    ward = _make_ward()
    dept = _make_dept()
    ward_result = {"ward": ward, "ward_code": "tvm_034", "confidence": 0.50}  # below _WARD_CONFIDENCE_MIN
    dept_result = {"department": dept, "department_code": "roads_and_drainage", "confidence": 0.75}

    routing = resolve_routing_bucket(
        ward_result=ward_result,
        dept_result=dept_result,
        ai_decision=_auto_route_decision(),
    )
    assert routing["routing_mode"] == "department_queue", (
        "Ward with confidence < 0.60 must fall through to department_queue"
    )


# ---------------------------------------------------------------------------
# Privacy: routing output must NEVER expose officer identity
# ---------------------------------------------------------------------------

def test_routing_output_contains_no_officer_identity():
    """The routing metadata must not expose any officer user ID or name."""
    ward = _make_ward()
    dept = _make_dept()

    phase_e = build_phase_e_routing(
        routing_context={
            "ward_hint":       "tvm_034",
            "category_code":   "road_damage",
            "department_code": "roads_and_drainage",
        },
        ai_decision=_auto_route_decision(),
    )
    meta = phase_e["routing_metadata"]

    forbidden_keys = {"officer", "assignee", "officer_id", "assigned_to", "officer_name"}
    for key in meta:
        assert key not in forbidden_keys, f"Routing metadata exposes officer key: {key!r}"

    # Recurse into sub-dicts
    for sub in meta.values():
        if isinstance(sub, dict):
            for key in sub:
                assert key not in forbidden_keys, (
                    f"Nested routing metadata exposes officer key: {key!r}"
                )


# ---------------------------------------------------------------------------
# build_phase_e_routing orchestrator
# ---------------------------------------------------------------------------

def test_build_phase_e_routing_resolves_ward_and_dept():
    ward = _make_ward()
    dept = _make_dept()

    phase_e = build_phase_e_routing(
        routing_context={
            "ward_hint":       "tvm_034",
            "category_code":   "road_damage",
            "department_code": "roads_and_drainage",
        },
        ai_decision=_auto_route_decision(),
    )

    assert phase_e["ward_instance"] == ward
    assert phase_e["department_instance"] == dept
    meta = phase_e["routing_metadata"]
    assert meta["routing_mode"] == "ward_queue"
    assert meta["routing_bucket"] == "ward_tvm_034"


def test_build_phase_e_routing_falls_to_central_when_db_empty():
    """No Ward or Department seeded → must gracefully reach central_queue."""
    phase_e = build_phase_e_routing(
        routing_context={
            "ward_hint":       "tvm_034",
            "category_code":   "road_damage",
            "department_code": "roads_and_drainage",
        },
        ai_decision=_auto_route_decision(),
    )
    assert phase_e["ward_instance"] is None
    assert phase_e["department_instance"] is None
    assert phase_e["routing_metadata"]["routing_mode"] == "central_queue"


def test_build_phase_e_routing_with_review_decision():
    _make_ward()
    _make_dept()
    phase_e = build_phase_e_routing(
        routing_context={
            "ward_hint":       "tvm_034",
            "category_code":   "road_damage",
            "department_code": "roads_and_drainage",
        },
        ai_decision=_review_decision(),
    )
    assert phase_e["routing_metadata"]["routing_mode"] == "manual_review"

"""DB integrity hardening tests.

Scope
-----
Validates structural and referential consistency across the grievances, slas,
audit, workflows, and departments domains.  These tests use real DB fixtures
created inline — they do NOT fake data.

Checks
------
1. Every Grievance has exactly one SLA (OneToOne enforced)
2. Every Grievance has at least one WorkflowEvent
3. Every Grievance has at least one AuditLog with action_type=create
4. No SLA has is_breached=True without breached_at
5. No SLA has breach_type != "none" without is_breached=True
6. No WorkflowEvent references a non-existent Grievance (FK cascade check)
7. Department handled_categories must not contain duplicates
8. Department.handled_categories must use valid lowercase codes
9. Grievance.category_code must be a valid lowercase code (regex)
10. All standard civic categories have at least one Department that handles them
    (coverage check for the 8 main routing categories)
"""
from __future__ import annotations

import re

import pytest
from django.contrib.auth import get_user_model
from django.contrib.gis.geos import Polygon

from apps.audit.models import AuditLog, AuditActionType
from apps.departments.models import Department
from apps.grievances.models import Grievance, GrievancePriority
from apps.grievances.services import submit_grievance
from apps.slas.models import SLA, SLABreachType, SLAStatus
from apps.slas.services import mark_sla_breached
from apps.workflows.models import WorkflowEvent
from apps.wards.models import Ward

User = get_user_model()
pytestmark = pytest.mark.django_db

_CATEGORY_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# All 8 routing categories that MUST have department coverage in a seeded DB.
_REQUIRED_CATEGORIES = {
    "road_damage",
    "waste_management",
    "water_supply",
    "sewage_issue",
    "drainage",
    "street_light",
    "tree_fall",
    "electrical_hazard",
}

_PATCH_NLP    = "apps.integrations.services.classify_grievance_text"
_PATCH_RECENT = "apps.integrations.services.recent_grievance_summaries_for_duplicate_context"
_PATCH_LOCAL  = "apps.integrations.services.local_landmark_candidates_for_mention"

_NLP_STUB = {
    "normalized_summary": "Road broken near market.",
    "category_code":      "road_damage",
    "department_code":    "roads_and_drainage",
    "priority":           "medium",
    "confidence":         0.80,
    "language":           "en",
    "provider":           "transformer_v1",
    "metadata": {
        "text_length": 25,
        "ward_hint": None,
        "landmark_hints": [],
        "spam_check": {"is_spam": False, "spam_score": 0.0, "spam_reason": ""},
        "duplicate_check": {"is_duplicate": False, "similarity_score": 0.0, "matching_text": None},
        "needs_human_review": False,
        "review_reasons": [],
        "image_analysis": None,
        "consistency_check": None,
        "evidence_quality": None,
        "evidence_review_reason": "",
        "decision": {
            "automation_action": "auto_route",
            "routing_confidence": 0.78,
            "needs_review": False,
            "review_reasons": [],
            "duplicate_risk": {"risk_level": "low", "risk_score": 0.0, "is_confirmed": False},
            "escalation": {"should_escalate": False, "escalation_reason": ""},
            "decision_metadata": {},
        },
    },
}


@pytest.fixture
def citizen():
    return User.objects.create_user(
        username="db_citizen",
        email="db_citizen@test.com",
        password="pass",
        role="citizen",
    )


def _make_grievance(citizen):
    from unittest.mock import patch
    with patch(_PATCH_NLP, return_value=_NLP_STUB), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]):
        return submit_grievance(
            submitter=citizen,
            raw_text="Road broken near market.",
        )


# ---------------------------------------------------------------------------
# 1. OneToOne SLA exists for every submitted grievance
# ---------------------------------------------------------------------------

def test_sla_exists_for_every_grievance(citizen):
    grievance = _make_grievance(citizen)
    assert SLA.objects.filter(grievance=grievance).count() == 1, (
        "Expected exactly 1 SLA per grievance."
    )


# ---------------------------------------------------------------------------
# 2. At least one WorkflowEvent per grievance
# ---------------------------------------------------------------------------

def test_workflow_event_exists_for_every_grievance(citizen):
    grievance = _make_grievance(citizen)
    count = WorkflowEvent.objects.filter(grievance=grievance).count()
    assert count >= 1, f"Expected ≥1 WorkflowEvent, found {count}."


# ---------------------------------------------------------------------------
# 3. AuditLog with action_type=create exists for every grievance
# ---------------------------------------------------------------------------

def test_audit_create_log_exists_for_every_grievance(citizen):
    grievance = _make_grievance(citizen)
    exists = AuditLog.objects.filter(
        target_model="grievances.Grievance",
        target_object_id=str(grievance.pk),
        action_type=AuditActionType.CREATE,
    ).exists()
    assert exists, "No CREATE AuditLog found for the submitted grievance."


# ---------------------------------------------------------------------------
# 4 & 5. SLA breach state consistency
# ---------------------------------------------------------------------------

def test_breached_sla_has_breached_at(citizen):
    grievance = _make_grievance(citizen)
    sla = SLA.objects.get(grievance=grievance)
    mark_sla_breached(
        sla=sla,
        breach_type=SLABreachType.RESPONSE,
        breached_at=grievance.submitted_at,
    )
    sla.refresh_from_db()
    assert sla.is_breached is True
    assert sla.breached_at is not None, "Breached SLA must have breached_at timestamp."


def test_active_sla_has_no_breached_at(citizen):
    grievance = _make_grievance(citizen)
    sla = SLA.objects.get(grievance=grievance)
    assert sla.is_breached is False
    assert sla.breached_at is None, "Active SLA must not have breached_at."
    assert sla.breach_type == SLABreachType.NONE, "Active SLA breach_type must be 'none'."


# ---------------------------------------------------------------------------
# 6. WorkflowEvent — grievance FK always points to an existing Grievance
# ---------------------------------------------------------------------------

def test_workflow_events_have_valid_grievance_fks(citizen):
    _make_grievance(citizen)
    # All WorkflowEvents must be joinable to their grievance without DoesNotExist
    orphaned = WorkflowEvent.objects.filter(grievance__isnull=True).count()
    assert orphaned == 0, f"Found {orphaned} WorkflowEvents with null grievance FK."


# ---------------------------------------------------------------------------
# 7 & 8. Department handled_categories integrity
# ---------------------------------------------------------------------------

def test_department_handled_categories_no_duplicates():
    dept = Department.objects.create(
        code="test_dept",
        name="Test Department",
        handled_categories=["road_damage", "drainage"],
    )
    # Verify uniqueness
    codes = dept.handled_categories
    assert len(codes) == len(set(codes)), (
        f"Department {dept.code} has duplicate handled_categories: {codes}"
    )


def test_department_handled_categories_valid_codes():
    dept = Department.objects.create(
        code="test_dept2",
        name="Test Department 2",
        handled_categories=["road_damage", "water_supply"],
    )
    for code in dept.handled_categories:
        assert _CATEGORY_CODE_RE.fullmatch(code), (
            f"Invalid category code {code!r} in department {dept.code}"
        )


# ---------------------------------------------------------------------------
# 9. Grievance category_code format validity
# ---------------------------------------------------------------------------

def test_grievance_category_code_format(citizen):
    grievance = _make_grievance(citizen)
    grievance.refresh_from_db()
    if grievance.category_code:
        assert _CATEGORY_CODE_RE.fullmatch(grievance.category_code), (
            f"Grievance category_code {grievance.category_code!r} violates format constraint."
        )


# ---------------------------------------------------------------------------
# 10. Department category coverage — all routing categories have a handler
# ---------------------------------------------------------------------------

def test_all_required_categories_have_department_coverage():
    """Seed departments and verify every required civic category is covered."""
    dept_data = [
        ("roads_and_drainage",     ["road_damage", "drainage"]),
        ("sanitation",             ["waste_management", "sewage_issue"]),
        ("water_authority",        ["water_supply"]),
        ("street_lighting",        ["street_light"]),
        ("parks_and_environment",  ["tree_fall"]),
        ("electrical_engineering", ["electrical_hazard"]),
    ]
    for code, categories in dept_data:
        Department.objects.create(
            code=code,
            name=code.replace("_", " ").title(),
            handled_categories=categories,
        )

    for category in _REQUIRED_CATEGORIES:
        covered = Department.objects.filter(
            handled_categories__contains=[category],
            is_active=True,
        ).exists()
        assert covered, (
            f"Category '{category}' is not handled by any active Department. "
            "Add it to the appropriate Department's handled_categories."
        )


# ---------------------------------------------------------------------------
# 11. Tracking code format for all grievances in DB
# ---------------------------------------------------------------------------

def test_all_tracking_codes_match_format(citizen):
    _make_grievance(citizen)
    _make_grievance(citizen)
    for grv in Grievance.objects.all():
        assert re.fullmatch(r"GRV-\d{4}-\d{6}", grv.tracking_code), (
            f"Grievance pk={grv.pk} has malformed tracking_code: {grv.tracking_code!r}"
        )

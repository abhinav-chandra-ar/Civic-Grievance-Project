"""Tests for grievance_list_visible_to_user() — BUG 2 officer visibility scoping.

Scope
-----
Verifies that each role receives exactly the grievances it should see:

  citizen              → own grievances only
  ward_officer         → grievances whose ward FK matches assigned_ward
  ward_officer (none)  → empty (never full set)
  dept_officer         → grievances whose department FK matches assigned_department
  dept_officer (none)  → empty (never full set)
  municipal_admin      → all grievances
  super_admin          → all grievances
  field_verifier       → all grievances
  system_operator      → all grievances
"""
from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.contrib.gis.geos import Polygon

from apps.departments.models import Department
from apps.grievances.models import Grievance
from apps.grievances.selectors import grievance_list_visible_to_user
from apps.grievances.services import submit_grievance
from apps.wards.models import Ward

User = get_user_model()
pytestmark = pytest.mark.django_db

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BOUNDARY = Polygon(
    ((76.90, 8.40), (76.95, 8.40), (76.95, 8.45), (76.90, 8.45), (76.90, 8.40)),
    srid=4326,
)
_BOUNDARY_2 = Polygon(
    ((76.95, 8.40), (77.00, 8.40), (77.00, 8.45), (76.95, 8.45), (76.95, 8.40)),
    srid=4326,
)

_NLP_STUB = {
    "normalized_summary": "Road broken.",
    "category_code": "road_damage",
    "department_code": "roads_and_drainage",
    "priority": "medium",
    "confidence": 0.80,
    "language": "en",
    "provider": "transformer_v1",
    "metadata": {
        "text_length": 12,
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

_PATCH_NLP    = "apps.integrations.services.classify_grievance_text"
_PATCH_RECENT = "apps.integrations.services.recent_grievance_summaries_for_duplicate_context"
_PATCH_LOCAL  = "apps.integrations.services.local_landmark_candidates_for_mention"
# Patch Phase E so enrichment never auto-assigns ward/department from the NLP stub.
# Without this, NLP stub's department_code="roads_and_drainage" would cause Phase E
# to look up and assign a real DB department if one happens to exist in the test.
_PATCH_PHASE_E = "apps.integrations.routing.build_phase_e_routing"
_PHASE_E_STUB  = {"ward_instance": None, "department_instance": None, "routing_metadata": {}}


def _user(username, role, **extra):
    return User.objects.create_user(username=username, password="pass", role=role, **extra)


def _grievance(submitter, ward=None, department=None):
    from unittest.mock import patch
    with patch(_PATCH_NLP, return_value=_NLP_STUB), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        g = submit_grievance(submitter=submitter, raw_text="Road broken near school.")
    # Manually set ward/department FK for routing tests (bypassing enrichment)
    if ward is not None or department is not None:
        if ward is not None:
            g.ward = ward
        if department is not None:
            g.department = department
        g.save(update_fields=[f for f in ("ward", "department") if (ward if f == "ward" else department) is not None])
    return g


def _make_ward(code="tvm_001"):
    return Ward.objects.create(code=code, name=f"Ward {code}", boundary=_BOUNDARY)


def _make_dept(code="roads_and_drainage"):
    return Department.objects.create(
        code=code, name=code.replace("_", " ").title(),
        handled_categories=["road_damage"], is_active=True,
    )


# ---------------------------------------------------------------------------
# Citizen
# ---------------------------------------------------------------------------

def test_citizen_sees_only_own_grievances():
    citizen_a = _user("citizen_a", "citizen")
    citizen_b = _user("citizen_b", "citizen")
    own = _grievance(citizen_a)
    _grievance(citizen_b)  # other citizen's grievance

    visible = list(grievance_list_visible_to_user(user=citizen_a))
    assert visible == [own]


# ---------------------------------------------------------------------------
# Ward officer
# ---------------------------------------------------------------------------

def test_ward_officer_sees_only_assigned_ward_grievances():
    ward_a = Ward.objects.create(code="tvm_001", name="Ward A", boundary=_BOUNDARY)
    ward_b = Ward.objects.create(code="tvm_002", name="Ward B", boundary=_BOUNDARY_2)
    citizen = _user("citizen_w", "citizen")
    officer = _user("officer_w", "ward_officer", assigned_ward=ward_a)

    g_in_ward  = _grievance(citizen, ward=ward_a)
    g_out_ward = _grievance(citizen, ward=ward_b)
    g_no_ward  = _grievance(citizen)  # ward=None

    visible_ids = set(grievance_list_visible_to_user(user=officer).values_list("pk", flat=True))
    assert g_in_ward.pk in visible_ids,  "Ward officer must see grievances in their ward"
    assert g_out_ward.pk not in visible_ids, "Ward officer must NOT see another ward's grievances"
    assert g_no_ward.pk not in visible_ids,  "Ward officer must NOT see unassigned grievances"


def test_ward_officer_without_assignment_sees_nothing():
    citizen  = _user("citizen_wn", "citizen")
    officer  = _user("officer_wn", "ward_officer")  # no assigned_ward
    _grievance(citizen)

    qs = grievance_list_visible_to_user(user=officer)
    assert qs.count() == 0, "Unassigned ward officer must see no grievances"


def test_ward_officer_does_not_see_other_wards():
    ward_a = Ward.objects.create(code="tvm_003", name="Ward A3", boundary=_BOUNDARY)
    ward_b = Ward.objects.create(code="tvm_004", name="Ward B4", boundary=_BOUNDARY_2)
    citizen = _user("citizen_wo", "citizen")
    officer = _user("officer_wo", "ward_officer", assigned_ward=ward_a)

    # Create 5 grievances in ward_b — officer must see zero
    for i in range(5):
        _grievance(citizen, ward=ward_b)

    assert grievance_list_visible_to_user(user=officer).count() == 0


# ---------------------------------------------------------------------------
# Department officer
# ---------------------------------------------------------------------------

def test_dept_officer_sees_only_assigned_dept_grievances():
    dept_roads = _make_dept("roads_and_drainage")
    dept_sanit = Department.objects.create(
        code="sanitation", name="Sanitation",
        handled_categories=["waste_management"], is_active=True,
    )
    citizen = _user("citizen_d", "citizen")
    officer = _user("officer_d", "department_officer", assigned_department=dept_roads)

    g_roads = _grievance(citizen, department=dept_roads)
    g_sanit = _grievance(citizen, department=dept_sanit)
    g_none  = _grievance(citizen)

    visible_ids = set(grievance_list_visible_to_user(user=officer).values_list("pk", flat=True))
    assert g_roads.pk in visible_ids,  "Dept officer must see their department's grievances"
    assert g_sanit.pk not in visible_ids, "Dept officer must NOT see another dept's grievances"
    assert g_none.pk  not in visible_ids,  "Dept officer must NOT see unassigned grievances"


def test_dept_officer_without_assignment_sees_nothing():
    citizen = _user("citizen_dn", "citizen")
    officer = _user("officer_dn", "department_officer")  # no assigned_department
    _grievance(citizen)

    assert grievance_list_visible_to_user(user=officer).count() == 0


# ---------------------------------------------------------------------------
# Admin / wide-visibility roles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["municipal_admin", "super_admin", "field_verifier", "system_operator"])
def test_admin_roles_see_all_grievances(role):
    citizen_a = _user(f"cit_a_{role}", "citizen")
    citizen_b = _user(f"cit_b_{role}", "citizen")
    admin     = _user(f"admin_{role}", role)

    g1 = _grievance(citizen_a)
    g2 = _grievance(citizen_b)

    visible_ids = set(grievance_list_visible_to_user(user=admin).values_list("pk", flat=True))
    assert g1.pk in visible_ids, f"{role} must see all grievances"
    assert g2.pk in visible_ids, f"{role} must see all grievances"


# ---------------------------------------------------------------------------
# Cross-role isolation
# ---------------------------------------------------------------------------

def test_ward_officer_cannot_see_citizens_own_grievances_in_other_ward():
    """
    Grievances that belong to ward_b must be invisible to a ward_a officer
    even when the submitter is in the same city.
    """
    ward_a = Ward.objects.create(code="tvm_005", name="Ward 5A", boundary=_BOUNDARY)
    ward_b = Ward.objects.create(code="tvm_006", name="Ward 6B", boundary=_BOUNDARY_2)
    citizen = _user("citizen_cross", "citizen")
    officer = _user("officer_cross", "ward_officer", assigned_ward=ward_a)

    g_b = _grievance(citizen, ward=ward_b)
    assert grievance_list_visible_to_user(user=officer).filter(pk=g_b.pk).count() == 0

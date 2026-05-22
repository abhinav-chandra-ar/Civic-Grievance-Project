"""Read-side queries for workflow history."""
from __future__ import annotations

from typing import Any

from django.db.models import QuerySet

from .models import WorkflowEvent

WORKFLOW_OPERATOR_ROLES = frozenset(
    {
        "ward_officer",
        "department_officer",
        "municipal_admin",
        "super_admin",
        "field_verifier",
        "system_operator",
    }
)


def workflow_event_list() -> QuerySet[WorkflowEvent]:
    """Return workflow events with primary display relations loaded."""
    return WorkflowEvent.objects.select_related("grievance", "actor", "assignee")


def workflow_event_list_for_grievance(*, grievance) -> QuerySet[WorkflowEvent]:
    """Return history events for one grievance."""
    return workflow_event_list().filter(grievance=grievance)


def workflow_event_list_visible_to_user(*, user: Any) -> QuerySet[WorkflowEvent]:
    """Return grievance-owned events unless the role is operational."""
    events = workflow_event_list()
    if getattr(user, "role", None) in WORKFLOW_OPERATOR_ROLES:
        return events
    return events.filter(grievance__submitter=user)

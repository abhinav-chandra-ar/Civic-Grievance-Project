"""Signal handlers that wire domain signals to email notifications.

Registered via IntegrationsConfig.ready().  All handlers are
side-effect-free with respect to the database — they only call email
helpers, which catch and log their own exceptions.  No handler may
raise: an unhandled exception here would propagate into the calling
transaction and roll it back.
"""
from __future__ import annotations

import logging

from django.dispatch import receiver

from apps.grievances.signals import grievance_submitted
from apps.slas.signals import sla_breached
from apps.workflows.models import WorkflowTransitionType
from apps.workflows.signals import workflow_event_recorded

from .emails import (
    send_grievance_status_changed_email,
    send_grievance_submitted_email,
    send_sla_breach_alert_email,
)

logger = logging.getLogger(__name__)


@receiver(grievance_submitted)
def handle_grievance_submitted(sender, *, grievance, **kwargs) -> None:
    """Event 1: confirmation email to the citizen on new grievance creation."""
    send_grievance_submitted_email(grievance)


@receiver(workflow_event_recorded)
def handle_workflow_event_recorded(sender, *, workflow_event, **kwargs) -> None:
    """Event 2: status-update email to the citizen on meaningful state transitions.

    Excluded cases (no email sent):
    - COMMENT events — no status change, remarks-only.
    - Initial creation event — previous_status == new_status == "submitted";
      the grievance_submitted handler already covers this moment.
    """
    if workflow_event.transition_type == WorkflowTransitionType.COMMENT:
        return
    if workflow_event.new_status == workflow_event.previous_status:
        return
    send_grievance_status_changed_email(workflow_event)


@receiver(sla_breached)
def handle_sla_breached(sender, *, sla, **kwargs) -> None:
    """Event 3: breach alert to all active admin/operator users."""
    send_sla_breach_alert_email(sla)

"""Email helpers for grievance lifecycle notifications.

Each function is a pure send-and-forget helper: it accepts a domain
object, builds a plain-text message, and calls Django's send_mail /
send_mass_mail.  Failures are caught, logged, and swallowed so that
email never rolls back a grievance transaction.

IDEMPOTENCY NOTE
---------------
These helpers currently send unconditionally.  When Celery retry tasks
are introduced, add a deduplication check immediately before each
send_mail / send_mass_mail call.  The check should look up a token keyed
on (event_type, object_pk, idempotency_key) in cache or a dedupe table
and short-circuit if already sent.  Each helper marks the insertion
point with a ``# FUTURE-IDEMPOTENCY`` comment.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.mail import send_mail, send_mass_mail

if TYPE_CHECKING:
    from apps.grievances.models import Grievance
    from apps.slas.models import SLA
    from apps.workflows.models import WorkflowEvent

logger = logging.getLogger(__name__)

_PORTAL_SIGNATURE = "— Civic Grievance Portal"
_ALERT_ROLES = frozenset({"municipal_admin", "super_admin", "system_operator"})


# ---------------------------------------------------------------------------
# Event 1 — grievance submitted
# ---------------------------------------------------------------------------

def send_grievance_submitted_email(grievance: "Grievance") -> None:
    """Notify the submitter that their grievance was received."""
    submitter = grievance.submitter
    if not submitter or not submitter.email:
        logger.warning(
            "grievance_submitted_email: skipping — no email on submitter of %s",
            grievance.tracking_code,
        )
        return

    greeting = f"Dear {submitter.first_name}," if submitter.first_name else "Dear Citizen,"
    raw = grievance.raw_text or ""
    excerpt = raw[:200].strip() + ("…" if len(raw) > 200 else "")
    submitted_at = grievance.submitted_at.strftime("%d %b %Y, %H:%M")
    status_label = grievance.status.replace("_", " ").title()

    subject = f"[Grievance Received] {grievance.tracking_code}"
    body = (
        f"{greeting}\n\n"
        "We have received your complaint and it is now being processed.\n\n"
        f"Tracking Code : {grievance.tracking_code}\n"
        f"Status        : {status_label}\n"
        f"Submitted At  : {submitted_at}\n\n"
        "Your Complaint:\n"
        f"{excerpt}\n\n"
        "Please keep your tracking code safe — you will need it to follow up.\n\n"
        f"{_PORTAL_SIGNATURE}"
    )

    # FUTURE-IDEMPOTENCY: check/insert token ("grievance_submitted", grievance.pk) before send
    _safe_send(
        subject=subject,
        body=body,
        recipient=submitter.email,
        context=f"grievance_submitted:{grievance.tracking_code}",
    )


# ---------------------------------------------------------------------------
# Event 2 — grievance status changed
# ---------------------------------------------------------------------------

def send_grievance_status_changed_email(workflow_event: "WorkflowEvent") -> None:
    """Notify the submitter when their grievance moves to a new status."""
    grievance = workflow_event.grievance
    submitter = grievance.submitter
    if not submitter or not submitter.email:
        logger.warning(
            "grievance_status_changed_email: skipping — no email on submitter of %s",
            grievance.tracking_code,
        )
        return

    previous_label = workflow_event.previous_status.replace("_", " ").title()
    new_label = workflow_event.new_status.replace("_", " ").title()
    occurred_at = workflow_event.occurred_at.strftime("%d %b %Y, %H:%M")

    remarks = (workflow_event.remarks or "").strip()
    remarks_line = f"Officer Remarks : {remarks}\n" if remarks else ""

    subject = f"[Grievance Update] {grievance.tracking_code} — Status: {new_label}"
    body = (
        "Dear Citizen,\n\n"
        "The status of your grievance has been updated.\n\n"
        f"Tracking Code   : {grievance.tracking_code}\n"
        f"Previous Status : {previous_label}\n"
        f"New Status      : {new_label}\n"
        f"{remarks_line}"
        f"Updated At      : {occurred_at}\n\n"
        f"{_PORTAL_SIGNATURE}"
    )

    # FUTURE-IDEMPOTENCY: check/insert token ("grievance_status_changed", workflow_event.pk) before send
    _safe_send(
        subject=subject,
        body=body,
        recipient=submitter.email,
        context=f"grievance_status_changed:{grievance.tracking_code}:{workflow_event.pk}",
    )


# ---------------------------------------------------------------------------
# Event 3 — SLA breached
# ---------------------------------------------------------------------------

def send_sla_breach_alert_email(sla: "SLA") -> None:
    """Send an SLA breach alert to all active municipal_admin / super_admin / system_operator users."""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    recipients = list(
        User.objects.filter(role__in=_ALERT_ROLES, is_active=True)
        .exclude(email="")
        .values_list("email", flat=True)
    )

    if not recipients:
        logger.warning(
            "sla_breach_alert_email: no active admin recipients found for SLA %s",
            sla.sla_code,
        )
        return

    grievance = sla.grievance
    status_label = grievance.status.replace("_", " ").title()
    breach_label = sla.breach_type.replace("_", " ").title()
    response_due = sla.response_due_at.strftime("%d %b %Y, %H:%M")
    resolution_due = sla.resolution_due_at.strftime("%d %b %Y, %H:%M")
    breached_at = sla.breached_at.strftime("%d %b %Y, %H:%M") if sla.breached_at else "N/A"

    subject = f"[SLA BREACH ALERT] {grievance.tracking_code} — {breach_label}"
    body = (
        "This is an automated SLA breach alert.\n\n"
        f"Grievance       : {grievance.tracking_code}\n"
        f"Current Status  : {status_label}\n"
        f"Breach Type     : {breach_label}\n"
        f"Breached At     : {breached_at}\n"
        f"Response Due    : {response_due}\n"
        f"Resolution Due  : {resolution_due}\n\n"
        "Please take immediate action to resolve this grievance.\n\n"
        f"{_PORTAL_SIGNATURE} (automated)"
    )

    # FUTURE-IDEMPOTENCY: check/insert token ("sla_breach_alert", sla.pk) before send
    messages = tuple(
        (subject, body, settings.DEFAULT_FROM_EMAIL, [recipient])
        for recipient in recipients
    )
    try:
        send_mass_mail(messages, fail_silently=False)
        logger.info(
            "sla_breach_alert_email: sent to %d recipient(s) for SLA %s",
            len(recipients),
            sla.sla_code,
        )
    except Exception:
        logger.exception(
            "sla_breach_alert_email: failed to send for SLA %s",
            sla.sla_code,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_send(*, subject: str, body: str, recipient: str, context: str) -> None:
    """Send a single email, logging any failure without re-raising."""
    try:
        send_mail(
            subject=subject,
            message=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient],
            fail_silently=False,
        )
        logger.info("email sent: context=%s recipient=%s", context, recipient)
    except Exception:
        logger.exception(
            "email send failed: context=%s recipient=%s", context, recipient
        )

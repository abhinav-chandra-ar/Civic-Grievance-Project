"""Unit tests for apps.integrations.emails helper functions.

Strategy
--------
Event 1 & 2 helpers are tested with plain MagicMock objects — no DB
needed.  Event 3 (SLA breach) queries the User table to find recipients,
so those tests are marked django_db.

All tests run against Django's in-memory email backend so no real SMTP
connection is made.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from unittest.mock import MagicMock, patch

import pytest
from django.core import mail
from django.test import override_settings

from apps.integrations.emails import (
    send_grievance_status_changed_email,
    send_grievance_submitted_email,
    send_sla_breach_alert_email,
)

# ---------------------------------------------------------------------------
# Shared settings override — applied per test via decorator
# ---------------------------------------------------------------------------
LOCMEM_EMAIL = {
    "EMAIL_BACKEND": "django.core.mail.backends.locmem.EmailBackend",
    "DEFAULT_FROM_EMAIL": "portal@test.example",
}


# ---------------------------------------------------------------------------
# Factories for mock domain objects (no DB)
# ---------------------------------------------------------------------------

def _submitter(email="citizen@example.com", first_name="Alice"):
    u = MagicMock()
    u.email = email
    u.first_name = first_name
    return u


def _grievance(
    tracking_code="GRV-2026-000001",
    raw_text="Pothole on main road.",
    status="submitted",
    submitter=None,
):
    g = MagicMock()
    g.tracking_code = tracking_code
    g.raw_text = raw_text
    g.status = status
    g.submitted_at = datetime(2026, 5, 23, 10, 30, tzinfo=dt_timezone.utc)
    g.submitter = submitter or _submitter()
    return g


def _workflow_event(
    previous_status="submitted",
    new_status="in_progress",
    remarks="Officer assigned.",
    grievance=None,
):
    e = MagicMock()
    e.pk = 42
    e.previous_status = previous_status
    e.new_status = new_status
    e.remarks = remarks
    e.occurred_at = datetime(2026, 5, 23, 11, 0, tzinfo=dt_timezone.utc)
    e.grievance = grievance or _grievance()
    return e


def _sla(
    sla_code="SLA-2026-000001",
    breach_type="response",
    grievance=None,
):
    s = MagicMock()
    s.sla_code = sla_code
    s.breach_type = breach_type
    s.breached_at = datetime(2026, 5, 23, 12, 0, tzinfo=dt_timezone.utc)
    s.response_due_at = datetime(2026, 5, 22, 9, 0, tzinfo=dt_timezone.utc)
    s.resolution_due_at = datetime(2026, 5, 24, 9, 0, tzinfo=dt_timezone.utc)
    s.grievance = grievance or _grievance(status="in_progress")
    return s


# ---------------------------------------------------------------------------
# Event 1 — grievance submitted
# ---------------------------------------------------------------------------

class TestSendGrievanceSubmittedEmail:
    def setup_method(self):
        mail.outbox = []

    @override_settings(**LOCMEM_EMAIL)
    def test_sends_to_submitter_email(self):
        send_grievance_submitted_email(_grievance())
        assert len(mail.outbox) == 1
        assert mail.outbox[0].to == ["citizen@example.com"]

    @override_settings(**LOCMEM_EMAIL)
    def test_subject_contains_tracking_code(self):
        send_grievance_submitted_email(_grievance())
        assert "GRV-2026-000001" in mail.outbox[0].subject

    @override_settings(**LOCMEM_EMAIL)
    def test_body_greets_by_first_name(self):
        send_grievance_submitted_email(_grievance())
        assert "Dear Alice," in mail.outbox[0].body

    @override_settings(**LOCMEM_EMAIL)
    def test_body_falls_back_to_dear_citizen_when_no_first_name(self):
        g = _grievance(submitter=_submitter(first_name=""))
        send_grievance_submitted_email(g)
        assert "Dear Citizen," in mail.outbox[0].body

    @override_settings(**LOCMEM_EMAIL)
    def test_body_contains_raw_text_excerpt(self):
        send_grievance_submitted_email(_grievance(raw_text="Broken streetlight near park."))
        assert "Broken streetlight" in mail.outbox[0].body

    @override_settings(**LOCMEM_EMAIL)
    def test_long_raw_text_is_truncated_with_ellipsis(self):
        long_text = "x" * 250
        send_grievance_submitted_email(_grievance(raw_text=long_text))
        assert "…" in mail.outbox[0].body

    @override_settings(**LOCMEM_EMAIL)
    def test_no_email_when_submitter_email_is_blank(self):
        g = _grievance(submitter=_submitter(email=""))
        send_grievance_submitted_email(g)
        assert len(mail.outbox) == 0

    @override_settings(**LOCMEM_EMAIL)
    def test_no_email_when_submitter_is_none(self):
        g = _grievance()
        g.submitter = None
        send_grievance_submitted_email(g)
        assert len(mail.outbox) == 0

    @override_settings(**LOCMEM_EMAIL)
    def test_smtp_failure_is_logged_not_raised(self, caplog):
        with patch("apps.integrations.emails.send_mail", side_effect=OSError("SMTP down")):
            with caplog.at_level(logging.ERROR, logger="apps.integrations.emails"):
                send_grievance_submitted_email(_grievance())  # must not raise
        assert "SMTP down" in caplog.text


# ---------------------------------------------------------------------------
# Event 2 — grievance status changed
# ---------------------------------------------------------------------------

class TestSendGrievanceStatusChangedEmail:
    def setup_method(self):
        mail.outbox = []

    @override_settings(**LOCMEM_EMAIL)
    def test_sends_to_submitter_email(self):
        send_grievance_status_changed_email(_workflow_event())
        assert len(mail.outbox) == 1
        assert mail.outbox[0].to == ["citizen@example.com"]

    @override_settings(**LOCMEM_EMAIL)
    def test_subject_contains_tracking_code_and_new_status(self):
        send_grievance_status_changed_email(_workflow_event(new_status="in_progress"))
        subject = mail.outbox[0].subject
        assert "GRV-2026-000001" in subject
        assert "In Progress" in subject

    @override_settings(**LOCMEM_EMAIL)
    def test_body_contains_old_and_new_status(self):
        e = _workflow_event(previous_status="submitted", new_status="in_progress")
        send_grievance_status_changed_email(e)
        body = mail.outbox[0].body
        assert "Submitted" in body
        assert "In Progress" in body

    @override_settings(**LOCMEM_EMAIL)
    def test_body_includes_remarks_when_present(self):
        e = _workflow_event(remarks="Assigned to ward officer.")
        send_grievance_status_changed_email(e)
        assert "Assigned to ward officer." in mail.outbox[0].body

    @override_settings(**LOCMEM_EMAIL)
    def test_body_omits_remarks_line_when_blank(self):
        e = _workflow_event(remarks="")
        send_grievance_status_changed_email(e)
        assert "Officer Remarks" not in mail.outbox[0].body

    @override_settings(**LOCMEM_EMAIL)
    def test_no_email_when_submitter_email_is_blank(self):
        g = _grievance(submitter=_submitter(email=""))
        e = _workflow_event(grievance=g)
        send_grievance_status_changed_email(e)
        assert len(mail.outbox) == 0

    @override_settings(**LOCMEM_EMAIL)
    def test_smtp_failure_is_logged_not_raised(self, caplog):
        with patch("apps.integrations.emails.send_mail", side_effect=OSError("SMTP down")):
            with caplog.at_level(logging.ERROR, logger="apps.integrations.emails"):
                send_grievance_status_changed_email(_workflow_event())  # must not raise
        assert "SMTP down" in caplog.text


# ---------------------------------------------------------------------------
# Event 3 — SLA breached  (requires DB for User query)
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestSendSlaBreachAlertEmail:
    def setup_method(self):
        mail.outbox = []

    @override_settings(**LOCMEM_EMAIL)
    def test_sends_to_all_admin_and_operator_users(self, django_user_model):
        django_user_model.objects.create_user(
            username="admin1", email="admin@example.com",
            password="x", role="municipal_admin",
        )
        django_user_model.objects.create_user(
            username="sysop1", email="sysop@example.com",
            password="x", role="system_operator",
        )
        send_sla_breach_alert_email(_sla())
        recipients = {msg.to[0] for msg in mail.outbox}
        assert "admin@example.com" in recipients
        assert "sysop@example.com" in recipients

    @override_settings(**LOCMEM_EMAIL)
    def test_citizen_is_not_alerted(self, django_user_model):
        django_user_model.objects.create_user(
            username="cit1", email="citizen@example.com",
            password="x", role="citizen",
        )
        send_sla_breach_alert_email(_sla())
        assert len(mail.outbox) == 0

    @override_settings(**LOCMEM_EMAIL)
    def test_no_email_when_no_admin_users_exist(self):
        send_sla_breach_alert_email(_sla())
        assert len(mail.outbox) == 0

    @override_settings(**LOCMEM_EMAIL)
    def test_inactive_admin_is_excluded(self, django_user_model):
        django_user_model.objects.create_user(
            username="inactive_admin", email="inactive@example.com",
            password="x", role="super_admin", is_active=False,
        )
        send_sla_breach_alert_email(_sla())
        assert len(mail.outbox) == 0

    @override_settings(**LOCMEM_EMAIL)
    def test_subject_contains_tracking_code_and_breach_type(self, django_user_model):
        django_user_model.objects.create_user(
            username="su1", email="su@example.com",
            password="x", role="super_admin",
        )
        send_sla_breach_alert_email(_sla(breach_type="response"))
        assert "GRV-2026-000001" in mail.outbox[0].subject
        assert "Response" in mail.outbox[0].subject

    @override_settings(**LOCMEM_EMAIL)
    def test_body_contains_deadlines(self, django_user_model):
        django_user_model.objects.create_user(
            username="su2", email="su2@example.com",
            password="x", role="super_admin",
        )
        send_sla_breach_alert_email(_sla())
        body = mail.outbox[0].body
        assert "Response Due" in body
        assert "Resolution Due" in body

    @override_settings(**LOCMEM_EMAIL)
    def test_smtp_failure_is_logged_not_raised(self, django_user_model, caplog):
        django_user_model.objects.create_user(
            username="su3", email="su3@example.com",
            password="x", role="super_admin",
        )
        with patch("apps.integrations.emails.send_mass_mail", side_effect=OSError("SMTP down")):
            with caplog.at_level(logging.ERROR, logger="apps.integrations.emails"):
                send_sla_breach_alert_email(_sla())  # must not raise
        assert "SMTP down" in caplog.text

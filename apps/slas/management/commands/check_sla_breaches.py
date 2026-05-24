"""Management command: check_sla_breaches

Scans active SLAs whose deadlines have passed, marks them as breached,
fires the ``sla_breached`` signal (which triggers the breach-alert email
to all admin/operator users), and optionally records an escalation workflow
event on the linked grievance.

Idempotency
-----------
The underlying ``sla_list_requiring_breach_check()`` selector filters on
``is_breached=False`` and ``sla_status=ACTIVE``, so re-running the command
never processes an already-breached SLA.  It is safe to run repeatedly.

Usage
-----
    python manage.py check_sla_breaches
    python manage.py check_sla_breaches --dry-run
    python manage.py check_sla_breaches --escalate
    python manage.py check_sla_breaches --escalate --dry-run
"""
from __future__ import annotations

import logging

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Scan active SLAs whose deadlines have passed, mark them breached, "
        "send alert emails, and (with --escalate) record escalation workflow events."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Report which SLAs would be breached without writing to the database.",
        )
        parser.add_argument(
            "--escalate",
            action="store_true",
            default=False,
            help=(
                "For each newly breached SLA, record an ESCALATION workflow event "
                "on the linked grievance (only when the grievance is not already "
                "in a terminal state: resolved / rejected / closed)."
            ),
        )

    def handle(self, *args, **options) -> None:
        dry_run: bool = options["dry_run"]
        escalate: bool = options["escalate"]
        now = timezone.now()

        from apps.slas.selectors import sla_list_requiring_breach_check
        from apps.slas.services import refresh_sla_deadline_status

        candidates = list(sla_list_requiring_breach_check(now=now))
        total = len(candidates)

        if total == 0:
            self.stdout.write(self.style.SUCCESS("No SLA breaches to process."))
            return

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f"[DRY-RUN] {total} SLA(s) would be marked breached:"
                )
            )
            for sla in candidates:
                self.stdout.write(
                    f"  {sla.sla_code}  grievance={sla.grievance.tracking_code}"
                    f"  response_due={sla.response_due_at.strftime('%Y-%m-%d %H:%M')}"
                    f"  resolution_due={sla.resolution_due_at.strftime('%Y-%m-%d %H:%M')}"
                )
            return

        breached_count = 0
        escalated_count = 0

        for sla in candidates:
            try:
                with transaction.atomic():
                    # refresh_sla_deadline_status() → mark_sla_breached() → sla_breached
                    # signal → handle_sla_breached() → send_sla_breach_alert_email()
                    # All of this is already wired and fires inside mark_sla_breached().
                    refresh_sla_deadline_status(sla=sla, now=now)
                    breached_count += 1
                    self.stdout.write(
                        f"  Breached: {sla.sla_code}  "
                        f"grievance={sla.grievance.tracking_code}  "
                        f"type={sla.breach_type}"
                    )

                    if escalate:
                        escalated_count += _escalate_linked_grievance(
                            sla=sla, now=now
                        )

            except Exception:  # noqa: BLE001
                logger.exception(
                    "check_sla_breaches: failed to process SLA %s",
                    sla.sla_code,
                )
                self.stderr.write(
                    self.style.ERROR(f"  ERROR processing {sla.sla_code} — see logs.")
                )

        summary = f"{breached_count}/{total} SLA(s) marked breached."
        if escalate:
            summary += f"  {escalated_count} grievance(s) escalated."
        self.stdout.write(self.style.SUCCESS(summary))


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({"resolved", "rejected", "closed"})


def _escalate_linked_grievance(*, sla, now) -> int:
    """Record an escalation workflow event for the SLA's grievance.

    Returns 1 if an escalation was recorded, 0 otherwise (terminal state).
    """
    from apps.workflows.services import escalate_grievance_from_system  # noqa: PLC0415

    grievance = sla.grievance
    if grievance.status in _TERMINAL_STATUSES:
        return 0

    reason = (
        f"SLA breached at {now.strftime('%Y-%m-%d %H:%M')} UTC "
        f"({sla.sla_code}, type={sla.breach_type})."
    )
    escalate_grievance_from_system(
        grievance=grievance,
        transition_reason=reason,
        escalation_metadata={
            "source": "sla_breach_monitoring",
            "sla_code": sla.sla_code,
            "breach_type": sla.breach_type,
            "breached_at": now.isoformat(),
        },
    )
    return 1

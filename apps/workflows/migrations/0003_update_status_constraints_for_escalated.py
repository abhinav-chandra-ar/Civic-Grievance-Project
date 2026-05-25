# Generated manually on 2026-05-25 — workflow redesign: expand status allow-lists
# on WorkflowEvent.previous_status and WorkflowEvent.new_status to include the
# two new GrievanceStatus values: "escalated" and "duplicate_flagged".

from django.db import migrations, models

_EXPANDED_STATUS_VALUES = [
    "submitted",
    "enrichment_pending",
    "triaged",
    "assigned",
    "in_progress",
    "resolved",
    "rejected",
    "closed",
    "escalated",
    "duplicate_flagged",
]

_EXPANDED_STATUS_CHOICES = [
    ("submitted", "Submitted"),
    ("enrichment_pending", "Enrichment pending"),
    ("triaged", "Triaged"),
    ("assigned", "Assigned"),
    ("in_progress", "In progress"),
    ("resolved", "Resolved"),
    ("rejected", "Rejected"),
    ("closed", "Closed"),
    ("escalated", "Escalated"),
    ("duplicate_flagged", "Duplicate flagged"),
]


class Migration(migrations.Migration):

    dependencies = [
        ("grievances", "0004_add_escalated_duplicate_flagged_status"),
        ("workflows", "0002_add_return_transition"),
    ]

    operations = [
        # ── Drop old 8-value CheckConstraints ──────────────────────────────
        migrations.RemoveConstraint(
            model_name="workflowevent",
            name="wf_previous_status_valid",
        ),
        migrations.RemoveConstraint(
            model_name="workflowevent",
            name="wf_new_status_valid",
        ),
        # ── Widen choices lists so Django validation accepts new values ────
        migrations.AlterField(
            model_name="workflowevent",
            name="previous_status",
            field=models.CharField(
                max_length=32,
                choices=_EXPANDED_STATUS_CHOICES,
            ),
        ),
        migrations.AlterField(
            model_name="workflowevent",
            name="new_status",
            field=models.CharField(
                max_length=32,
                choices=_EXPANDED_STATUS_CHOICES,
            ),
        ),
        # ── Re-add CheckConstraints with 10-value allow-lists ─────────────
        migrations.AddConstraint(
            model_name="workflowevent",
            constraint=models.CheckConstraint(
                condition=models.Q(previous_status__in=_EXPANDED_STATUS_VALUES),
                name="wf_previous_status_valid",
            ),
        ),
        migrations.AddConstraint(
            model_name="workflowevent",
            constraint=models.CheckConstraint(
                condition=models.Q(new_status__in=_EXPANDED_STATUS_VALUES),
                name="wf_new_status_valid",
            ),
        ),
    ]

# Generated manually on 2026-05-25 — workflow redesign correction.
# "escalated" is not a lifecycle state; remove it from the status allow-list.
# The only enrichment-driven state retained is "duplicate_flagged".

from django.db import migrations, models

# Authoritative 9-value lifecycle state list (no "escalated")
_STATUS_VALUES = [
    "submitted",
    "enrichment_pending",
    "triaged",
    "assigned",
    "in_progress",
    "resolved",
    "rejected",
    "closed",
    "duplicate_flagged",
]

_STATUS_CHOICES = [
    ("submitted", "Submitted"),
    ("enrichment_pending", "Enrichment pending"),
    ("triaged", "Triaged"),
    ("assigned", "Assigned"),
    ("in_progress", "In progress"),
    ("resolved", "Resolved"),
    ("rejected", "Rejected"),
    ("closed", "Closed"),
    ("duplicate_flagged", "Duplicate flagged"),
]


class Migration(migrations.Migration):

    dependencies = [
        ("grievances", "0004_add_escalated_duplicate_flagged_status"),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name="grievance",
            name="grievance_status_valid",
        ),
        migrations.AlterField(
            model_name="grievance",
            name="status",
            field=models.CharField(
                choices=_STATUS_CHOICES,
                db_index=True,
                default="submitted",
                max_length=32,
            ),
        ),
        migrations.AddConstraint(
            model_name="grievance",
            constraint=models.CheckConstraint(
                condition=models.Q(status__in=_STATUS_VALUES),
                name="grievance_status_valid",
            ),
        ),
    ]

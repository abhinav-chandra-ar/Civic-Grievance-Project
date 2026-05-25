# Generated manually on 2026-05-25 — workflow redesign: add ESCALATED + DUPLICATE_FLAGGED.

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
        ("grievances", "0003_grievance_grv_submitter_submitted_idx_and_more"),
    ]

    operations = [
        # 1. Drop the existing DB-level CHECK constraint (hardwired to 8 values).
        migrations.RemoveConstraint(
            model_name="grievance",
            name="grievance_status_valid",
        ),
        # 2. Widen the choices list on the field so Django admin / DRF serializers
        #    accept the two new values.
        migrations.AlterField(
            model_name="grievance",
            name="status",
            field=models.CharField(
                choices=_EXPANDED_STATUS_CHOICES,
                db_index=True,
                default="submitted",
                max_length=32,
            ),
        ),
        # 3. Re-add the CHECK constraint with the 10-value allow-list.
        migrations.AddConstraint(
            model_name="grievance",
            constraint=models.CheckConstraint(
                condition=models.Q(status__in=_EXPANDED_STATUS_VALUES),
                name="grievance_status_valid",
            ),
        ),
    ]

# Generated manually on 2026-05-25 — workflow redesign correction.
# Remove "escalated" from WorkflowEvent status allow-lists.
# Escalation is an alert/metadata flag, not a lifecycle state transition.

from django.db import migrations, models

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
        ("grievances", "0005_remove_escalated_from_status_constraint"),
        ("workflows", "0003_update_status_constraints_for_escalated"),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name="workflowevent",
            name="wf_previous_status_valid",
        ),
        migrations.RemoveConstraint(
            model_name="workflowevent",
            name="wf_new_status_valid",
        ),
        migrations.AlterField(
            model_name="workflowevent",
            name="previous_status",
            field=models.CharField(max_length=32, choices=_STATUS_CHOICES),
        ),
        migrations.AlterField(
            model_name="workflowevent",
            name="new_status",
            field=models.CharField(max_length=32, choices=_STATUS_CHOICES),
        ),
        migrations.AddConstraint(
            model_name="workflowevent",
            constraint=models.CheckConstraint(
                condition=models.Q(previous_status__in=_STATUS_VALUES),
                name="wf_previous_status_valid",
            ),
        ),
        migrations.AddConstraint(
            model_name="workflowevent",
            constraint=models.CheckConstraint(
                condition=models.Q(new_status__in=_STATUS_VALUES),
                name="wf_new_status_valid",
            ),
        ),
    ]

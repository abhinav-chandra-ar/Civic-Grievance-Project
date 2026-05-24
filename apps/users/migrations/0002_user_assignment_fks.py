"""Add assigned_ward and assigned_department FK fields to the User model.

These fields are required for proper ward-officer and department-officer
visibility scoping in grievance_list_visible_to_user().  Both are nullable so
that existing officers are not forced into an assignment on deployment.
"""
from __future__ import annotations

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("departments", "0002_department_dept_categories_gin_idx"),
        ("users", "0001_initial"),
        ("wards", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="assigned_ward",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="assigned_officers",
                to="wards.ward",
                help_text="Ward this officer is assigned to (ward_officer role only).",
            ),
        ),
        migrations.AddField(
            model_name="user",
            name="assigned_department",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="assigned_officers",
                to="departments.department",
                help_text="Department this officer is assigned to (department_officer role only).",
            ),
        ),
    ]

"""App configuration for the departments app."""
from __future__ import annotations

from django.apps import AppConfig


class DepartmentsConfig(AppConfig):
    name = "apps.departments"
    label = "departments"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Departments"

    def ready(self) -> None:
        from . import signals  # noqa: F401

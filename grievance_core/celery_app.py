"""Celery application for grievance-core.

Task routing, queue priorities, and the Celery Beat schedule land in
Module 17 (Queue workers). This file only sets up the Celery app, points it
at Django settings, and autodiscovers @shared_task functions from all apps.
"""
from __future__ import annotations

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "grievance_core.settings.dev")

celery_app = Celery("grievance_core")

# Load every setting prefixed with CELERY_ from Django settings
celery_app.config_from_object("django.conf:settings", namespace="CELERY")

# Discover tasks.py in every installed app
celery_app.autodiscover_tasks()


@celery_app.task(bind=True, name="grievance_core.debug.ping")
def ping(self) -> str:  # type: ignore[no-untyped-def]
    """Lightweight task used by the readiness probe to confirm broker connectivity."""
    return f"pong from {self.request.id}"

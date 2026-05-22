"""grievance-core — civic grievance intelligence platform, transactional core.

Importing the Celery app here ensures @shared_task decorators are picked up
whenever Django starts.
"""
from __future__ import annotations

from .celery_app import celery_app

__all__ = ("celery_app",)

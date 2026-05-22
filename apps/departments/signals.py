"""Domain signals emitted by department services."""
from __future__ import annotations

from django.dispatch import Signal

department_created = Signal()
department_updated = Signal()

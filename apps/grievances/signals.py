"""Domain signals emitted by grievance services."""
from __future__ import annotations

from django.dispatch import Signal

grievance_submitted = Signal()
grievance_updated = Signal()
grievance_status_changed = Signal()

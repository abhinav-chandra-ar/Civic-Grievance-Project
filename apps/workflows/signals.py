"""Domain signals emitted by workflow services."""
from __future__ import annotations

from django.dispatch import Signal

workflow_event_recorded = Signal()

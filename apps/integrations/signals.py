"""Signals emitted by integration orchestration services."""
from __future__ import annotations

from django.dispatch import Signal

integration_call_completed = Signal()

"""Domain signals emitted by audit services."""
from __future__ import annotations

from django.dispatch import Signal

audit_log_recorded = Signal()

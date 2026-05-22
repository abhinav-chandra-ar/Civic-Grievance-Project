"""Domain signals emitted by SLA services."""
from __future__ import annotations

from django.dispatch import Signal

sla_created = Signal()
sla_updated = Signal()
sla_breached = Signal()

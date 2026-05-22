"""Domain signals emitted by attachment services."""
from __future__ import annotations

from django.dispatch import Signal

attachment_registered = Signal()
attachment_updated = Signal()

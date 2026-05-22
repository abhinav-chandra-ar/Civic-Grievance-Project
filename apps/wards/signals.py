"""Domain signals emitted by ward services."""
from __future__ import annotations

from django.dispatch import Signal

ward_created = Signal()
ward_updated = Signal()

"""Domain signals emitted by landmark services."""
from __future__ import annotations

from django.dispatch import Signal

landmark_created = Signal()
landmark_updated = Signal()

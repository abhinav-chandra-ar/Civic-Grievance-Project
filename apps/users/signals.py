"""Domain signals emitted by user services."""
from __future__ import annotations

from django.dispatch import Signal

user_registered = Signal()
user_profile_updated = Signal()

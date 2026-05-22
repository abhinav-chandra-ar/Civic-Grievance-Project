"""Read-side queries for users."""
from __future__ import annotations

from django.db.models import QuerySet

from .models import User


def user_list(*, active_only: bool = True) -> QuerySet[User]:
    """Return users ordered for stable API and admin-adjacent reads."""
    users = User.objects.order_by("id")
    if active_only:
        return users.filter(is_active=True)
    return users


def user_get_by_id(*, user_id: int) -> User:
    """Return a user by primary key."""
    return User.objects.get(pk=user_id)


def user_get_by_phone_number(*, phone_number: str) -> User | None:
    """Return an active user matching an already-normalized contact number."""
    return user_list().filter(phone_number=phone_number).first()

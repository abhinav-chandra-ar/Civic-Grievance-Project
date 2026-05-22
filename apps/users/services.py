"""Write-side application services for users."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction

from .models import PreferredLanguage, User, UserRole
from .signals import user_profile_updated, user_registered

PROFILE_WRITE_FIELDS = frozenset(
    {
        "first_name",
        "last_name",
        "email",
        "phone_number",
        "preferred_language",
        "additional_translations",
    }
)


def normalize_phone_number(phone_number: str) -> str:
    """Trim benign user input around an E.164 number."""
    return phone_number.strip()


def _prepare_profile_values(values: Mapping[str, Any]) -> dict[str, Any]:
    prepared = {field: values[field] for field in PROFILE_WRITE_FIELDS if field in values}
    if "phone_number" in prepared:
        prepared["phone_number"] = normalize_phone_number(prepared["phone_number"])
    return prepared


@transaction.atomic
def create_user(
    *,
    username: str,
    password: str,
    role: str = UserRole.CITIZEN,
    **profile_values: Any,
) -> User:
    """Create and validate an application user through the custom model."""
    prepared = _prepare_profile_values(profile_values)
    user = User(username=username, role=role, **prepared)
    validate_password(password, user)
    user.set_password(password)
    user.full_clean()
    user.save()
    user_registered.send(sender=User, user=user)
    return user


@transaction.atomic
def update_user_profile(*, user: User, values: Mapping[str, Any]) -> User:
    """Update mutable profile values while leaving identity and role untouched."""
    prepared = _prepare_profile_values(values)
    unknown_fields = set(values) - PROFILE_WRITE_FIELDS
    if unknown_fields:
        raise ValidationError(f"Unsupported profile fields: {', '.join(sorted(unknown_fields))}")

    for field, value in prepared.items():
        setattr(user, field, value)

    if prepared:
        user.full_clean()
        user.save(update_fields=prepared)
        user_profile_updated.send(sender=User, user=user, updated_fields=frozenset(prepared))
    return user


def set_user_role(*, user: User, role: str | UserRole) -> User:
    """Change a closed user role from trusted administrative call sites."""
    user.role = str(role)
    user.full_clean()
    user.save(update_fields=["role"])
    return user


__all__ = [
    "PreferredLanguage",
    "UserRole",
    "create_user",
    "normalize_phone_number",
    "set_user_role",
    "update_user_profile",
]

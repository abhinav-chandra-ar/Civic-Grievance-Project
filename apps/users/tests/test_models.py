"""Model tests for the custom user."""
from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError

from apps.users.models import PreferredLanguage, User, UserRole


def test_user_role_choices_cover_foundation_rbac_roles() -> None:
    assert set(UserRole.values) == {
        "citizen",
        "ward_officer",
        "department_officer",
        "municipal_admin",
        "super_admin",
        "field_verifier",
        "system_operator",
    }


def test_phone_number_must_be_e164() -> None:
    user = User(username="citizen", phone_number="9876543210")

    with pytest.raises(ValidationError) as exc_info:
        user.full_clean()

    assert "phone_number" in exc_info.value.message_dict


def test_phone_number_accepts_e164_and_profile_translations() -> None:
    user = User(
        username="citizen",
        phone_number="+919876543210",
        preferred_language=PreferredLanguage.MALAYALAM,
        additional_translations={"ml": {"display_name": "Asha"}},
    )

    user.full_clean()


def test_role_helpers_expose_role_access_hooks() -> None:
    user = User(username="admin", role=UserRole.MUNICIPAL_ADMIN)

    assert user.is_municipal_admin
    assert user.is_platform_admin
    assert not user.is_citizen

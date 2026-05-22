"""Serializer and service tests for user profile writes."""
from __future__ import annotations

import pytest

from apps.users.models import UserRole
from apps.users.serializers import UserCreateSerializer, UserProfileUpdateSerializer

pytestmark = pytest.mark.django_db


def test_create_serializer_hashes_password_and_normalizes_contact() -> None:
    serializer = UserCreateSerializer(
        data={
            "username": "citizen",
            "password": "A-long-password-2026!",
            "phone_number": " +919876543210 ",
            "role": UserRole.CITIZEN,
        }
    )

    assert serializer.is_valid(), serializer.errors
    user = serializer.save()
    assert user.check_password("A-long-password-2026!")
    assert user.phone_number == "+919876543210"


def test_profile_serializer_rejects_translation_arrays(django_user_model) -> None:
    user = django_user_model.objects.create_user(username="citizen", password="password")
    serializer = UserProfileUpdateSerializer(
        user,
        data={"additional_translations": ["ml"]},
        partial=True,
    )

    assert not serializer.is_valid()
    assert "additional_translations" in serializer.errors

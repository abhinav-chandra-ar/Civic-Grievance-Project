"""DRF serializers for users."""
from __future__ import annotations

from typing import Any

from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import User, UserRole
from .services import create_user, update_user_profile


class UserSerializer(serializers.ModelSerializer[User]):
    """Read-only public representation for application user data."""

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "first_name",
            "last_name",
            "email",
            "phone_number",
            "role",
            "preferred_language",
            "additional_translations",
            "is_active",
            "date_joined",
        )
        read_only_fields = fields


class UserCreateSerializer(serializers.ModelSerializer[User]):
    """Create users without exposing password persistence details."""

    password = serializers.CharField(write_only=True, trim_whitespace=False)

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "password",
            "first_name",
            "last_name",
            "email",
            "phone_number",
            "role",
            "preferred_language",
            "additional_translations",
        )
        read_only_fields = ("id",)

    def validate_password(self, value: str) -> str:
        validate_password(value)
        return value

    def create(self, validated_data: dict[str, Any]) -> User:
        password = validated_data.pop("password")
        return create_user(password=password, **validated_data)


class UserProfileUpdateSerializer(serializers.ModelSerializer[User]):
    """Mutable self-service profile fields."""

    class Meta:
        model = User
        fields = (
            "first_name",
            "last_name",
            "email",
            "phone_number",
            "preferred_language",
            "additional_translations",
        )

    def update(self, instance: User, validated_data: dict[str, Any]) -> User:
        return update_user_profile(user=instance, values=validated_data)

    def validate_additional_translations(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise serializers.ValidationError("Expected an object keyed by language code.")
        return value


class CitizenRegisterSerializer(serializers.ModelSerializer[User]):
    """Public self-registration for citizens.

    Role is always CITIZEN regardless of payload content — the field is not
    declared in Meta.fields (DRF drops undeclared keys before validation) and
    create() passes role=UserRole.CITIZEN explicitly to the service layer.
    """

    password = serializers.CharField(write_only=True, trim_whitespace=False)

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "password",
            "first_name",
            "last_name",
            "email",
            "phone_number",
            "preferred_language",
        )
        read_only_fields = ("id",)

    def validate_password(self, value: str) -> str:
        validate_password(value)
        return value

    def create(self, validated_data: dict[str, Any]) -> User:
        password = validated_data.pop("password")
        return create_user(password=password, role=UserRole.CITIZEN, **validated_data)

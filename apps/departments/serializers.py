"""DRF serializers for departments."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .models import Department
from .services import create_department, update_department


class DepartmentSerializer(serializers.ModelSerializer[Department]):
    """Read representation for department routing metadata."""

    class Meta:
        model = Department
        fields = (
            "id",
            "code",
            "name",
            "translated_names",
            "handled_categories",
            "is_active",
            "escalation_metadata",
            "sla_metadata",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class DepartmentWriteSerializer(serializers.ModelSerializer[Department]):
    """Create and update department metadata through service functions."""

    class Meta:
        model = Department
        fields = (
            "code",
            "name",
            "translated_names",
            "handled_categories",
            "is_active",
            "escalation_metadata",
            "sla_metadata",
        )

    def create(self, validated_data: dict[str, Any]) -> Department:
        return create_department(**validated_data)

    def update(self, instance: Department, validated_data: dict[str, Any]) -> Department:
        return update_department(department=instance, values=validated_data)

"""User model for identity, contact details, and application role hooks."""
from __future__ import annotations

from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models

E164_PHONE_NUMBER_VALIDATOR = RegexValidator(
    regex=r"^\+[1-9]\d{1,14}$",
    message="Enter a phone number in E.164 format, such as +919876543210.",
)


def validate_translation_mapping(value: object) -> None:
    """Keep translated profile content keyed by language code."""
    if not isinstance(value, dict):
        raise ValidationError("Additional translations must be a JSON object.")


class UserRole(models.TextChoices):
    CITIZEN = "citizen", "Citizen"
    WARD_OFFICER = "ward_officer", "Ward officer"
    DEPARTMENT_OFFICER = "department_officer", "Department officer"
    MUNICIPAL_ADMIN = "municipal_admin", "Municipal admin"
    SUPER_ADMIN = "super_admin", "Super admin"
    FIELD_VERIFIER = "field_verifier", "Field verifier"
    SYSTEM_OPERATOR = "system_operator", "System operator"


class PreferredLanguage(models.TextChoices):
    ENGLISH = "en", "English"
    MALAYALAM = "ml", "Malayalam"


class User(AbstractUser):
    """Custom user model shared by all grievance-core applications."""

    role = models.CharField(
        max_length=32,
        choices=UserRole.choices,
        default=UserRole.CITIZEN,
        db_index=True,
    )
    phone_number = models.CharField(
        max_length=16,
        blank=True,
        db_index=True,
        validators=[E164_PHONE_NUMBER_VALIDATOR],
        help_text="Optional contact number in E.164 format.",
    )
    preferred_language = models.CharField(
        max_length=2,
        choices=PreferredLanguage.choices,
        default=PreferredLanguage.ENGLISH,
    )
    additional_translations = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_translation_mapping],
        help_text="Additional translated user profile values keyed by language.",
    )

    class Meta:
        db_table = "users_user"
        verbose_name = "User"
        verbose_name_plural = "Users"
        indexes = [
            models.Index(fields=["role", "is_active"], name="users_role_active_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(role__in=UserRole.values),
                name="users_role_valid",
            ),
            models.CheckConstraint(
                condition=models.Q(preferred_language__in=PreferredLanguage.values),
                name="users_preferred_language_valid",
            ),
        ]

    def has_role(self, *roles: str | UserRole) -> bool:
        """Return whether the user holds one of the supplied closed roles."""
        return self.role in {str(role) for role in roles}

    @property
    def is_citizen(self) -> bool:
        return self.has_role(UserRole.CITIZEN)

    @property
    def is_ward_officer(self) -> bool:
        return self.has_role(UserRole.WARD_OFFICER)

    @property
    def is_department_officer(self) -> bool:
        return self.has_role(UserRole.DEPARTMENT_OFFICER)

    @property
    def is_municipal_admin(self) -> bool:
        return self.has_role(UserRole.MUNICIPAL_ADMIN)

    @property
    def is_platform_admin(self) -> bool:
        return self.has_role(UserRole.MUNICIPAL_ADMIN, UserRole.SUPER_ADMIN)

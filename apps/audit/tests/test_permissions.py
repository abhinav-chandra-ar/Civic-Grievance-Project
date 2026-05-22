"""Permission tests for audit log reads."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.audit.permissions import IsAuditReaderRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_audit_reader_permission_accepts_super_admin_safe_method() -> None:
    request = APIRequestFactory().get("/audit/")
    request.user = UserStub(role="super_admin")

    assert IsAuditReaderRole().has_permission(request, view=None)


def test_audit_reader_permission_rejects_post_even_for_admin() -> None:
    request = APIRequestFactory().post("/audit/")
    request.user = UserStub(role="super_admin")

    assert not IsAuditReaderRole().has_permission(request, view=None)


def test_audit_reader_permission_rejects_citizen() -> None:
    request = APIRequestFactory().get("/audit/")
    request.user = UserStub(role="citizen")

    assert not IsAuditReaderRole().has_permission(request, view=None)

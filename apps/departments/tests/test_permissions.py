"""Permission tests for department endpoints."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.departments.permissions import IsDepartmentManagerRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_department_manager_permission_accepts_municipal_admin() -> None:
    request = APIRequestFactory().post("/departments/")
    request.user = UserStub(role="municipal_admin")

    assert IsDepartmentManagerRole().has_permission(request, view=None)


def test_department_manager_permission_accepts_system_operator() -> None:
    request = APIRequestFactory().post("/departments/")
    request.user = UserStub(role="system_operator")

    assert IsDepartmentManagerRole().has_permission(request, view=None)


def test_department_manager_permission_rejects_super_admin() -> None:
    # Department config is city operations; super_admin is platform governance only.
    request = APIRequestFactory().post("/departments/")
    request.user = UserStub(role="super_admin")

    assert not IsDepartmentManagerRole().has_permission(request, view=None)


def test_department_manager_permission_rejects_citizen() -> None:
    request = APIRequestFactory().post("/departments/")
    request.user = UserStub(role="citizen")

    assert not IsDepartmentManagerRole().has_permission(request, view=None)

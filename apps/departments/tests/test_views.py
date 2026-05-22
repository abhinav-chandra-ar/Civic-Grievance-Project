"""View tests for departments."""
from __future__ import annotations

import pytest
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.departments.models import Department
from apps.departments.views import DepartmentViewSet

pytestmark = pytest.mark.django_db


def test_department_list_excludes_inactive_departments(django_user_model) -> None:
    user = django_user_model.objects.create_user(username="viewer", password="password")
    Department.objects.create(code="roads", name="Roads")
    Department.objects.create(code="retired", name="Retired", is_active=False)
    request = APIRequestFactory().get("/departments/")
    force_authenticate(request, user=user)

    response = DepartmentViewSet.as_view({"get": "list"})(request)

    assert response.status_code == status.HTTP_200_OK
    assert [department["code"] for department in response.data["results"]] == ["roads"]

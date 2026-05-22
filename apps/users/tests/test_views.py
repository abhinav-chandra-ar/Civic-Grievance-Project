"""View tests for users endpoints."""
from __future__ import annotations

import pytest
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.users.views import CurrentUserView

pytestmark = pytest.mark.django_db


def test_current_user_view_returns_authenticated_profile(django_user_model) -> None:
    user = django_user_model.objects.create_user(
        username="citizen",
        password="password",
        phone_number="+919876543210",
    )
    request = APIRequestFactory().get("/users/me/")
    force_authenticate(request, user=user)

    response = CurrentUserView.as_view()(request)

    assert response.status_code == status.HTTP_200_OK
    assert response.data["username"] == "citizen"
    assert response.data["phone_number"] == "+919876543210"

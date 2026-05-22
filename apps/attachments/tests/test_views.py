"""View tests for attachment visibility."""
from __future__ import annotations

import pytest
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.attachments.services import register_attachment
from apps.attachments.views import AttachmentViewSet
from apps.grievances.services import submit_grievance

pytestmark = pytest.mark.django_db


def test_citizen_list_returns_own_grievance_attachments(django_user_model) -> None:
    citizen = django_user_model.objects.create_user(username="citizen", password="password")
    other = django_user_model.objects.create_user(username="other", password="password")
    own_grievance = submit_grievance(submitter=citizen, raw_text="Own image.")
    other_grievance = submit_grievance(submitter=other, raw_text="Other image.")
    register_attachment(
        uploader=citizen,
        grievance=own_grievance,
        storage_reference="own/photo.jpg",
        original_filename="photo.jpg",
        content_type="image/jpeg",
        file_size_bytes=100,
        content_hash="d" * 64,
    )
    register_attachment(
        uploader=other,
        grievance=other_grievance,
        storage_reference="other/photo.jpg",
        original_filename="photo.jpg",
        content_type="image/jpeg",
        file_size_bytes=100,
        content_hash="e" * 64,
    )
    request = APIRequestFactory().get("/attachments/")
    force_authenticate(request, user=citizen)

    response = AttachmentViewSet.as_view({"get": "list"})(request)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data["results"]) == 1
    assert response.data["results"][0]["storage_reference"] == "own/photo.jpg"

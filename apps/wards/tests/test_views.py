"""View tests for ward GeoJSON endpoints."""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Polygon
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.wards.models import Ward
from apps.wards.views import WardViewSet

pytestmark = pytest.mark.django_db


def test_ward_retrieve_returns_geojson_feature(django_user_model) -> None:
    user = django_user_model.objects.create_user(username="viewer", password="password")
    ward = Ward.objects.create(
        code="ward_01",
        name="Ward 01",
        boundary=Polygon(
            ((76.0, 10.0), (76.0, 10.1), (76.1, 10.1), (76.1, 10.0), (76.0, 10.0)),
            srid=4326,
        ),
    )
    request = APIRequestFactory().get("/wards/ward_01/")
    force_authenticate(request, user=user)

    response = WardViewSet.as_view({"get": "retrieve"})(request, pk=ward.pk)

    assert response.status_code == status.HTTP_200_OK
    assert response.data["type"] == "Feature"
    assert response.data["geometry"]["type"] == "Polygon"

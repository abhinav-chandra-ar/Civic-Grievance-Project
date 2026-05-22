"""View tests for landmark GeoJSON endpoints."""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Point
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.landmarks.models import Landmark, LandmarkType
from apps.landmarks.views import LandmarkViewSet

pytestmark = pytest.mark.django_db


def test_landmark_retrieve_returns_geojson_feature(django_user_model) -> None:
    user = django_user_model.objects.create_user(username="viewer", password="password")
    landmark = Landmark.objects.create(
        code="central_market",
        primary_name="Central Market",
        location=Point(76.27, 9.98, srid=4326),
        landmark_type=LandmarkType.MARKET,
    )
    request = APIRequestFactory().get("/landmarks/central_market/")
    force_authenticate(request, user=user)

    response = LandmarkViewSet.as_view({"get": "retrieve"})(request, pk=landmark.pk)

    assert response.status_code == status.HTTP_200_OK
    assert response.data["type"] == "Feature"
    assert response.data["geometry"]["type"] == "Point"

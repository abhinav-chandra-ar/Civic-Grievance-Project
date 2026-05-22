"""Read-side queries for landmarks."""
from __future__ import annotations

from django.contrib.gis.geos import Point
from django.contrib.gis.measure import Distance
from django.db.models import QuerySet

from .models import Landmark, LandmarkType


def landmark_list(*, active_only: bool = True) -> QuerySet[Landmark]:
    """Return landmarks using stable display ordering."""
    landmarks = Landmark.objects.all()
    if active_only:
        return landmarks.filter(is_active=True)
    return landmarks


def landmark_get_by_code(*, code: str) -> Landmark:
    """Return a landmark by its stable external code."""
    return Landmark.objects.get(code=code)


def landmark_list_by_type(
    *, landmark_type: str | LandmarkType, active_only: bool = True
) -> QuerySet[Landmark]:
    """Return landmarks filtered by a closed landmark type."""
    return landmark_list(active_only=active_only).filter(landmark_type=str(landmark_type))


def landmark_list_near_point(
    *, point: Point, radius_meters: float, active_only: bool = True
) -> QuerySet[Landmark]:
    """Return landmarks within a metric search radius around a point."""
    return landmark_list(active_only=active_only).filter(
        location__distance_lte=(point, Distance(m=radius_meters))
    )

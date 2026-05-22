"""Read-side queries for wards."""
from __future__ import annotations

from django.contrib.gis.geos import Point
from django.db.models import QuerySet

from .models import Ward


def ward_list(*, active_only: bool = True) -> QuerySet[Ward]:
    """Return wards using stable display ordering."""
    wards = Ward.objects.all()
    if active_only:
        return wards.filter(is_active=True)
    return wards


def ward_get_by_code(*, code: str) -> Ward:
    """Return a ward by its stable external code."""
    return Ward.objects.get(code=code)


def ward_list_containing_point(*, point: Point, active_only: bool = True) -> QuerySet[Ward]:
    """Return wards whose PostGIS polygon contains a point."""
    return ward_list(active_only=active_only).filter(boundary__contains=point)

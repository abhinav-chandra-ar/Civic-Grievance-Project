"""Selectors are intentionally minimal for the integrations boundary."""
from __future__ import annotations

from apps.landmarks.selectors import landmark_list


def local_landmark_candidates_for_mention(*, mention: str, limit: int = 5) -> list[dict[str, object]]:
    """Return repo-local landmark candidates for orchestration enrichment only."""
    normalized = mention.strip()
    if not normalized:
        return []

    candidates = landmark_list().filter(primary_name__icontains=normalized)[:limit]
    return [
        {
            "code": landmark.code,
            "primary_name": landmark.primary_name,
            "landmark_type": landmark.landmark_type,
            "source": "local_landmark_catalog",
        }
        for landmark in candidates
    ]


def recent_grievance_summaries_for_duplicate_context(
    *, ward_code: str | None, limit: int = 50, exclude_pk: int | None = None
) -> list[str]:
    """Return recent normalised summaries from the DB for duplicate detection.

    Supplies real text context to the Phase-A Jaccard duplicate detector inside
    ``analyze_complaint()``.  The query is intentionally bounded (``LIMIT``) so
    it can never become a full-table scan on a live corpus.

    Parameters
    ----------
    ward_code
        TVMC ward code (e.g. ``"tvm_001"``).  When provided the result is
        scoped to grievances in that ward, giving more relevant context for
        the duplicate check.  Pass ``None`` or ``""`` for a global sample.
    limit
        Maximum number of summaries to return (default 50).
    exclude_pk
        Primary key of a grievance to exclude from the pool.  Pass the PK of
        the grievance currently being enriched so it cannot match its own
        previously stored ``normalized_summary`` and be falsely flagged as a
        confirmed self-duplicate on re-enrichment.

    Returns
    -------
    list[str]
        A list of non-empty ``normalized_summary`` values ordered newest first.
        Returns an empty list when the query fails.
    """
    from apps.grievances.models import Grievance  # noqa: PLC0415 — lazy to avoid circular import

    qs = Grievance.objects.exclude(normalized_summary="").order_by("-submitted_at")
    if ward_code:
        qs = qs.filter(ward__code=ward_code)
    if exclude_pk is not None:
        qs = qs.exclude(pk=exclude_pk)
    return list(qs.values_list("normalized_summary", flat=True)[:limit])

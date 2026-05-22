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

"""apps/integrations/routing.py

Phase E — KSMART-style routing intelligence.

Resolves ward, department, and routing bucket from ML analysis results
without exposing officer identity or internal assignment details.

Routing hierarchy  (Kerala LSG / KSMART-aligned)
-------------------------------------------------
1. ``manual_review``    — AI flagged the complaint for human review or
                          rejection.  No automated routing happens.
2. ``ward_queue``       — Ward resolved from the ML landmark alias scan
                          with confidence ≥ 0.60.  Complaint enters the
                          ward's internal processing queue.
3. ``department_queue`` — Department resolved via category mapping but no
                          ward was pinpointed.  Complaint enters the
                          department's processing queue.
4. ``central_queue``    — Fallback when neither ward nor department could
                          be resolved.  Falls to the central triage queue.

Privacy / visibility rules
--------------------------
* Officers see *queues*, not named assignments.  No officer identity
  appears in any routing output.
* Citizens see tracking code + status only.
* Department admins see their department queue.
* Super-admins see full routing metadata via ``status_metadata``.

Design constraints
------------------
* No GIS in this phase — ward resolution uses alias-based ``ward_hint``
  produced by the ML landmark extractor (``apps.ml.analyzer``).
* All DB access is isolated inside lazy function-level imports to avoid
  circular dependencies at module load time.
* All three resolver functions return gracefully on any failure; they
  never raise.  This guarantees enrichment always completes even when
  wards or departments are not yet seeded.
"""
from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)

# Minimum ward resolution confidence required to prefer ``ward_queue``
# over ``department_queue``.  The ML alias scan returns 0.85 on a
# direct ward/landmark match; lower values indicate weaker evidence.
_WARD_CONFIDENCE_MIN: float = 0.60


# ---------------------------------------------------------------------------
# 1. Ward resolver
# ---------------------------------------------------------------------------


def resolve_ward_from_hint(*, ward_hint: str | None) -> dict[str, Any]:
    """Resolve a Ward DB record from the ML-generated ward-code hint.

    The ML ``extract_landmarks()`` function scans a comprehensive alias
    dictionary covering all 101 TVMC wards, transport hubs, hospitals,
    temples, and named junctions.  When a match is found it sets
    ``ward_hint`` to the matched ward code (e.g. ``"tvm_034"``).  This
    function looks that code up in the live Ward table so the Grievance
    FK can be set.

    Parameters
    ----------
    ward_hint
        Ward code produced by the ML landmark extractor, or ``None``
        when no location alias was matched in the complaint text.

    Returns
    -------
    dict
        * ``ward``       — :class:`~apps.wards.models.Ward` instance or ``None``
        * ``ward_code``  — str or ``None``
        * ``confidence`` — float 0.0–1.0
        * ``source``     — one of ``"landmark_alias_match"``,
          ``"ward_not_in_db"``, ``"unresolved"``
    """
    if not ward_hint:
        return {
            "ward":       None,
            "ward_code":  None,
            "confidence": 0.0,
            "source":     "unresolved",
        }

    # Lazy import — routing.py must not hold a module-level reference to
    # Ward because apps.integrations loads early in the Django start chain.
    from apps.wards.models import Ward  # noqa: PLC0415

    try:
        ward = Ward.objects.get(code=ward_hint, is_active=True)
        return {
            "ward":       ward,
            "ward_code":  ward.code,
            "confidence": 0.85,
            "source":     "landmark_alias_match",
        }
    except Ward.DoesNotExist:
        _logger.debug(
            "Phase E ward resolver: ward_hint %r not found in DB "
            "(ward not yet seeded?)",
            ward_hint,
        )
        return {
            "ward":       None,
            "ward_code":  ward_hint,
            "confidence": 0.0,
            "source":     "ward_not_in_db",
        }


# ---------------------------------------------------------------------------
# 2. Department resolver
# ---------------------------------------------------------------------------


def resolve_department_from_category(
    *,
    category_code: str,
    department_code: str,
) -> dict[str, Any]:
    """Resolve a Department DB record from ML-generated category/department hints.

    Resolution order
    ----------------
    1. **Direct code match** — the ML ``detect_department()`` function
       maps each category to a department code string (e.g.
       ``"roads_and_drainage"``).  If a Department with that code exists
       in the DB it is returned (highest confidence: 0.90).
    2. **GIN-indexed category fallback** — queries departments whose
       ``handled_categories`` JSON list contains ``category_code``.
       Uses the GIN index added during the DB optimisation phase
       (migration ``departments.0002``).

    Parameters
    ----------
    category_code
        Civic category code produced by the ML engine
        (e.g. ``"road_damage"``).
    department_code
        Department code hint produced by the ML engine
        (e.g. ``"roads_and_drainage"``).

    Returns
    -------
    dict
        * ``department``       — :class:`~apps.departments.models.Department`
          instance or ``None``
        * ``department_code``  — str or ``None``
        * ``confidence``       — float 0.0–1.0
        * ``source``           — one of ``"direct_code_match"``,
          ``"category_match"``, ``"unresolved"``
    """
    from apps.departments.models import Department  # noqa: PLC0415

    # Step 1: direct code lookup (ML engine already maps categories →
    # department codes via _CATEGORY_TO_DEPT in apps.ml.analyzer).
    if department_code:
        try:
            dept = Department.objects.get(code=department_code, is_active=True)
            return {
                "department":      dept,
                "department_code": dept.code,
                "confidence":      0.90,
                "source":          "direct_code_match",
            }
        except Department.DoesNotExist:
            _logger.debug(
                "Phase E dept resolver: department_code %r not in DB.",
                department_code,
            )

    # Step 2: handled_categories GIN index fallback.
    if category_code:
        dept = (
            Department.objects.filter(
                handled_categories__contains=[category_code],
                is_active=True,
            )
            .order_by("id")
            .first()
        )
        if dept is not None:
            return {
                "department":      dept,
                "department_code": dept.code,
                "confidence":      0.75,
                "source":          "category_match",
            }

    return {
        "department":      None,
        "department_code": None,
        "confidence":      0.0,
        "source":          "unresolved",
    }


# ---------------------------------------------------------------------------
# 3. Routing bucket resolver
# ---------------------------------------------------------------------------


def resolve_routing_bucket(
    *,
    ward_result: dict[str, Any],
    dept_result: dict[str, Any],
    ai_decision: dict[str, Any],
) -> dict[str, Any]:
    """Determine the internal routing queue bucket for a grievance.

    Officers see queues, not named assignments — this function never
    exposes any officer identity.

    Routing hierarchy (evaluated top-to-bottom, first match wins)
    -------------------------------------------------------------
    1. ``manual_review``    — AI flagged for review or rejection.
    2. ``ward_queue``       — Ward DB record found, confidence ≥ 0.60.
    3. ``department_queue`` — Department DB record found.
    4. ``central_queue``    — Fallback; no FK resolution succeeded.

    Parameters
    ----------
    ward_result
        Output of :func:`resolve_ward_from_hint`.
    dept_result
        Output of :func:`resolve_department_from_category`.
    ai_decision
        The AI decision dict (``automation_action``, ``needs_review``, …).

    Returns
    -------
    dict
        * ``routing_bucket`` — unique queue identifier string
        * ``routing_mode``   — ``"ward_queue"`` | ``"department_queue"`` |
          ``"central_queue"`` | ``"manual_review"``
        * ``confidence``     — float 0.0–1.0
    """
    automation_action = str(ai_decision.get("automation_action", ""))
    needs_review      = bool(ai_decision.get("needs_review", False))

    # Manual review takes absolute precedence over all automated routing.
    if automation_action in ("review_required", "reject") or needs_review:
        return {
            "routing_bucket": "manual_review",
            "routing_mode":   "manual_review",
            "confidence":     1.0,
        }

    ward_instance = ward_result.get("ward")
    ward_code     = ward_result.get("ward_code")
    ward_conf     = float(ward_result.get("confidence", 0.0))

    dept_instance = dept_result.get("department")
    dept_code     = dept_result.get("department_code")
    dept_conf     = float(dept_result.get("confidence", 0.0))

    # Ward queue: DB record resolved AND confidence threshold met.
    if ward_instance is not None and ward_conf >= _WARD_CONFIDENCE_MIN:
        return {
            "routing_bucket": f"ward_{ward_code}",
            "routing_mode":   "ward_queue",
            "confidence":     ward_conf,
        }

    # Department queue: DB record resolved (any confidence).
    if dept_instance is not None:
        return {
            "routing_bucket": f"dept_{dept_code}",
            "routing_mode":   "department_queue",
            "confidence":     dept_conf,
        }

    # Central queue: fallback for unresolvable complaints.
    return {
        "routing_bucket": "central",
        "routing_mode":   "central_queue",
        "confidence":     0.0,
    }


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------


def _serializable_ward_result(ward_result: dict[str, Any]) -> dict[str, Any]:
    """Strip the Ward ORM instance from a resolution dict for JSON storage."""
    ward = ward_result.get("ward")
    return {
        "ward_id":    ward.pk if ward is not None else None,
        "ward_code":  ward_result.get("ward_code"),
        "confidence": ward_result.get("confidence"),
        "source":     ward_result.get("source"),
    }


def _serializable_dept_result(dept_result: dict[str, Any]) -> dict[str, Any]:
    """Strip the Department ORM instance from a resolution dict for JSON storage."""
    dept = dept_result.get("department")
    return {
        "department_id":   dept.pk if dept is not None else None,
        "department_code": dept_result.get("department_code"),
        "confidence":      dept_result.get("confidence"),
        "source":          dept_result.get("source"),
    }


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------


def build_phase_e_routing(
    *,
    routing_context: dict[str, Any],
    ai_decision: dict[str, Any],
) -> dict[str, Any]:
    """Orchestrate all Phase E resolvers and return the combined routing payload.

    Called by :func:`apps.integrations.services.enrich_grievance_with_ai`
    after the AI analysis phase completes.  The result both populates FK
    fields on the Grievance record and writes a routing audit trail into
    ``status_metadata``.

    Parameters
    ----------
    routing_context
        Extracted from the ``analyze_grievance_submission()`` payload
        under the ``"routing_context"`` key.  Expected keys:
        ``ward_hint``, ``category_code``, ``department_code``.
        Missing keys are treated as empty/None — no error is raised.
    ai_decision
        The ``ai_decision`` dict extracted from the enrichment payload
        (``automation_action``, ``needs_review``, etc.).

    Returns
    -------
    dict
        * ``ward_instance``       — :class:`~apps.wards.models.Ward` or ``None``
        * ``department_instance`` — :class:`~apps.departments.models.Department`
          or ``None``
        * ``routing_metadata``    — JSON-serialisable audit dict suitable for
          direct storage in ``Grievance.status_metadata["phase_e_routing"]``
    """
    ward_hint       = routing_context.get("ward_hint") or None
    category_code   = routing_context.get("category_code") or ""
    department_code = routing_context.get("department_code") or ""

    ward_result = resolve_ward_from_hint(ward_hint=ward_hint)
    dept_result = resolve_department_from_category(
        category_code=category_code,
        department_code=department_code,
    )
    routing = resolve_routing_bucket(
        ward_result=ward_result,
        dept_result=dept_result,
        ai_decision=ai_decision,
    )

    return {
        "ward_instance":       ward_result.get("ward"),
        "department_instance": dept_result.get("department"),
        "routing_metadata": {
            "ward_resolution":       _serializable_ward_result(ward_result),
            "department_resolution": _serializable_dept_result(dept_result),
            "routing_bucket":        routing["routing_bucket"],
            "routing_mode":          routing["routing_mode"],
            "routing_confidence":    routing["confidence"],
        },
    }

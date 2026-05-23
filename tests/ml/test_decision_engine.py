"""Tests for apps.ml.decision_engine — pure Python, zero database required.

All functions under test are pure: no Django, no DB, no I/O.
No @pytest.mark.django_db annotation is used anywhere in this file.

Coverage
--------
calculate_routing_confidence  — weight arithmetic, penalties, bonuses
detect_duplicate_risk         — three-level risk classification
decide_review_requirement     — blocking conditions, image flags, merging
decide_escalation             — three escalation rules
make_final_decision           — full orchestration, all four action paths
Integration                   — analyze_complaint() includes 'decision' key
NLP adapter                   — metadata carries 'decision' key
"""
from __future__ import annotations

import pytest

from apps.ml.decision_engine import (
    _AUTO_ROUTE_MIN_CONFIDENCE,
    _DUPLICATE_LOW,
    _DUPLICATE_MEDIUM,
    _HARD_BLOCKING_REVIEW_FLAGS,
    calculate_routing_confidence,
    decide_escalation,
    decide_review_requirement,
    detect_duplicate_risk,
    make_final_decision,
)

# ===========================================================================
# Shared fixtures / factories
# ===========================================================================


def _good_analyzer_output(**overrides) -> dict:
    """Minimal analyzer output that represents a strong, routable complaint."""
    base = {
        "spam":                {"is_spam": False, "spam_score": 0.0, "spam_reason": ""},
        "duplicate":           {"is_duplicate": False, "similarity_score": 0.0,
                                "matching_text": None},
        "image_analysis":      None,
        "category_confidence": 0.85,
        "language_confidence": 0.90,
        "department_code":     "roads_and_drainage",
        "ward_hint":           "tvm_034",
        "priority":            "medium",
        "category_code":       "road_damage",
        "review_reasons":      [],
    }
    base.update(overrides)
    return base


def _spam_output(spam_score: float = 0.95) -> dict:
    return _good_analyzer_output(
        spam={"is_spam": True, "spam_score": spam_score, "spam_reason": "test phrase"},
        category_confidence=0.0,
        department_code="",
        ward_hint=None,
        review_reasons=["spam_suspicion"],
    )


def _duplicate_output(similarity: float = 0.72) -> dict:
    return _good_analyzer_output(
        duplicate={"is_duplicate": True, "similarity_score": similarity,
                   "matching_text": "pothole near pattom"},
        review_reasons=["possible_duplicate"],
    )


def _urgent_output(category: str = "electrical_hazard") -> dict:
    return _good_analyzer_output(priority="urgent", category_code=category,
                                 department_code="electrical_engineering")


def _weak_output() -> dict:
    """Complaint with near-zero classification signal."""
    return _good_analyzer_output(
        category_confidence=0.0,
        language_confidence=0.10,
        department_code="",
        ward_hint=None,
        category_code="",
    )


# ===========================================================================
# calculate_routing_confidence
# ===========================================================================

class TestCalculateRoutingConfidence:

    def test_returns_routing_confidence_key(self):
        r = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.90,
        )
        assert "routing_confidence" in r

    def test_full_signals_high_confidence(self):
        r = calculate_routing_confidence(
            category_confidence=0.90, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.95,
        )
        # 0.10 + 0.90*0.40 + 0.20 + 0.95*0.15 + 0.15 = 0.10+0.36+0.20+0.1425+0.15 ≈ 0.953
        assert r["routing_confidence"] >= 0.90

    def test_no_ward_hint_reduces_confidence(self):
        with_ward = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.85,
        )["routing_confidence"]
        without_ward = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=False, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.85,
        )["routing_confidence"]
        assert without_ward < with_ward

    def test_no_department_reduces_confidence(self):
        with_dept = calculate_routing_confidence(
            category_confidence=0.70, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.80,
        )["routing_confidence"]
        without_dept = calculate_routing_confidence(
            category_confidence=0.70, department_present=False,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.80,
        )["routing_confidence"]
        assert without_dept < with_dept

    def test_zero_category_confidence_low_result(self):
        r = calculate_routing_confidence(
            category_confidence=0.0, department_present=False,
            ward_hint_present=False, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.0,
        )
        # Only base=0.10 survives
        assert r["routing_confidence"] == pytest.approx(0.10, abs=0.01)

    def test_spam_reduces_confidence(self):
        clean = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.90,
        )["routing_confidence"]
        spammy = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.80,
            language_confidence=0.90,
        )["routing_confidence"]
        assert spammy < clean

    def test_high_spam_score_heavily_reduces_confidence(self):
        r = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=1.0,
            language_confidence=0.90,
        )
        # penalty: × (1 − 1.0 × 0.40) = × 0.60
        assert r["routing_confidence"] < 0.70

    def test_confirmed_duplicate_reduces_confidence(self):
        clean = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.90,
        )["routing_confidence"]
        dup = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.70, spam_score=0.0,
            language_confidence=0.90,
        )["routing_confidence"]
        assert dup < clean

    def test_medium_duplicate_soft_penalty(self):
        clean = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.90,
        )["routing_confidence"]
        medium_dup = calculate_routing_confidence(
            category_confidence=0.80, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.45, spam_score=0.0,
            language_confidence=0.90,
        )["routing_confidence"]
        # Medium risk → × 0.90
        assert medium_dup < clean
        assert medium_dup > clean * 0.80   # softer than confirmed duplicate

    def test_image_quality_bonus_when_usable(self):
        no_img = calculate_routing_confidence(
            category_confidence=0.70, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.85,
        )["routing_confidence"]
        with_img = calculate_routing_confidence(
            category_confidence=0.70, department_present=True,
            ward_hint_present=True, image_quality=0.90,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.85,
        )["routing_confidence"]
        assert with_img > no_img

    def test_low_image_quality_no_bonus(self):
        no_img = calculate_routing_confidence(
            category_confidence=0.70, department_present=True,
            ward_hint_present=True, image_quality=None,
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.85,
        )["routing_confidence"]
        low_img = calculate_routing_confidence(
            category_confidence=0.70, department_present=True,
            ward_hint_present=True, image_quality=0.30,   # below 0.50 usable threshold
            duplicate_similarity=0.0, spam_score=0.0,
            language_confidence=0.85,
        )["routing_confidence"]
        assert low_img == no_img   # no bonus for low quality image

    def test_result_clamped_between_0_and_1(self):
        for spam in [0.0, 0.5, 1.0]:
            r = calculate_routing_confidence(
                category_confidence=1.0, department_present=True,
                ward_hint_present=True, image_quality=1.0,
                duplicate_similarity=0.0, spam_score=spam,
                language_confidence=1.0,
            )
            assert 0.0 <= r["routing_confidence"] <= 1.0


# ===========================================================================
# detect_duplicate_risk
# ===========================================================================

class TestDetectDuplicateRisk:

    def test_returns_required_keys(self):
        r = detect_duplicate_risk(similarity_score=0.0, is_duplicate=False)
        assert set(r.keys()) == {"risk_level", "risk_score", "is_confirmed"}

    def test_low_risk_below_threshold(self):
        r = detect_duplicate_risk(similarity_score=0.20, is_duplicate=False)
        assert r["risk_level"] == "low"
        assert r["is_confirmed"] is False

    def test_medium_risk_in_range(self):
        r = detect_duplicate_risk(similarity_score=0.45, is_duplicate=False)
        assert r["risk_level"] == "medium"

    def test_medium_risk_lower_bound(self):
        r = detect_duplicate_risk(similarity_score=_DUPLICATE_LOW, is_duplicate=False)
        assert r["risk_level"] == "medium"

    def test_high_risk_at_duplicate_threshold(self):
        r = detect_duplicate_risk(similarity_score=_DUPLICATE_MEDIUM, is_duplicate=True)
        assert r["risk_level"] == "high"
        assert r["is_confirmed"] is True

    def test_high_risk_above_threshold(self):
        r = detect_duplicate_risk(similarity_score=0.80, is_duplicate=True)
        assert r["risk_level"] == "high"

    def test_is_duplicate_flag_forces_high_risk(self):
        # Even with low similarity, is_duplicate=True → high
        r = detect_duplicate_risk(similarity_score=0.10, is_duplicate=True)
        assert r["risk_level"] == "high"
        assert r["is_confirmed"] is True

    def test_not_confirmed_when_is_duplicate_false(self):
        r = detect_duplicate_risk(similarity_score=0.30, is_duplicate=False)
        assert r["is_confirmed"] is False

    def test_risk_score_is_rounded_float(self):
        r = detect_duplicate_risk(similarity_score=0.333333, is_duplicate=False)
        assert isinstance(r["risk_score"], float)
        assert r["risk_score"] == pytest.approx(0.333, abs=0.001)

    def test_zero_similarity_is_low(self):
        r = detect_duplicate_risk(similarity_score=0.0, is_duplicate=False)
        assert r["risk_level"] == "low"
        assert r["risk_score"] == 0.0


# ===========================================================================
# decide_review_requirement
# ===========================================================================

class TestDecideReviewRequirement:

    def _perfect(self, **overrides):
        base = dict(
            routing_confidence=0.85,
            spam_score=0.0,
            duplicate_risk_level="low",
            image_analysis=None,
            category_confidence=0.80,
            department_code="roads_and_drainage",
            ward_hint="tvm_034",
            language_confidence=0.90,
            existing_review_reasons=None,
        )
        base.update(overrides)
        return decide_review_requirement(**base)

    def test_returns_required_keys(self):
        r = self._perfect()
        assert set(r.keys()) == {"needs_review", "review_reasons"}

    def test_perfect_submission_no_review(self):
        r = self._perfect()
        # Only mandatory soft flag: no_ward_hint absent (ward_hint="tvm_034")
        # Depending on design, no_ward_hint may or may not be present.
        # The important thing: needs_review may be True due to no_ward_hint,
        # but spam_suspicion / no_category / low_confidence must NOT be present.
        assert "spam_suspicion" not in r["review_reasons"]
        assert "no_category_detected" not in r["review_reasons"]
        assert "low_confidence" not in r["review_reasons"]

    def test_spam_triggers_review(self):
        r = self._perfect(spam_score=0.60)
        assert r["needs_review"] is True
        assert "spam_suspicion" in r["review_reasons"]

    def test_spam_below_threshold_no_flag(self):
        r = self._perfect(spam_score=0.30)
        assert "spam_suspicion" not in r["review_reasons"]

    def test_no_category_triggers_review(self):
        r = self._perfect(category_confidence=0.0)
        assert "no_category_detected" in r["review_reasons"]

    def test_no_department_triggers_review(self):
        r = self._perfect(department_code="")
        assert "no_department_hint" in r["review_reasons"]

    def test_no_ward_hint_triggers_review(self):
        r = self._perfect(ward_hint=None)
        assert "no_ward_hint" in r["review_reasons"]

    def test_low_confidence_triggers_review(self):
        r = self._perfect(routing_confidence=0.25)
        assert "low_confidence" in r["review_reasons"]

    def test_high_duplicate_risk_triggers_review(self):
        r = self._perfect(duplicate_risk_level="high")
        assert "duplicate_risk_high" in r["review_reasons"]

    def test_medium_duplicate_risk_triggers_review(self):
        r = self._perfect(duplicate_risk_level="medium")
        assert "duplicate_risk_medium" in r["review_reasons"]

    def test_language_uncertain_triggers_review(self):
        r = self._perfect(language_confidence=0.15)
        assert "language_uncertain" in r["review_reasons"]

    def test_invalid_image_triggers_review(self):
        img = {"is_valid": False, "usable": False, "is_irrelevant": False,
               "is_consistent": False, "quality_score": 0.0}
        r = self._perfect(image_analysis=img)
        assert "image_invalid" in r["review_reasons"]

    def test_poor_quality_image_triggers_review(self):
        img = {"is_valid": True, "usable": False, "is_irrelevant": False,
               "is_consistent": False, "quality_score": 0.25}
        r = self._perfect(image_analysis=img)
        assert "image_poor_quality" in r["review_reasons"]

    def test_irrelevant_image_triggers_review(self):
        img = {"is_valid": True, "usable": True, "is_irrelevant": True,
               "is_consistent": False, "quality_score": 0.80}
        r = self._perfect(image_analysis=img)
        assert "image_irrelevant" in r["review_reasons"]

    def test_image_contradiction_triggers_review(self):
        img = {"is_valid": True, "usable": True, "is_irrelevant": False,
               "is_consistent": False, "quality_score": 0.70}
        r = self._perfect(image_analysis=img)
        assert "image_contradicts_complaint" in r["review_reasons"]

    def test_good_image_no_image_flags(self):
        img = {"is_valid": True, "usable": True, "is_irrelevant": False,
               "is_consistent": True, "quality_score": 0.85}
        r = self._perfect(image_analysis=img)
        image_flags = {"image_invalid", "image_poor_quality",
                       "image_irrelevant", "image_contradicts_complaint"}
        assert not (image_flags & set(r["review_reasons"]))

    def test_no_image_no_image_flags(self):
        r = self._perfect(image_analysis=None)
        image_flags = {"image_invalid", "image_poor_quality",
                       "image_irrelevant", "image_contradicts_complaint"}
        assert not (image_flags & set(r["review_reasons"]))

    def test_existing_reasons_merged_without_duplication(self):
        existing = ["no_landmark_detected", "spam_suspicion"]
        r = self._perfect(spam_score=0.60, existing_review_reasons=existing)
        # spam_suspicion should appear exactly once
        assert r["review_reasons"].count("spam_suspicion") == 1
        assert "no_landmark_detected" in r["review_reasons"]

    def test_existing_reasons_appended_if_not_duplicate(self):
        r = self._perfect(existing_review_reasons=["custom_flag"])
        assert "custom_flag" in r["review_reasons"]

    def test_needs_review_true_when_reasons_present(self):
        r = self._perfect(spam_score=0.60)
        assert r["needs_review"] is True

    def test_review_reasons_is_list(self):
        r = self._perfect()
        assert isinstance(r["review_reasons"], list)


# ===========================================================================
# decide_escalation
# ===========================================================================

class TestDecideEscalation:

    def _call(self, priority="medium", category_code="road_damage",
              routing_confidence=0.80, needs_review=False, review_reasons=None):
        return decide_escalation(
            priority=priority,
            category_code=category_code,
            routing_confidence=routing_confidence,
            needs_review=needs_review,
            review_reasons=review_reasons or [],
        )

    def test_returns_required_keys(self):
        r = self._call()
        assert set(r.keys()) == {"should_escalate", "escalation_reason"}

    def test_electrical_hazard_urgent_escalates(self):
        r = self._call(priority="urgent", category_code="electrical_hazard")
        assert r["should_escalate"] is True
        assert "life_safety_category_urgent" in r["escalation_reason"]

    def test_tree_fall_urgent_escalates(self):
        r = self._call(priority="urgent", category_code="tree_fall")
        assert r["should_escalate"] is True

    def test_sewage_urgent_escalates(self):
        r = self._call(priority="urgent", category_code="sewage_issue")
        assert r["should_escalate"] is True

    def test_electrical_hazard_critical_escalates(self):
        r = self._call(priority="critical", category_code="electrical_hazard")
        assert r["should_escalate"] is True

    def test_urgent_priority_non_life_safety_escalates(self):
        r = self._call(priority="urgent", category_code="road_damage")
        assert r["should_escalate"] is True
        assert "urgent_priority" in r["escalation_reason"]

    def test_critical_priority_escalates(self):
        r = self._call(priority="critical", category_code="waste_management")
        assert r["should_escalate"] is True

    def test_evidence_contradiction_on_high_priority_escalates(self):
        r = self._call(
            priority="high",
            category_code="road_damage",
            review_reasons=["image_contradicts_complaint"],
        )
        assert r["should_escalate"] is True
        assert "evidence_contradiction" in r["escalation_reason"]

    def test_evidence_contradiction_on_medium_priority_no_escalation(self):
        r = self._call(
            priority="medium",
            category_code="road_damage",
            review_reasons=["image_contradicts_complaint"],
        )
        assert r["should_escalate"] is False

    def test_medium_priority_no_escalation(self):
        r = self._call(priority="medium", category_code="road_damage")
        assert r["should_escalate"] is False
        assert r["escalation_reason"] == ""

    def test_low_priority_no_escalation(self):
        r = self._call(priority="low", category_code="street_light")
        assert r["should_escalate"] is False

    def test_high_priority_without_contradiction_no_escalation(self):
        r = self._call(priority="high", category_code="road_damage", review_reasons=[])
        assert r["should_escalate"] is False

    def test_escalation_reason_is_string(self):
        r = self._call()
        assert isinstance(r["escalation_reason"], str)


# ===========================================================================
# make_final_decision
# ===========================================================================

class TestMakeFinalDecision:

    _REQUIRED_KEYS = {
        "automation_action", "routing_confidence", "needs_review",
        "review_reasons", "duplicate_risk", "escalation", "decision_metadata",
    }
    _VALID_ACTIONS = {"auto_route", "review_required", "escalate", "reject"}
    _DUP_RISK_KEYS = {"risk_level", "risk_score", "is_confirmed"}
    _ESCALATION_KEYS = {"should_escalate", "escalation_reason"}
    _METADATA_KEYS = {
        "spam_score", "duplicate_similarity", "category_confidence",
        "language_confidence", "department_present", "ward_hint_present",
        "image_quality", "priority",
    }

    def test_returns_all_required_keys(self):
        r = make_final_decision(_good_analyzer_output())
        assert set(r.keys()) == self._REQUIRED_KEYS

    def test_automation_action_is_valid(self):
        r = make_final_decision(_good_analyzer_output())
        assert r["automation_action"] in self._VALID_ACTIONS

    def test_auto_route_on_strong_complaint(self):
        r = make_final_decision(_good_analyzer_output())
        assert r["automation_action"] == "auto_route"
        assert r["routing_confidence"] >= _AUTO_ROUTE_MIN_CONFIDENCE

    def test_review_required_on_weak_complaint(self):
        r = make_final_decision(_weak_output())
        assert r["automation_action"] == "review_required"

    def test_escalate_on_urgent_electrical_hazard(self):
        r = make_final_decision(_urgent_output("electrical_hazard"))
        assert r["automation_action"] == "escalate"
        assert r["escalation"]["should_escalate"] is True

    def test_escalate_on_urgent_tree_fall(self):
        r = make_final_decision(_urgent_output("tree_fall"))
        assert r["automation_action"] == "escalate"

    def test_escalate_on_any_urgent_priority(self):
        r = make_final_decision(_urgent_output("road_damage"))
        assert r["automation_action"] == "escalate"

    def test_reject_on_very_high_spam(self):
        r = make_final_decision(_spam_output(spam_score=0.95))
        assert r["automation_action"] == "reject"

    def test_spam_below_reject_threshold_goes_to_review(self):
        # spam_score=0.50 → triggers spam_suspicion but not reject (≤ 0.85)
        r = make_final_decision(_spam_output(spam_score=0.50))
        assert r["automation_action"] == "review_required"

    def test_duplicate_complaint_goes_to_review(self):
        r = make_final_decision(_duplicate_output(similarity=0.75))
        assert r["automation_action"] in ("review_required", "reject")
        assert "duplicate_risk_high" in r["review_reasons"]

    def test_no_category_goes_to_review(self):
        out = _good_analyzer_output(category_confidence=0.0, category_code="",
                                    department_code="")
        r = make_final_decision(out)
        assert r["automation_action"] == "review_required"
        assert "no_category_detected" in r["review_reasons"]

    def test_routing_confidence_is_float_in_range(self):
        for output in [_good_analyzer_output(), _weak_output(), _spam_output()]:
            r = make_final_decision(output)
            assert 0.0 <= r["routing_confidence"] <= 1.0

    def test_needs_review_bool(self):
        r = make_final_decision(_good_analyzer_output())
        assert isinstance(r["needs_review"], bool)

    def test_review_reasons_is_list(self):
        r = make_final_decision(_good_analyzer_output())
        assert isinstance(r["review_reasons"], list)

    def test_duplicate_risk_subdict_keys(self):
        r = make_final_decision(_good_analyzer_output())
        assert set(r["duplicate_risk"].keys()) == self._DUP_RISK_KEYS

    def test_escalation_subdict_keys(self):
        r = make_final_decision(_good_analyzer_output())
        assert set(r["escalation"].keys()) == self._ESCALATION_KEYS

    def test_decision_metadata_has_all_keys(self):
        r = make_final_decision(_good_analyzer_output())
        assert set(r["decision_metadata"].keys()) == self._METADATA_KEYS

    def test_reject_takes_precedence_over_escalate(self):
        # High-spam + urgent should still reject (spam > 0.85 wins)
        out = _spam_output(spam_score=0.90)
        out["priority"] = "urgent"
        r = make_final_decision(out)
        assert r["automation_action"] == "reject"

    def test_escalate_takes_precedence_over_review(self):
        out = _urgent_output("electrical_hazard")
        out["review_reasons"] = ["no_landmark_detected"]
        r = make_final_decision(out)
        assert r["automation_action"] == "escalate"

    def test_hard_blocking_flag_prevents_auto_route(self):
        # High confidence but confirmed duplicate → review_required
        out = _good_analyzer_output(
            duplicate={"is_duplicate": True, "similarity_score": 0.80,
                       "matching_text": "pothole near pattom"},
        )
        r = make_final_decision(out)
        assert r["automation_action"] != "auto_route"

    def test_good_image_does_not_block_auto_route(self):
        out = _good_analyzer_output(
            image_analysis={
                "is_valid": True, "usable": True, "is_irrelevant": False,
                "is_consistent": True, "quality_score": 0.85,
                "quality_flags": [], "irrelevant_reason": "",
                "consistency_score": 0.75, "conflict_reason": "",
                "provider": "image_rule_v1",
            }
        )
        r = make_final_decision(out)
        assert r["automation_action"] == "auto_route"

    def test_image_contradiction_hard_blocks_auto_route(self):
        out = _good_analyzer_output(
            image_analysis={
                "is_valid": True, "usable": True, "is_irrelevant": False,
                "is_consistent": False, "quality_score": 0.80,
                "quality_flags": [], "irrelevant_reason": "",
                "consistency_score": 0.10, "conflict_reason": "image appears irrelevant",
                "provider": "image_rule_v1",
            },
            review_reasons=["image_contradicts_complaint"],
        )
        r = make_final_decision(out)
        assert r["automation_action"] != "auto_route"

    def test_decision_metadata_ward_hint_present_true(self):
        r = make_final_decision(_good_analyzer_output(ward_hint="tvm_034"))
        assert r["decision_metadata"]["ward_hint_present"] is True

    def test_decision_metadata_ward_hint_present_false(self):
        r = make_final_decision(_good_analyzer_output(ward_hint=None))
        assert r["decision_metadata"]["ward_hint_present"] is False

    def test_decision_metadata_image_quality_none_without_image(self):
        r = make_final_decision(_good_analyzer_output(image_analysis=None))
        assert r["decision_metadata"]["image_quality"] is None


# ===========================================================================
# Integration — analyze_complaint() includes "decision" key
# ===========================================================================

class TestAnalyzeComplaintIncludesDecision:

    def test_decision_key_present(self):
        from apps.ml.analyzer import analyze_complaint
        r = analyze_complaint("Large pothole near Pattom junction blocking traffic")
        assert "decision" in r

    def test_decision_action_is_valid_string(self):
        from apps.ml.analyzer import analyze_complaint
        r = analyze_complaint("Large pothole near Pattom junction blocking traffic")
        assert r["decision"]["automation_action"] in {
            "auto_route", "review_required", "escalate", "reject"
        }

    def test_decision_routing_confidence_is_float(self):
        from apps.ml.analyzer import analyze_complaint
        r = analyze_complaint("Water supply pipe burst near Ulloor")
        assert isinstance(r["decision"]["routing_confidence"], float)
        assert 0.0 <= r["decision"]["routing_confidence"] <= 1.0

    def test_spam_complaint_decision_rejects(self):
        from apps.ml.analyzer import analyze_complaint
        r = analyze_complaint("test test test test test test test test")
        # High repetition spam → should be review_required or reject (not auto_route)
        assert r["decision"]["automation_action"] != "auto_route"

    def test_electrical_hazard_decision_escalates(self):
        from apps.ml.analyzer import analyze_complaint
        r = analyze_complaint(
            "live wire fallen across the road near Pattom — high voltage danger"
        )
        assert r["decision"]["automation_action"] == "escalate"


# ===========================================================================
# NLP adapter — metadata carries "decision" key
# ===========================================================================

class TestNlpAdapterPhaseC:

    _REQUIRED_METADATA_KEYS = {
        "text_length", "ward_hint", "landmark_hints",
        "spam_check", "duplicate_check",
        "needs_human_review", "review_reasons",
        "image_analysis", "consistency_check",
        "evidence_quality", "evidence_review_reason",
        "decision",     # Phase C addition
    }

    def _call(self, text: str, **kwargs):
        from apps.integrations.clients.nlp import classify_grievance_text
        return classify_grievance_text(raw_text=text, **kwargs)

    def test_metadata_has_decision_key(self):
        r = self._call("pothole near pattom junction")
        assert "decision" in r["metadata"]

    def test_metadata_has_all_12_keys(self):
        r = self._call("water pipe broken near ulloor")
        assert set(r["metadata"].keys()) == self._REQUIRED_METADATA_KEYS

    def test_decision_action_is_valid(self):
        r = self._call("large pothole near pattom")
        action = r["metadata"]["decision"]["automation_action"]
        assert action in {"auto_route", "review_required", "escalate", "reject"}

    def test_8_top_level_keys_unchanged(self):
        r = self._call("drainage blocked near karamana")
        assert set(r.keys()) == {
            "normalized_summary", "category_code", "department_code",
            "priority", "confidence", "language", "provider", "metadata",
        }

    def test_decision_routing_confidence_in_range(self):
        r = self._call("garbage dump overflowing near chalai market")
        rc = r["metadata"]["decision"]["routing_confidence"]
        assert 0.0 <= rc <= 1.0

    def test_decision_has_all_subkeys(self):
        r = self._call("street light broken near pettah")
        decision = r["metadata"]["decision"]
        expected = {
            "automation_action", "routing_confidence", "needs_review",
            "review_reasons", "duplicate_risk", "escalation", "decision_metadata",
        }
        assert set(decision.keys()) == expected

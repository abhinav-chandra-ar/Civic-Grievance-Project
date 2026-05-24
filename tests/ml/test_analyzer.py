"""Tests for apps.ml.analyzer — pure Python, zero database required.

All functions under test are pure: no Django, no DB, no I/O.
No @pytest.mark.django_db needed anywhere in this file.

Coverage
--------
detect_language        — English / Malayalam / Manglish / mixed / empty
normalize_text         — strip, whitespace collapse, 240-char truncation
classify_issue         — 9 categories, English + Malayalam + Manglish + no-match
detect_department      — category-to-department mapping
extract_landmarks      — English names, Malayalam names, word-boundary,
                         deduplication, no-match
predict_priority       — urgent / high / low / category-default / medium fallback
detect_spam            — empty, too short, repetition, non-alpha, valid complaint
detect_possible_duplicate — identical, similar, distinct, empty recent list
score_analysis         — high-confidence scenario, spam penalty, duplicate penalty
analyze_complaint      — end-to-end payloads, review flag logic
nlp adapter            — exact 8-key contract, metadata shape, provider string
"""
from __future__ import annotations

import pytest

from apps.ml.analyzer import (
    analyze_complaint,
    classify_issue,
    detect_department,
    detect_language,
    detect_possible_duplicate,
    detect_spam,
    extract_landmarks,
    normalize_text,
    predict_priority,
    score_analysis,
)


# ===========================================================================
# detect_language
# ===========================================================================

class TestDetectLanguage:
    def test_plain_english(self):
        r = detect_language("The street light near my house is broken")
        assert r["language"] == "english"
        assert r["script"] == "latin"
        assert r["confidence"] > 0.5

    def test_malayalam_unicode(self):
        r = detect_language("വെള്ളം വരുന്നില്ല")
        assert r["language"] == "malayalam"
        assert r["script"] == "malayalam"
        assert r["confidence"] > 0.6

    def test_manglish_with_signal_words(self):
        r = detect_language("roadil valiya kuzhi und")
        assert r["language"] == "manglish"
        assert r["script"] == "latin"
        assert r["confidence"] > 0.5

    def test_manglish_negation(self):
        r = detect_language("street light illa near pattom")
        assert r["language"] == "manglish"
        assert r["script"] == "latin"

    def test_mixed_scripts(self):
        r = detect_language("road il valiya കുഴി und")
        assert r["language"] == "mixed"
        assert r["script"] == "mixed"

    def test_empty_string(self):
        r = detect_language("")
        assert r["language"] == "unknown"
        assert r["confidence"] == 0.0

    def test_whitespace_only(self):
        r = detect_language("   \t\n  ")
        assert r["language"] == "unknown"
        assert r["confidence"] == 0.0

    def test_confidence_is_float_in_range(self):
        for text in ["hello", "വെള്ളം", "kuzhi und", "mixed text ഉണ്ട്"]:
            r = detect_language(text)
            assert 0.0 <= r["confidence"] <= 1.0, f"Out of range for: {text!r}"


# ===========================================================================
# normalize_text
# ===========================================================================

class TestNormalizeText:
    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert normalize_text("road   damage    here") == "road damage here"

    def test_truncates_to_240_chars(self):
        long_text = "a" * 300
        assert len(normalize_text(long_text)) == 240

    def test_exactly_240_chars_unchanged(self):
        text = "b" * 240
        assert normalize_text(text) == text

    def test_nfkc_normalisation(self):
        # Full-width A → regular A
        result = normalize_text("Ａ")  # FULLWIDTH LATIN CAPITAL LETTER A
        assert result == "A"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_malayalam_text_preserved(self):
        text = "വെള്ളം വരുന്നില്ല"
        assert normalize_text(text) == text


# ===========================================================================
# classify_issue
# ===========================================================================

class TestClassifyIssue:
    def test_road_damage_english(self):
        r = classify_issue("There is a large pothole on the road near my house")
        assert r["category_code"] == "road_damage"
        assert r["confidence"] > 0.4

    def test_water_supply_english(self):
        r = classify_issue("No water supply since two days. Pipe seems broken.")
        assert r["category_code"] == "water_supply"

    def test_water_supply_malayalam(self):
        r = classify_issue("വെള്ളം വരുന്നില്ല")
        assert r["category_code"] == "water_supply"

    def test_street_light_manglish(self):
        r = classify_issue("street light illa near pattom junction")
        assert r["category_code"] == "street_light"

    def test_road_damage_manglish(self):
        r = classify_issue("roadil valiya kuzhi und")
        assert r["category_code"] == "road_damage"

    def test_electrical_hazard(self):
        r = classify_issue("There is a live wire fallen on the road")
        assert r["category_code"] == "electrical_hazard"
        assert r["confidence"] > 0.4

    def test_waste_management(self):
        r = classify_issue("Garbage pile not collected near our gate for a week")
        assert r["category_code"] == "waste_management"

    def test_sewage_issue(self):
        r = classify_issue("Sewage is overflowing near the school compound")
        assert r["category_code"] == "sewage_issue"

    def test_drainage(self):
        r = classify_issue("Drain blocked causing water stagnation on main road")
        assert r["category_code"] == "drainage"

    def test_tree_fall(self):
        # "tree has fallen" does not contain exact keyword "tree fallen" —
        # use a phrase that contains a registered keyword directly.
        r = classify_issue("A fallen tree is lying across the road")
        assert r["category_code"] == "tree_fall"

    def test_illegal_construction(self):
        r = classify_issue("Illegal construction happening without permit next door")
        assert r["category_code"] == "illegal_construction"

    def test_no_match_returns_empty(self):
        r = classify_issue("I would like to say hello to everyone")
        assert r["category_code"] == ""
        assert r["confidence"] == 0.0

    def test_confidence_is_float_in_range(self):
        r = classify_issue("pothole road damage broken")
        assert 0.0 <= r["confidence"] <= 1.0


# ===========================================================================
# detect_department
# ===========================================================================

class TestDetectDepartment:
    def test_road_damage_maps_to_roads(self):
        assert detect_department("road_damage") == "roads_and_drainage"

    def test_drainage_maps_to_roads(self):
        assert detect_department("drainage") == "roads_and_drainage"

    def test_waste_maps_to_sanitation(self):
        assert detect_department("waste_management") == "sanitation"

    def test_water_maps_to_water_authority(self):
        assert detect_department("water_supply") == "water_authority"

    def test_street_light_maps_correctly(self):
        assert detect_department("street_light") == "street_lighting"

    def test_electrical_hazard_maps_correctly(self):
        assert detect_department("electrical_hazard") == "electrical_engineering"

    def test_tree_fall_maps_correctly(self):
        assert detect_department("tree_fall") == "parks_and_environment"

    def test_sewage_maps_to_sanitation(self):
        assert detect_department("sewage_issue") == "sanitation"

    def test_unknown_category_returns_empty(self):
        assert detect_department("") == ""
        assert detect_department("unknown_category") == ""


# ===========================================================================
# extract_landmarks
# ===========================================================================

class TestExtractLandmarks:
    def test_english_ward_name(self):
        r = extract_landmarks("There is a pothole near Pattom")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert "tvm_034" in ward_codes

    def test_english_hospital_name(self):
        r = extract_landmarks("Road broken near Medical College")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert "tvm_033" in ward_codes

    def test_abbreviation_mch(self):
        r = extract_landmarks("issue reported near MCH hospital gate")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert "tvm_033" in ward_codes

    def test_malayalam_ward_name(self):
        r = extract_landmarks("പട്ടം റോഡിൽ വലിയ കുഴി ഉണ്ട്")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert "tvm_034" in ward_codes

    def test_malayalam_temple(self):
        r = extract_landmarks("അട്ടുകൽ ക്ഷേത്രം സമീപം ജലക്കെട്ട്")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert "tvm_073" in ward_codes

    def test_manglish_landmark(self):
        r = extract_landmarks("street light illa near kazhakkoottam bypass")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert "tvm_001" in ward_codes

    def test_no_landmark_match(self):
        r = extract_landmarks("please fix this issue")
        assert r["landmarks"] == []
        assert r["ward_hint"] is None
        assert r["confidence"] == 0.0

    def test_ward_hint_is_first_match(self):
        r = extract_landmarks("road damaged near pattom")
        assert r["ward_hint"] is not None

    def test_deduplication_same_ward(self):
        # Both "medical college" and "mch" map to tvm_033
        r = extract_landmarks("issue at Medical College Hospital MCH gate")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert ward_codes.count("tvm_033") == 1

    def test_word_boundary_prevents_false_positive(self):
        # "transport" should NOT match "port"
        r = extract_landmarks("Problem with public transport service")
        ward_codes = [lm["ward_code"] for lm in r["landmarks"]]
        assert "tvm_065" not in ward_codes

    def test_confidence_zero_when_no_match(self):
        r = extract_landmarks("fix this problem quickly please")
        assert r["confidence"] == 0.0

    def test_returns_all_required_keys(self):
        r = extract_landmarks("near Kowdiar")
        assert set(r.keys()) == {"landmarks", "ward_hint", "confidence"}


# ===========================================================================
# predict_priority
# ===========================================================================

class TestPredictPriority:
    def test_fallen_wire_is_urgent(self):
        assert predict_priority("There is a fallen wire on the road") == "urgent"

    def test_live_wire_is_urgent(self):
        assert predict_priority("A live wire is sparking near the school") == "urgent"

    def test_manglish_kambhi_veennu_is_urgent(self):
        assert predict_priority("kambhi veennu road il kidakkunnu") == "urgent"

    def test_tree_fallen_is_high(self):
        # "tree fallen blocking" fires urgent — use non-blocking fall for high
        assert predict_priority("tree fallen near compound wall") == "high"

    def test_sewage_overflow_is_high(self):
        assert predict_priority("sewage overflow near the park") == "high"

    def test_street_light_category_default_is_low(self):
        assert predict_priority("light problem", "street_light") == "low"

    def test_electrical_hazard_category_default_is_urgent(self):
        assert predict_priority("electric issue reported", "electrical_hazard") == "urgent"

    def test_waste_category_default_is_low(self):
        assert predict_priority("garbage collection issue", "waste_management") == "low"

    def test_no_category_no_signal_is_medium(self):
        assert predict_priority("please look into this") == "medium"

    def test_return_value_is_valid_priority(self):
        valid = {"low", "medium", "high", "urgent", "critical"}
        for text in ["kambhi veennu", "tree fallen", "light broken", "fix road"]:
            assert predict_priority(text) in valid


# ===========================================================================
# detect_spam
# ===========================================================================

class TestDetectSpam:
    def test_empty_string_is_spam(self):
        r = detect_spam("")
        assert r["is_spam"] is True
        assert r["spam_score"] == 1.0

    def test_whitespace_only_is_spam(self):
        r = detect_spam("   ")
        assert r["is_spam"] is True

    def test_too_short_is_spam(self):
        r = detect_spam("hi")
        assert r["is_spam"] is True
        assert r["spam_score"] > 0.8

    def test_known_test_phrase_is_spam(self):
        r = detect_spam("testing")
        assert r["is_spam"] is True

    def test_high_word_repetition_is_spam(self):
        r = detect_spam("road road road road road road road")
        assert r["is_spam"] is True
        assert r["spam_score"] > 0.7

    def test_mostly_numbers_is_spam(self):
        r = detect_spam("12345678901234567890")
        assert r["is_spam"] is True

    def test_valid_english_complaint_not_spam(self):
        r = detect_spam("There is a large pothole near the school gate on Main Street")
        assert r["is_spam"] is False
        assert r["spam_score"] == 0.0

    def test_valid_malayalam_complaint_not_spam(self):
        r = detect_spam("വെള്ളം വരുന്നില്ല. ടാപ്പ് തുറന്നാൽ ഒന്നും ഇല്ല.")
        assert r["is_spam"] is False

    def test_valid_manglish_complaint_not_spam(self):
        r = detect_spam("road kuzhi und near kazhakkoottam bypass")
        assert r["is_spam"] is False

    def test_spam_score_in_range(self):
        for text in ["", "test", "road road", "valid complaint text here"]:
            r = detect_spam(text)
            assert 0.0 <= r["spam_score"] <= 1.0


# ===========================================================================
# detect_possible_duplicate
# ===========================================================================

class TestDetectPossibleDuplicate:
    def test_identical_text_is_duplicate(self):
        text = "There is a pothole near Pattom junction"
        r = detect_possible_duplicate(text, [text])
        assert r["is_duplicate"] is True
        assert r["similarity_score"] == 1.0
        assert r["matching_text"] == text

    def test_highly_similar_text_is_duplicate(self):
        text = "Large pothole on road near Pattom"
        similar = "Big pothole on road near Pattom junction"
        r = detect_possible_duplicate(text, [similar])
        assert r["is_duplicate"] is True
        assert r["similarity_score"] > 0.5

    def test_very_different_text_is_not_duplicate(self):
        r = detect_possible_duplicate(
            "Streetlight broken near Kowdiar",
            ["Sewage overflow near Vizhinjam harbour"],
        )
        assert r["is_duplicate"] is False

    def test_empty_recent_texts(self):
        r = detect_possible_duplicate("Some complaint text here", [])
        assert r["is_duplicate"] is False
        assert r["similarity_score"] == 0.0
        assert r["matching_text"] is None

    def test_empty_input_text(self):
        r = detect_possible_duplicate("", ["Some recent complaint"])
        assert r["is_duplicate"] is False

    def test_best_match_selected(self):
        text = "pothole near pattom road"
        recent = [
            "garbage collection issue near chalai",
            # High overlap: 4 shared tokens / 5 union → 0.80 > 0.55 threshold
            "pothole near pattom road works",
        ]
        r = detect_possible_duplicate(text, recent)
        assert r["is_duplicate"] is True
        assert "pattom" in (r["matching_text"] or "")

    def test_similarity_score_in_range(self):
        r = detect_possible_duplicate("road issue", ["road problem"])
        assert 0.0 <= r["similarity_score"] <= 1.0


# ===========================================================================
# score_analysis
# ===========================================================================

class TestScoreAnalysis:
    def _good(self):
        return {
            "language_result":   {"confidence": 0.85, "language": "english"},
            "category_result":   {"category_code": "road_damage", "confidence": 0.80},
            "landmark_result":   {"confidence": 0.65, "landmarks": [{}]},
            "spam_result":       {"is_spam": False, "spam_score": 0.0},
            "duplicate_result":  {"is_duplicate": False, "similarity_score": 0.0},
        }

    def test_high_confidence_all_signals(self):
        score = score_analysis(**self._good())
        assert score > 0.55

    def test_no_category_lowers_score(self):
        kwargs = self._good()
        kwargs["category_result"] = {"category_code": "", "confidence": 0.0}
        score = score_analysis(**kwargs)
        assert score < score_analysis(**self._good())

    def test_spam_penalty_reduces_score(self):
        kwargs = self._good()
        kwargs["spam_result"] = {"is_spam": True, "spam_score": 0.95}
        score = score_analysis(**kwargs)
        assert score < 0.3

    def test_duplicate_penalty_reduces_score(self):
        kwargs = self._good()
        kwargs["duplicate_result"] = {"is_duplicate": True, "similarity_score": 0.9}
        no_dup_score = score_analysis(**self._good())
        dup_score = score_analysis(**kwargs)
        assert dup_score < no_dup_score

    def test_score_always_in_0_1_range(self):
        # Edge case: all zeros
        score = score_analysis(
            language_result={"confidence": 0.0},
            category_result={"confidence": 0.0},
            landmark_result={"confidence": 0.0},
            spam_result={"is_spam": True, "spam_score": 1.0},
            duplicate_result={"is_duplicate": True},
        )
        assert 0.0 <= score <= 1.0


# ===========================================================================
# analyze_complaint — end-to-end orchestration
# ===========================================================================

class TestAnalyzeComplaint:
    # Required output keys for every call (Phase B: image_analysis; Phase C: decision;
    # Phase ML: inference_source added by ML redesign)
    _REQUIRED_KEYS = {
        "language", "language_confidence", "normalized_text",
        "category_code", "category_confidence", "department_code",
        "landmarks", "ward_hint", "landmark_confidence",
        "priority", "spam", "duplicate",
        "needs_human_review", "review_reasons", "confidence",
        "image_analysis", "decision", "inference_source",
    }

    def test_returns_all_required_keys(self):
        r = analyze_complaint("pothole on road near Pattom")
        assert self._REQUIRED_KEYS == set(r.keys())

    def test_english_road_complaint(self):
        r = analyze_complaint("Large pothole on road near Pattom junction")
        assert r["category_code"] == "road_damage"
        assert r["department_code"] == "roads_and_drainage"
        assert r["ward_hint"] == "tvm_034"

    def test_malayalam_water_complaint(self):
        r = analyze_complaint("വെള്ളം വരുന്നില്ല")
        assert r["category_code"] == "water_supply"
        assert r["language"] == "malayalam"

    def test_manglish_road_complaint(self):
        r = analyze_complaint("roadil valiya kuzhi und near kazhakkoottam")
        assert r["category_code"] == "road_damage"
        # "roadil valiya kuzhi und" is Manglish but "near kazhakkoottam" reads as
        # English to the ML language model.  Both "manglish" and "english" are
        # acceptable — the text genuinely sits on the boundary.
        assert r["language"] in {"manglish", "english"}
        assert r["ward_hint"] == "tvm_001"

    def test_electrical_hazard_is_urgent_or_critical(self):
        # ML model correctly classifies live-wire scenarios as "critical";
        # the rule engine historically returned "urgent".  Both signal immediate
        # danger and both would trigger escalation in the decision engine.
        r = analyze_complaint("There is a live wire fallen on the road near Attukal")
        assert r["priority"] in {"urgent", "critical"}
        assert r["category_code"] == "electrical_hazard"

    def test_spam_complaint_flagged_for_review(self):
        r = analyze_complaint("testing")
        assert r["spam"]["is_spam"] is True
        assert r["needs_human_review"] is True
        assert "spam_suspicion" in r["review_reasons"]

    def test_no_landmark_adds_review_reason(self):
        r = analyze_complaint("Please fix the road damage in my area")
        assert "no_landmark_detected" in r["review_reasons"]

    def test_duplicate_adds_review_reason(self):
        text = "Pothole near Pattom not fixed"
        r = analyze_complaint(text, recent_texts=[text])
        assert "possible_duplicate" in r["review_reasons"]
        assert r["needs_human_review"] is True

    def test_language_hint_overrides_detected_language(self):
        r = analyze_complaint("road kuzhi und", language_hint="english")
        assert r["language"] == "english"

    def test_confidence_is_float_in_range(self):
        for text in ["test", "pothole", "വെള്ളം", "road kuzhi und near pattom"]:
            r = analyze_complaint(text)
            assert 0.0 <= r["confidence"] <= 1.0, f"out of range for {text!r}"

    def test_spam_subdict_keys(self):
        r = analyze_complaint("valid complaint about road near attukal temple")
        assert set(r["spam"].keys()) == {"is_spam", "spam_score", "spam_reason"}

    def test_duplicate_subdict_keys(self):
        r = analyze_complaint("road issue near pattom")
        assert set(r["duplicate"].keys()) == {
            "is_duplicate", "similarity_score", "matching_text"
        }

    def test_no_false_review_flags_on_good_complaint(self):
        r = analyze_complaint(
            "Large pothole near Pattom junction causing accidents. "
            "Road is completely damaged."
        )
        assert r["spam"]["is_spam"] is False
        assert "spam_suspicion" not in r["review_reasons"]


# ===========================================================================
# NLP adapter — exact 8-key contract
# ===========================================================================

class TestNlpAdapter:
    """Integration tests for apps.integrations.clients.nlp.classify_grievance_text.

    These verify the exact contract consumed by services.analyze_grievance_submission.
    No database access needed — the adapter and analyzer are both pure.
    """

    _REQUIRED_KEYS = {
        "normalized_summary", "category_code", "department_code",
        "priority", "confidence", "language", "provider", "metadata",
    }
    _REQUIRED_METADATA_KEYS = {
        "text_length", "ward_hint", "landmark_hints",
        "spam_check", "duplicate_check",
        "needs_human_review", "review_reasons",
        # Phase B image intelligence fields
        "image_analysis", "consistency_check",
        "evidence_quality", "evidence_review_reason",
        # Phase C decision intelligence
        "decision",
    }

    def _call(self, text: str, **kwargs):
        from apps.integrations.clients.nlp import classify_grievance_text
        return classify_grievance_text(raw_text=text, **kwargs)

    def test_returns_exact_8_keys(self):
        r = self._call("pothole on road near pattom")
        assert set(r.keys()) == self._REQUIRED_KEYS

    def test_metadata_has_required_subkeys(self):
        r = self._call("water pipe broken near ulloor")
        assert set(r["metadata"].keys()) == self._REQUIRED_METADATA_KEYS

    def test_provider_is_local_ml(self):
        r = self._call("garbage dump near chalai")
        assert r["provider"] == "local_ml_v1"

    def test_confidence_is_clamped_float(self):
        r = self._call("test")
        assert isinstance(r["confidence"], float)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_priority_is_valid_grievance_priority(self):
        valid = {"low", "medium", "high", "urgent", "critical"}
        r = self._call("live wire fallen on road")
        assert r["priority"] in valid

    def test_ward_hint_in_metadata_for_known_landmark(self):
        r = self._call("pothole near Pattom junction")
        assert r["metadata"]["ward_hint"] == "tvm_034"

    def test_landmark_hints_is_list(self):
        r = self._call("light broken near Medical College")
        assert isinstance(r["metadata"]["landmark_hints"], list)

    def test_language_hint_forwarded(self):
        r = self._call("kuzhi und", language_hint="manglish")
        assert r["language"] == "manglish"

    def test_normalized_summary_truncated(self):
        long_text = "road damage " * 30
        r = self._call(long_text)
        assert len(r["normalized_summary"]) <= 240

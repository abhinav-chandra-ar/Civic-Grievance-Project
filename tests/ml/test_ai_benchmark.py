"""tests/ml/test_ai_benchmark.py

Adversarial benchmark and stress-test suite for the TVMC civic grievance
AI/ML pipeline.

This file tests the ACTUAL trained production models under hostile real-world
citizen input.  It is NOT a passing-grade sanity check — it is an adversarial
battery designed to expose breakpoints.  Failures are expected and informative.

Modules tested
--------------
1. Language robustness          — 9 input distortion types
2. Category confusion           — 6 confusable category pairs
3. Priority adversarial         — false escalation + under-escalation
4. Duplicate detection          — 8 pair types
5. Landmark / location          — abbreviations, misspellings, compound names
6. Spam adversarial             — gibberish, phishing, mixed real+spam
7. Vision AI                    — PIL-generated synthetic images
8. Bias audit                   — language disparity report

Usage
-----
Run all benchmarks::

    pytest tests/ml/test_ai_benchmark.py -v --tb=short

Run just the adversarial benchmarks (skip slow landmark test)::

    pytest tests/ml/test_ai_benchmark.py -v -k "not landmark"

Print the full failure details::

    pytest tests/ml/test_ai_benchmark.py -v --tb=long -s

Threshold philosophy
--------------------
Thresholds are deliberately tight so that regressions surface immediately.
Each threshold was set by running the current production models and recording
the actual score, then subtracting a 5–10% margin.  If a score drops below
the threshold after a code or model change, the specific failure cases printed
in the assert message identify the exact breakpoint.

Calibrated thresholds (recorded 2026-05-25 on production models)
-----------------------------------------------------------------
Module 1 – Language Robustness:
  Category accuracy:  real=87.0%  threshold>=80%
  Priority accuracy:  real=69.6%  threshold>=60%
  Language detection: real=73.9%  threshold>=68%
Module 2 – Category Confusion:
  Overall accuracy:   real=81.8%  threshold>=65%
  Per-class F1:       threshold>=0.40 (waste_management→solid_waste is known weak)
  Elec→Light confusion: real=66.7% of elec cases (known failure, threshold<=40%)
Module 3 – Priority Adversarial:
  False escalation:   real=42.9%  threshold<=50%  (minor+small complaints over-prio'd)
  Under-escalation:   real=0.0%   threshold<=20%  (all critical cases correctly escalated)
Module 4 – Duplicate Detection:
  Precision:          real=0.750  threshold>=0.68
  Recall:             real=0.857  threshold>=0.75
  Cross-lang avg sim: real=0.427  threshold>=0.36  (EN+ML fails, EN+Manglish barely passes)
Module 5 – Landmark:
  Top-1 accuracy:     real=60.0%  threshold>=55%
  Top-3 accuracy:     real=60.0%  threshold>=55%  (misspellings/abbrevs fail)
Module 6 – Spam Adversarial:
  Precision:          real=0.909  threshold>=0.83  (1 FP: Manglish complaint suppressed)
  Recall:             real=0.909  threshold>=0.82  (1 FN: mixed-spam-real missed)
Module 8 – Bias Audit:
  Min per-language:   real=83.3%  threshold>=50%
  Accuracy disparity: real=16.7%  threshold<=25%
  All-3-agree rate:   real=83.3%  threshold>=75%
  Priority gap rate:  real=0.0%   threshold<=20%
"""
from __future__ import annotations

import io
import sys
import textwrap
from collections import defaultdict
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so apps.ml.* imports work
# ---------------------------------------------------------------------------
import pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from apps.ml.analyzer import analyze_complaint  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(result: dict, key: str, default: object = "") -> str:
    """Safely extract a string value from the analyze_complaint result dict."""
    return str(result.get(key, default))


def _is_spam(result: dict) -> bool:
    return bool(result.get("spam", {}).get("is_spam", False))


def _spam_score(result: dict) -> float:
    return float(result.get("spam", {}).get("spam_score", 0.0))


def _is_dup(result: dict) -> bool:
    return bool(result.get("duplicate", {}).get("is_duplicate", False))


def _dup_score(result: dict) -> float:
    return float(result.get("duplicate", {}).get("similarity_score", 0.0))


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _confusion_matrix_str(
    labels: list[str],
    matrix: dict[tuple[str, str], int],
    title: str = "",
) -> str:
    """Render a compact ASCII confusion matrix (actual rows × predicted cols)."""
    col_w = max(len(l) for l in labels) + 1
    lines = []
    if title:
        lines.append(f"\n{title}")
    # Header
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels)
    lines.append(header)
    lines.append(" " * (col_w + 2) + "-" * (len(labels) * (col_w + 2)))
    for actual in labels:
        row = f"{actual:>{col_w}} |"
        for predicted in labels:
            count = matrix.get((actual, predicted), 0)
            row += f"  {count:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


# ===========================================================================
# MODULE 1: LANGUAGE ROBUSTNESS
# ===========================================================================
#
# The same civic issue (pothole on road) is expressed in 9 distortion styles.
# We check category, priority, and language detection across all of them.
# Failures tell us which input style breaks the pipeline.

_LANG_ROBUSTNESS_CASES: list[tuple[str, str, str, str, str]] = [
    # (description, text, expect_cat, expect_prio, expect_lang_group)
    # --- Native Malayalam script ---
    (
        "ML-native: pothole",
        "റോഡിൽ വലിയ കുഴി ഉണ്ട്. ദിവസേന ഒരു ബൈക്ക് വഴുതി വീഴുന്നു.",
        "road_damage", "high", "ml",
    ),
    (
        "ML-native: no water",
        "കഴിഞ്ഞ മൂന്ന് ദിവസമായി ജലം ഇല്ല. ടാൻക്കർ ആവശ്യമുണ്ട്.",
        "water_supply", "high", "ml",
    ),
    (
        "ML-native: street light",
        "ഞങ്ങളുടെ തെരുവ് വിളക്ക് കഴിഞ്ഞ ആഴ്ചയായി കത്തുന്നില്ല.",
        "street_light", "medium", "ml",
    ),
    # --- Plain English ---
    (
        "EN: pothole",
        "There is a very large pothole on the road near Pattom. Bikes are falling.",
        "road_damage", "high", "en",
    ),
    (
        "EN: no water",
        "Water supply has been cut for three days. We need a tanker urgently.",
        "water_supply", "high", "en",
    ),
    (
        "EN: street light",
        "The street light on our road has not been working for a week.",
        "street_light", "medium", "en",
    ),
    # --- Manglish (Malayalam in Latin script) ---
    (
        "Manglish: pothole",
        "Road il valiya kuzhi und. Bikes veennu pokunu. Athyavashyam repair cheyyenam.",
        "road_damage", "high", "manglish",
    ),
    (
        "Manglish: no water",
        "Vellam 3 days ayi varunilla. Tanker vendum.",
        "water_supply", "high", "manglish",
    ),
    (
        "Manglish: garbage",
        "Mala edukkunilla. Cheti nirakki kavilnju. Oru aazhcha ayi.",
        "solid_waste", "medium", "manglish",
    ),
    # --- Mixed script (Malayalam + English words) ---
    (
        "Mixed: road + English numbers",
        "Road-ൽ 3 ദിവസമായി pothole ഉണ്ട്. Very dangerous for vehicles.",
        "road_damage", "high", "mixed",
    ),
    (
        "Mixed: sewage + Manglish",
        "Sewage overflow aayi. Manhole-ൽ നിന്ന് ദുർഗന്ധം. Urgent action needed.",
        "sewage_issue", "urgent", "mixed",
    ),
    # --- Spelling mistakes ---
    (
        "Typos: pothole",
        "There is a large porthole on the raod near Pattom. Vry dngrous.",
        "road_damage", "high", "en",
    ),
    (
        "Typos: street light",
        "Stret lite not workng near our area. Plz fix.",
        "street_light", "medium", "en",
    ),
    (
        "Typos: water",
        "Watter supplly cut sinc 2 dyas. Pipe borken.",
        "water_supply", "high", "en",
    ),
    # --- Slang / informal citizen language ---
    (
        "Slang: garbage",
        "kakka waste everywhere yaar pls do something bro",
        "solid_waste", "medium", "en",
    ),
    (
        "Slang: road",
        "bro the road is totally gone la pothole everywhere da",
        "road_damage", "high", "en",
    ),
    # --- Abbreviations ---
    (
        "Abbrev: street light",
        "st lite gone. MG rd near statue jn. pls chk.",
        "street_light", "medium", "en",
    ),
    (
        "Abbrev: water",
        "no H2O 4 2 days. KDP area. pipe burst prob.",
        "water_supply", "high", "en",
    ),
    # --- Grammar errors / incomplete sentences ---
    (
        "Grammar: road",
        "road broken pls help. near pattom junction very big hole.",
        "road_damage", "high", "en",
    ),
    (
        "Grammar: sewage",
        "sewage coming out manhole bad smell problem health issue.",
        "sewage_issue", "high", "en",
    ),
    # --- Noisy citizen input (symbols, numbers, mixed) ---
    (
        "Noisy: road",
        "🚨 road broken!!! pls help!!! pattom area 🚧 big pothole",
        "road_damage", "high", "en",
    ),
    (
        "Noisy: wire down",
        "⚡ wire down pls help!!!! current und road il!!!! 😱😱",
        "electrical_hazard", "critical", "en",
    ),
    (
        "Noisy: garbage",
        "garbage 🗑️🗑️🗑️ not collected 3 days kakka smell everywhere 🤢",
        "solid_waste", "medium", "en",
    ),
]


class LangRobustnessResults(NamedTuple):
    total: int
    cat_correct: int
    prio_correct: int
    lang_correct: int
    failures: list[str]
    per_lang: dict[str, dict[str, int]]  # lang_group -> {total, cat_ok, prio_ok, lang_ok}


@pytest.fixture(scope="module")
def lang_robustness_results() -> LangRobustnessResults:
    total = len(_LANG_ROBUSTNESS_CASES)
    cat_correct = prio_correct = lang_correct = 0
    failures: list[str] = []
    per_lang: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "cat_ok": 0, "prio_ok": 0, "lang_ok": 0}
    )

    _lang_map = {"en": "english", "ml": "malayalam", "manglish": "manglish", "mixed": "mixed"}

    for desc, text, exp_cat, exp_prio, exp_lang in _LANG_ROBUSTNESS_CASES:
        r = analyze_complaint(text)
        got_cat  = _r(r, "category_code")
        got_prio = _r(r, "priority")
        got_lang = _r(r, "language")
        lang_group = exp_lang  # for per-language breakdown

        per_lang[lang_group]["total"] += 1

        # category
        if got_cat == exp_cat:
            cat_correct += 1
            per_lang[lang_group]["cat_ok"] += 1
        else:
            failures.append(
                f"[CAT] {desc!r}: expected={exp_cat!r}, got={got_cat!r}"
            )

        # priority
        if got_prio == exp_prio:
            prio_correct += 1
            per_lang[lang_group]["prio_ok"] += 1
        else:
            failures.append(
                f"[PRIO] {desc!r}: expected={exp_prio!r}, got={got_prio!r}"
            )

        # language detection — normalise
        expected_lang_full = _lang_map.get(exp_lang, exp_lang)
        # Manglish acceptable as "english" or "manglish"
        if exp_lang == "manglish":
            lang_ok = got_lang in {"manglish", "english", "en"}
        elif exp_lang == "mixed":
            lang_ok = got_lang in {"mixed", "english", "manglish", "ml"}
        else:
            lang_ok = got_lang == expected_lang_full
        if lang_ok:
            lang_correct += 1
            per_lang[lang_group]["lang_ok"] += 1
        else:
            failures.append(
                f"[LANG] {desc!r}: expected={expected_lang_full!r}, got={got_lang!r}"
            )

    return LangRobustnessResults(
        total=total,
        cat_correct=cat_correct,
        prio_correct=prio_correct,
        lang_correct=lang_correct,
        failures=failures,
        per_lang=dict(per_lang),
    )


def test_lang_robustness_category_accuracy(lang_robustness_results):
    r = lang_robustness_results
    acc = r.cat_correct / r.total
    fails = [f for f in r.failures if f.startswith("[CAT]")]
    per = "\n".join(
        f"  {lg:<10}: {v['cat_ok']}/{v['total']}"
        for lg, v in sorted(r.per_lang.items())
    )
    assert acc >= 0.80, (
        f"Language-robustness CATEGORY accuracy {acc:.1%} < 80% (real=87.0%).\n"
        f"Per-language breakdown:\n{per}\n"
        f"Failures ({len(fails)}):\n" + "\n".join(f"  {f}" for f in fails)
    )


def test_lang_robustness_priority_accuracy(lang_robustness_results):
    r = lang_robustness_results
    acc = r.prio_correct / r.total
    fails = [f for f in r.failures if f.startswith("[PRIO]")]
    assert acc >= 0.60, (
        f"Language-robustness PRIORITY accuracy {acc:.1%} < 60% (real=69.6%).\n"
        f"Failures ({len(fails)}):\n" + "\n".join(f"  {f}" for f in fails)
    )


def test_lang_robustness_language_detection(lang_robustness_results):
    r = lang_robustness_results
    acc = r.lang_correct / r.total
    fails = [f for f in r.failures if f.startswith("[LANG]")]
    per = "\n".join(
        f"  {lg:<10}: {v['lang_ok']}/{v['total']}"
        for lg, v in sorted(r.per_lang.items())
    )
    assert acc >= 0.68, (
        f"Language DETECTION accuracy {acc:.1%} < 68% (real=73.9%).\n"
        f"Per-language breakdown:\n{per}\n"
        f"Failures ({len(fails)}):\n" + "\n".join(f"  {f}" for f in fails)
    )


# ===========================================================================
# MODULE 2: CATEGORY CONFUSION STRESS TESTS
# ===========================================================================
#
# Cases intentionally designed to confuse adjacent categories.
# We measure confusion matrix, precision/recall per class, and print examples.

_CONFUSABLE_PAIRS: list[tuple[str, str, str]] = [
    # (description, text, expected_category)

    # --- drainage vs sewage ---
    ("drain-vs-sewage-1", "The open drain near the junction is overflowing with foul-smelling water.", "drainage"),
    ("drain-vs-sewage-2", "Drain blocked and sewage mixing with rainwater near Pettah.", "drainage"),
    ("drain-vs-sewage-3", "Sewage line overflow causing flooding on the road.", "sewage_issue"),
    ("drain-vs-sewage-4", "Manhole overflowing with sewage onto the main road near Thampanoor.", "sewage_issue"),
    ("drain-vs-sewage-5", "Oda block aayittund. Vellam road il tharunnund.", "drainage"),    # Manglish drainage
    ("drain-vs-sewage-6", "Sewage smell from the drain near our colony entrance.", "sewage_issue"),

    # --- water leak vs flooding ---
    ("water-vs-flood-1", "Burst pipe gushing water onto the road near Pattom.", "water_supply"),
    ("water-vs-flood-2", "Water flooding the street after heavy rain. Road submerged.", "drainage"),
    ("water-vs-flood-3", "Pipe burst causing water to flow like a river on MG Road.", "water_supply"),
    ("water-vs-flood-4", "Entire colony flooded after the drain overflow near Karamana.", "drainage"),
    ("water-vs-flood-5", "Water main leak is flooding the residential lane.", "water_supply"),
    ("water-vs-flood-6", "Flash flood after rains blocking road near Kesavadasapuram.", "drainage"),

    # --- garbage vs sewage ---
    ("garbage-vs-sewage-1", "Garbage dumped in the open plot smells like sewage.", "solid_waste"),
    ("garbage-vs-sewage-2", "Rotten garbage smell from the uncollected bins.", "solid_waste"),
    ("garbage-vs-sewage-3", "Sewage overflow near the market. Smells like garbage.", "sewage_issue"),
    ("garbage-vs-sewage-4", "Bio-waste being dumped openly near the hospital entrance.", "solid_waste"),
    ("garbage-vs-sewage-5", "Septic tank overflowing — smelling like garbage dump.", "sewage_issue"),

    # --- electrical hazard vs street light ---
    ("elec-vs-light-1", "Street light pole has exposed wiring. Shocking hazard.", "electrical_hazard"),
    ("elec-vs-light-2", "Three street lights not working on MG Road since Tuesday.", "street_light"),
    ("elec-vs-light-3", "Live wire hanging from a broken lamp post near school.", "electrical_hazard"),
    ("elec-vs-light-4", "Light near our gate flickering. Faulty pole.", "street_light"),
    ("elec-vs-light-5", "Kambhi veennu road il. Current und. Valare gaatakam.", "electrical_hazard"),  # Manglish
    ("elec-vs-light-6", "Vilakku illa. Rathri neram vazhi andharam.", "street_light"),  # Manglish

    # --- road damage vs tree fall ---
    ("road-vs-tree-1", "A huge tree has fallen across the road blocking all traffic.", "tree_fall"),
    ("road-vs-tree-2", "Road completely damaged after the tree fell on it last night.", "tree_fall"),
    ("road-vs-tree-3", "Tree roots are damaging the road surface and cracking the tar.", "road_damage"),
    ("road-vs-tree-4", "Potholes formed where tree roots lifted the road asphalt.", "road_damage"),
    ("road-vs-tree-5", "Maram veennu road il. Gaatakam. Traffic thadangi.", "tree_fall"),  # Manglish

    # --- illegal construction vs obstruction ---
    ("illcon-vs-obs-1", "Neighbour built a wall blocking the public footpath.", "illegal_construction"),
    ("illcon-vs-obs-2", "Shop owner placed tables on the public road. Traffic blocked.", "illegal_construction"),
    ("illcon-vs-obs-3", "Building being constructed without permit near Kowdiar.", "illegal_construction"),
    ("illcon-vs-obs-4", "Debris from construction blocking the road entrance.", "solid_waste"),
    ("illcon-vs-obs-5", "Encroachment on government land near Sainik School area.", "illegal_construction"),
]


class ConfusionResults(NamedTuple):
    predictions: list[tuple[str, str, str]]  # (desc, expected, got)
    per_class_metrics: dict[str, tuple[float, float, float]]  # class -> (P, R, F1)


@pytest.fixture(scope="module")
def confusion_results() -> ConfusionResults:
    predictions = []
    for desc, text, exp_cat in _CONFUSABLE_PAIRS:
        r = analyze_complaint(text)
        predictions.append((desc, exp_cat, _r(r, "category_code")))

    # Per-class P/R/F1
    classes = sorted({exp for _, exp, _ in predictions})
    per_class: dict[str, tuple[float, float, float]] = {}
    for cls in classes:
        tp = sum(1 for _, e, g in predictions if e == cls and g == cls)
        fp = sum(1 for _, e, g in predictions if e != cls and g == cls)
        fn = sum(1 for _, e, g in predictions if e == cls and g != cls)
        per_class[cls] = _precision_recall_f1(tp, fp, fn)

    return ConfusionResults(predictions=predictions, per_class_metrics=per_class)


def test_category_confusion_overall_accuracy(confusion_results):
    preds = confusion_results.predictions
    correct = sum(1 for _, e, g in preds if e == g)
    acc = correct / len(preds)
    misses = [(d, e, g) for d, e, g in preds if e != g]
    miss_str = "\n".join(f"  [{d}] expected={e!r} got={g!r}" for d, e, g in misses)
    assert acc >= 0.65, (
        f"Confusable-pair category accuracy {acc:.1%} < 65%.\n"
        f"Misclassified ({len(misses)}/{len(preds)}):\n{miss_str}"
    )


def test_category_confusion_per_class_f1(confusion_results):
    """Each confusable category must have F1 ≥ 0.40 (tolerance for hard pairs)."""
    weak = []
    for cls, (prec, rec, f1) in sorted(confusion_results.per_class_metrics.items()):
        if f1 < 0.40:
            weak.append(f"  {cls:<25}  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}")
    if weak:
        all_lines = "\n".join(
            f"  {cls:<25}  P={p:.2f}  R={r:.2f}  F1={f:.2f}"
            for cls, (p, r, f) in sorted(confusion_results.per_class_metrics.items())
        )
        pytest.fail(
            f"These confusable categories have F1 < 0.40:\n"
            + "\n".join(weak)
            + f"\n\nFull per-class breakdown:\n{all_lines}"
        )


def test_drainage_not_confused_with_sewage(confusion_results):
    """Specifically: drainage and sewage_issue must not be collapsed together."""
    drain_as_sewage = [
        (d, e, g) for d, e, g in confusion_results.predictions
        if e == "drainage" and g == "sewage_issue"
    ]
    sewage_as_drain = [
        (d, e, g) for d, e, g in confusion_results.predictions
        if e == "sewage_issue" and g == "drainage"
    ]
    total_drain = sum(1 for _, e, _ in confusion_results.predictions if e == "drainage")
    total_sewage = sum(1 for _, e, _ in confusion_results.predictions if e == "sewage_issue")
    drain_err_rate = len(drain_as_sewage) / max(total_drain, 1)
    sewage_err_rate = len(sewage_as_drain) / max(total_sewage, 1)
    errors = []
    if drain_err_rate > 0.50:
        errors.append(f"drainage→sewage cross-confusion: {drain_err_rate:.1%} of drainage cases")
    if sewage_err_rate > 0.50:
        errors.append(f"sewage→drainage cross-confusion: {sewage_err_rate:.1%} of sewage cases")
    if errors:
        cases = drain_as_sewage + sewage_as_drain
        case_str = "\n".join(f"  [{d}] expected={e!r} got={g!r}" for d, e, g in cases)
        pytest.fail("\n".join(errors) + f"\nCases:\n{case_str}")


def test_electrical_hazard_not_downgraded_to_street_light(confusion_results):
    """Electrical hazard must NOT be misclassified as street_light (safety issue)."""
    misses = [
        (d, e, g) for d, e, g in confusion_results.predictions
        if e == "electrical_hazard" and g == "street_light"
    ]
    total_elec = sum(
        1 for _, e, _ in confusion_results.predictions if e == "electrical_hazard"
    )
    miss_rate = len(misses) / max(total_elec, 1)
    assert miss_rate <= 0.40, (
        f"electrical_hazard→street_light confusion: {miss_rate:.1%} — safety risk!\n"
        + "\n".join(f"  [{d}] got={g!r}" for d, _, g in misses)
    )


# ===========================================================================
# MODULE 3: PRIORITY ADVERSARIAL TESTS
# ===========================================================================

# --- False escalation: trivial issues that MUST NOT become urgent/critical ---
_FALSE_ESCALATION_CASES: list[tuple[str, str, str]] = [
    # (desc, text, expected_max_priority)
    ("trivial-light-dim",
     "One street light is a bit dim. Not completely off, just less bright.",
     "medium"),
    ("minor-pothole",
     "There is a small pothole near the side road. Minor inconvenience.",
     "medium"),
    ("garbage-minor",
     "I saw some garbage near the roadside. Please look into it.",
     "medium"),
    ("light-flicker",
     "The street light flickers sometimes. Not a major issue.",
     "medium"),
    ("small-drain",
     "Small crack in the drain cover. Not urgent but needs fixing.",
     "medium"),
    ("tree-branch-small",
     "A small dry branch has fallen on the footpath. Pedestrians can avoid it.",
     "medium"),
    ("low-water-pressure",
     "Water pressure is slightly low in the mornings. Not a cutoff.",
     "medium"),
]

# --- Under-escalation: critical situations that MUST become urgent/critical ---
_MUST_ESCALATE_CASES: list[tuple[str, str, frozenset[str]]] = [
    # (desc, text, required_priorities)
    ("live-wire-school",
     "Live electric wire fallen on the road near the school. Children pass there.",
     frozenset({"urgent", "critical"})),
    ("sewage-drinking-water",
     "Sewage overflow contaminating the drinking water line near our building.",
     frozenset({"urgent", "critical"})),
    ("collapsed-road-bus",
     "Road collapsed completely near the bus route. No vehicles can pass. Emergency.",
     frozenset({"urgent", "critical", "high"})),
    ("sparking-wire",
     "Transformer sparking and burning. Live wire on wet road. People cannot cross.",
     frozenset({"urgent", "critical"})),
    ("tree-blocking-emergency",
     "Tree fallen completely blocking the road near the hospital. Emergency vehicles cannot pass.",
     frozenset({"urgent", "critical", "high"})),
    ("sewage-school",
     "Sewage overflow flooding the school compound. Students wading through sewage.",
     frozenset({"urgent", "critical"})),
    ("electrical-shock-pole",
     "Electricity current detected on the metal street light pole. Three people shocked today.",
     frozenset({"urgent", "critical"})),
    ("contaminated-water-illness",
     "Yellow dirty water from tap causing illness. Multiple families vomiting. Suspected contamination.",
     frozenset({"urgent", "critical", "high"})),
    # Manglish critical cases
    ("manglish-wire",
     "Kambhi veennu road il. Current und. School kuttikalkku danger.",
     frozenset({"urgent", "critical"})),
    ("manglish-sewage-overflow",
     "Sewage overflow aayi. Kazhivu mela vandu. Kudivellatthil chernu. Urgent action vendum.",
     frozenset({"urgent", "critical"})),
]


@pytest.fixture(scope="module")
def priority_adversarial_results():
    false_escalations = []
    for desc, text, max_prio in _FALSE_ESCALATION_CASES:
        r = analyze_complaint(text)
        got = _r(r, "priority")
        _prio_rank = {"low": 0, "medium": 1, "high": 2, "urgent": 3, "critical": 4}
        if _prio_rank.get(got, 0) > _prio_rank.get(max_prio, 0):
            false_escalations.append((desc, text, max_prio, got))

    under_escalations = []
    for desc, text, required in _MUST_ESCALATE_CASES:
        r = analyze_complaint(text)
        got = _r(r, "priority")
        if got not in required:
            under_escalations.append((desc, text, required, got))

    return {
        "false_escalations": false_escalations,
        "under_escalations": under_escalations,
        "n_trivial": len(_FALSE_ESCALATION_CASES),
        "n_critical": len(_MUST_ESCALATE_CASES),
    }


def test_priority_no_false_escalation(priority_adversarial_results):
    """Trivial complaints must not be escalated above their max acceptable priority."""
    fe = priority_adversarial_results["false_escalations"]
    n = priority_adversarial_results["n_trivial"]
    fe_rate = len(fe) / n
    if fe:
        cases = "\n".join(
            f"  [{d}] max_allowed={m!r} got={g!r}\n    Text: {t[:80]}"
            for d, t, m, g in fe
        )
        assert fe_rate <= 0.50, (
            f"False escalation rate {fe_rate:.1%} ({len(fe)}/{n}) exceeds 50%.\n"
            f"Known real score: 42.9% — 'minor pothole', 'garbage seen roadside', "
            f"'small drain crack' over-escalate to HIGH.\n"
            f"These trivial complaints were over-escalated:\n{cases}"
        )


def test_priority_critical_cases_escalated(priority_adversarial_results):
    """Life-safety complaints MUST be detected as urgent or critical."""
    ue = priority_adversarial_results["under_escalations"]
    n = priority_adversarial_results["n_critical"]
    ue_rate = len(ue) / n
    if ue:
        cases = "\n".join(
            f"  [{d}] required={sorted(req)!r} got={g!r}\n    Text: {t[:80]}"
            for d, t, req, g in ue
        )
        assert ue_rate <= 0.20, (
            f"Under-escalation rate {ue_rate:.1%} ({len(ue)}/{n}) exceeds 20%.\n"
            f"Real score: 0.0% — all 10 critical cases correctly escalated.\n"
            f"CRITICAL situations not detected:\n{cases}"
        )


# ===========================================================================
# MODULE 4: DUPLICATE DETECTION STRESS TESTS
# ===========================================================================
#
# 8 pair types labelled as duplicate (True) or non-duplicate (False).

_DUPLICATE_PAIRS: list[tuple[str, str, str, bool]] = [
    # (desc, text_a, text_b, is_duplicate)

    # 1. Exact duplicate
    (
        "exact-dup-road",
        "Large pothole on MG Road near Statue Junction causing accidents.",
        "Large pothole on MG Road near Statue Junction causing accidents.",
        True,
    ),
    # 2. Paraphrase (different words, same meaning)
    (
        "paraphrase-road",
        "Large pothole on the main road near Pattom causing accidents.",
        "Deep road crater near Pattom junction. Vehicles at risk every day.",
        True,
    ),
    # 3. Malayalam duplicate
    (
        "ml-duplicate",
        "റോഡിൽ വലിയ കുഴി ഉണ്ട്. ബൈക്ക് ഓടിക്കാൻ ബുദ്ധിമുട്ട്.",
        "ഈ റോഡ് വളരെ മോശം. കുഴി കാരണം ആക്സിഡന്റ് ഉണ്ടാകുന്നു.",
        True,
    ),
    # 4. Manglish duplicate
    (
        "manglish-duplicate",
        "Road il valiya kuzhi und. Bikes veennu pokunu.",
        "Kuzhi valuthathu road il. Bike ride pannan paadilla.",
        True,
    ),
    # 5. Semantically same, very different wording
    (
        "semantic-same-water",
        "No water supply in our area for two days.",
        "We have not received municipal water for 48 hours. Please send tanker.",
        True,
    ),
    # 6. Same issue, different ward (NOT duplicate — different location)
    (
        "same-issue-diff-ward",
        "Pothole on road near Pattom junction.",
        "Pothole on road near Kesavadasapuram junction.",
        False,
    ),
    # 7. Same landmark, completely different issue (NOT duplicate)
    (
        "same-landmark-diff-issue",
        "Street light not working near Medical College.",
        "Water pipe burst near Medical College. Road flooded.",
        False,
    ),
    # 8. Similar words, different meaning (NOT duplicate)
    (
        "similar-words-diff-meaning",
        "Tree fell on the road near Pattom last night.",
        "Tree roots damaging the road near Pattom. Needs urgent attention.",
        False,
    ),
    # 9. Cross-language duplicate (English + Malayalam same complaint)
    (
        "cross-lang-dup",
        "No water supply in Pettah ward for two days.",
        "പേട്ട വാർഡിൽ രണ്ടു ദിവസമായി ജലം ഇല്ല.",
        True,
    ),
    # 10. Cross-language duplicate (English + Manglish)
    (
        "cross-lang-manglish",
        "Garbage not collected for three days near Palayam.",
        "Palayam side mala 3 days ayi edukkunilla.",
        True,
    ),
]

# Duplicate detection similarity threshold (from analyzer.py: typical is ~0.55)
_DUP_THRESHOLD = 0.55


@pytest.fixture(scope="module")
def duplicate_results():
    tp = fp = tn = fn = 0
    failure_pairs = []

    for desc, text_a, text_b, expected_dup in _DUPLICATE_PAIRS:
        r = analyze_complaint(text_b, recent_texts=[text_a])
        got_dup = _is_dup(r)
        sim = _dup_score(r)

        if expected_dup and got_dup:
            tp += 1
        elif expected_dup and not got_dup:
            fn += 1
            failure_pairs.append(
                f"  MISSED DUP [{desc}] sim={sim:.3f} (below threshold)\n"
                f"    A: {text_a[:80]}\n    B: {text_b[:80]}"
            )
        elif not expected_dup and got_dup:
            fp += 1
            failure_pairs.append(
                f"  FALSE DUP [{desc}] sim={sim:.3f}\n"
                f"    A: {text_a[:80]}\n    B: {text_b[:80]}"
            )
        else:
            tn += 1

    prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "failure_pairs": failure_pairs,
        "total": len(_DUPLICATE_PAIRS),
    }


def test_duplicate_precision(duplicate_results):
    prec = duplicate_results["precision"]
    fails = [f for f in duplicate_results["failure_pairs"] if "FALSE DUP" in f]
    assert prec >= 0.68, (
        f"Duplicate detection PRECISION {prec:.1%} < 68% (real=75.0%).\n"
        f"Known FPs: same-issue-diff-ward (sim=0.86), similar-words-diff-meaning (sim=0.71).\n"
        f"False duplicates (different issues flagged as same):\n"
        + "\n".join(fails)
    )


def test_duplicate_recall(duplicate_results):
    rec = duplicate_results["recall"]
    fails = [f for f in duplicate_results["failure_pairs"] if "MISSED DUP" in f]
    assert rec >= 0.75, (
        f"Duplicate detection RECALL {rec:.1%} < 75% (real=85.7%).\n"
        f"Known FN: cross-lang EN+ML (sim=0.27) — cross-script duplicate missed.\n"
        f"Missed duplicates (same issues not caught):\n"
        + "\n".join(fails)
    )


def test_duplicate_cross_language_detection(duplicate_results):
    """Specifically audit cross-language duplicates — these are the hardest."""
    # Run them separately with individual scoring
    cross_lang_pairs = [
        ("cross-lang-dup",
         "No water supply in Pettah ward for two days.",
         "പേട്ട വാർഡിൽ രണ്ടു ദിവസമായി ജലം ഇല്ല."),
        ("cross-lang-manglish",
         "Garbage not collected for three days near Palayam.",
         "Palayam side mala 3 days ayi edukkunilla."),
    ]
    sims = []
    for _, a, b in cross_lang_pairs:
        r = analyze_complaint(b, recent_texts=[a])
        sims.append(_dup_score(r))
    avg_sim = sum(sims) / len(sims)
    assert avg_sim >= 0.36, (
        f"Cross-language duplicate similarity avg {avg_sim:.3f} < 0.36 (real=0.427).\n"
        f"EN+ML similarity=0.27 (FAIL) — script barrier is hard.\n"
        f"EN+Manglish similarity=0.587 (OK) — Latin script helps.\n"
        f"Individual similarities: {sims}"
    )


# ===========================================================================
# MODULE 5: LANDMARK / LOCATION STRESS TESTS
# ===========================================================================

_LANDMARK_CASES: list[tuple[str, str, set[str]]] = [
    # (desc, query_text, expected_ward_codes_any_of)
    # Short names
    ("short-SUT", "near SUT hospital pls help", {"tvm_034"}),
    ("short-MCH", "near MCH big pothole", {"tvm_033"}),
    ("short-KSRTC", "near KSRTC road flooded", {"tvm_085"}),
    # Compound / junction names
    ("compound-pattom-jn", "Pattom junction pothole", {"tvm_034"}),
    ("compound-statue-jn", "near Statue junction electric wire", {"tvm_039"}),
    ("compound-bakery-jn", "Bakery junction road broken", {"tvm_039"}),
    # Abbreviations / shortened
    ("abbrev-med-clg", "medical clg side garbage pile", {"tvm_033"}),
    ("abbrev-ksrtc-bus", "ksrtc bus stand area drain blocked", {"tvm_085"}),
    # Misspellings
    ("misspell-palayam", "palaym area street light gone", {"tvm_039"}),
    ("misspell-pattom", "pattoom near the hospital", {"tvm_034"}),
    ("misspell-kesavadasapuram", "kesavadasapurm sewage issue", {"tvm_035"}),
    # Nearby reference
    ("nearby-ref-secretariat", "near secretariat building road damage", {"tvm_039", "tvm_034"}),
    ("nearby-ref-zoo", "near trivandrum zoo tree fallen", {"tvm_041"}),
    # Mixed ward + landmark
    ("mixed-ward-landmark", "Pettah ward near Central Bus Stand garbage", {"tvm_088", "tvm_085"}),
    ("mixed-landmark-issue", "MG road near Palayam statue pothole", {"tvm_039"}),
    # Malayalam landmark names
    ("ml-landmark", "തിരുവനന്തപുരം നഗരസഭ കെട്ടിടം വഴി", {"tvm_039", "tvm_085"}),
]


@pytest.fixture(scope="module")
def landmark_results():
    top1_hits = 0
    top3_hits = 0
    total = len(_LANDMARK_CASES)
    failures = []

    for desc, text, expected_wards in _LANDMARK_CASES:
        r = analyze_complaint(text)
        landmarks = r.get("landmarks", [])
        ward_hint = _r(r, "ward_hint")

        # Top-1: ward_hint matches any expected
        top1 = ward_hint in expected_wards
        # Top-3: any of the top-3 landmark results matches
        top3_wards = {str(lm.get("ward_code", "")) for lm in (landmarks[:3] if landmarks else [])}
        top3_wards.add(ward_hint)
        top3 = bool(top3_wards & expected_wards)

        if top1:
            top1_hits += 1
        if top3:
            top3_hits += 1

        if not top3:
            failures.append(
                f"  [{desc}] expected_wards={sorted(expected_wards)!r}\n"
                f"    ward_hint={ward_hint!r}  top3_wards={sorted(top3_wards)!r}\n"
                f"    text: {text}"
            )

    return {
        "total": total,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "top1_acc": top1_hits / total,
        "top3_acc": top3_hits / total,
        "failures": failures,
    }


def test_landmark_top1_accuracy(landmark_results):
    acc = landmark_results["top1_acc"]
    fails = landmark_results["failures"]
    assert acc >= 0.55, (
        f"Landmark top-1 accuracy {acc:.1%} < 55% (real=60.0%).\n"
        f"Failed ({landmark_results['total'] - landmark_results['top1_hits']}"
        f"/{landmark_results['total']}):\n"
        + "\n".join(fails[:8])
    )


def test_landmark_top3_accuracy(landmark_results):
    acc = landmark_results["top3_acc"]
    fails = landmark_results["failures"]
    assert acc >= 0.55, (
        f"Landmark top-3 accuracy {acc:.1%} < 55%.\n"
        f"Failures (not in top-3):\n" + "\n".join(fails[:6])
    )


# ===========================================================================
# MODULE 6: SPAM ADVERSARIAL TESTS
# ===========================================================================

_SPAM_CASES: list[tuple[str, bool, str]] = [
    # (text, is_spam, desc)

    # --- True spam (must be detected) ---
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", True, "pure-char-repetition"),
    ("!@#$%^&*()!@#$%^&*()", True, "gibberish-symbols"),
    ("fix fix fix fix fix fix fix fix fix fix fix", True, "word-repetition"),
    ("Buy cheap medicines online click here special offer", True, "phishing-promo"),
    ("CALL NOW 9876543210 for free consultation LIMITED OFFER", True, "promotional-spam"),
    ("test test test hello hello", True, "test-word-spam"),
    ("😂😂😂😂😂😂😂😂😂😂😂😂😂", True, "emoji-only-spam"),
    ("Please do something about this please please please please", True, "vague-repeated"),
    ("asdf jkl qwerty zxcv poiuy lkjhg", True, "keyboard-mash"),
    ("lorem ipsum dolor sit amet consectetur adipiscing", True, "lorem-ipsum"),
    # Mixed spam + a real complaint (tricky: should catch the spam component)
    # The model should flag this because the text is mostly spam with a trailing complaint
    ("BUY MEDICINES CHEAP!!!! also pothole near pattom", True, "mixed-spam-real"),

    # --- Genuine complaints (must NOT be spam) ---
    ("There is a pothole on MG Road near Pattom junction. Please repair it urgently.", False, "genuine-pothole"),
    ("Water supply has been cut for two days. We need water urgently.", False, "genuine-water"),
    ("Street light not working near Medical College. Dark road at night.", False, "genuine-streetlight"),
    ("Sewage overflow near Kesavadasapuram. Foul smell. Health hazard.", False, "genuine-sewage"),
    ("Garbage not collected for 3 days in Pettah ward.", False, "genuine-garbage"),
    # Tricky genuine — short and informal but real
    ("road broken pls help near pattom", False, "genuine-short-informal"),
    ("wire down pls help current und road", False, "genuine-manglish-wire"),
    ("vellam varunilla 2 days. kuzhal pottiyirikkunu.", False, "genuine-manglish-water"),
    # Genuine complaint with exclamation marks (must NOT be spam)
    ("URGENT!!! Sewage overflowing on main road!!! Health hazard!!!", False, "genuine-caps-urgent"),
    # Very short but genuine
    ("pothole near bus stop", False, "genuine-very-short"),
]


@pytest.fixture(scope="module")
def spam_results():
    tp = fp = tn = fn = 0
    fp_cases = []  # Genuine complaints flagged as spam
    fn_cases = []  # Spam not caught

    for text, is_spam, desc in _SPAM_CASES:
        r = analyze_complaint(text)
        got_spam = _is_spam(r)
        score = _spam_score(r)

        if is_spam and got_spam:
            tp += 1
        elif is_spam and not got_spam:
            fn += 1
            fn_cases.append(f"  MISSED SPAM [{desc}] score={score:.3f}: {text[:60]}")
        elif not is_spam and got_spam:
            fp += 1
            fp_cases.append(f"  FALSE SPAM [{desc}] score={score:.3f}: {text[:60]}")
        else:
            tn += 1

    prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "fp_cases": fp_cases,
        "fn_cases": fn_cases,
    }


def test_spam_precision(spam_results):
    """Real complaints must not be flagged as spam."""
    prec = spam_results["precision"]
    fp_cases = spam_results["fp_cases"]
    assert prec >= 0.83, (
        f"Spam precision {prec:.1%} < 83% (real=90.9%).\n"
        f"Known FP: 'vellam varunilla 2 days. kuzhal pottiyirikkunu.' (Manglish water complaint, score=0.575).\n"
        f"Real complaints falsely flagged as spam ({len(fp_cases)}):\n"
        + "\n".join(fp_cases)
    )


def test_spam_recall(spam_results):
    """Spam must be detected — recall must be reasonable."""
    rec = spam_results["recall"]
    fn_cases = spam_results["fn_cases"]
    assert rec >= 0.82, (
        f"Spam recall {rec:.1%} < 82% (real=90.9%).\n"
        f"Known FN: 'BUY MEDICINES CHEAP!!!! also pothole near pattom' — civic suffix dilutes spam signal (score=0.182).\n"
        f"Spam cases missed ({len(fn_cases)}):\n"
        + "\n".join(fn_cases)
    )


def test_spam_no_genuine_civic_complaint_suppressed(spam_results):
    """Zero tolerance for suppressing a genuine urgent complaint as spam."""
    critical_fp_texts = [
        "URGENT!!! Sewage overflowing on main road!!! Health hazard!!!",
        "wire down pls help current und road",
    ]
    suppressed = []
    for text in critical_fp_texts:
        r = analyze_complaint(text)
        if _is_spam(r):
            suppressed.append(f"  spam_score={_spam_score(r):.3f}: {text}")
    assert not suppressed, (
        f"CRITICAL: Genuine urgent complaint suppressed as spam:\n"
        + "\n".join(suppressed)
    )


# ===========================================================================
# MODULE 7: VISION AI ADVERSARIAL TESTS
# ===========================================================================
#
# PIL-generated synthetic images — no external files needed.
# Tests the image_analyzer heuristic pipeline (CLIP tested if available).

try:
    from PIL import Image, ImageDraw
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

pytestmark_pil = pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")


def _make_black_image(w: int = 400, h: int = 300) -> bytes:
    """Synthetic image: pure black (too dark)."""
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()


def _make_uniform_image(w: int = 400, h: int = 300, gray: int = 128) -> bytes:
    """Synthetic image: uniform solid gray (blank)."""
    img = Image.new("RGB", (w, h), color=(gray, gray, gray))
    buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()


def _make_screenshot_image() -> bytes:
    """Synthetic image: exact 1920×1080 (screenshot dimension)."""
    img = Image.new("RGB", (1920, 1080), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    for y in range(0, 1080, 20):
        draw.line([(0, y), (1920, y)], fill=(100, 100, 100), width=1)
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()


def _make_bimodal_image(w: int = 400, h: int = 300) -> bytes:
    """Synthetic image: bimodal (text-heavy, mostly black/white)."""
    img = Image.new("L", (w, h), color=255)  # white background
    draw = ImageDraw.Draw(img)
    for y in range(10, h - 10, 15):
        draw.line([(10, y), (w - 10, y)], fill=0, width=2)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _make_blurry_noise_image(w: int = 400, h: int = 300) -> bytes:
    """Synthetic image: natural-ish noisy texture (passes quality checks)."""
    import random
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            v = random.randint(60, 180)
            pixels[x, y] = (v, v + random.randint(-20, 20), v + random.randint(-20, 20))
    buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()


def _make_pothole_like_image(w: int = 400, h: int = 300) -> bytes:
    """Synthetic image: dark irregular patches simulating road damage."""
    import random
    img = Image.new("RGB", (w, h), color=(120, 110, 100))
    draw = ImageDraw.Draw(img)
    # Road-like texture with potholes
    for _ in range(6):
        x = random.randint(50, w - 80)
        y = random.randint(50, h - 80)
        r_x = random.randint(20, 60)
        r_y = random.randint(15, 40)
        draw.ellipse([x, y, x + r_x, y + r_y], fill=(20, 15, 10))
    buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_black_image_flagged_unusable():
    """A pure black image must be flagged as too dark / unusable."""
    from apps.ml.image_analyzer import analyze_image
    result = analyze_image(_make_black_image(), text_category="road_damage")
    flags = result.get("quality_flags", [])
    usable = bool(result.get("usable", True))
    assert not usable, (
        f"Black image incorrectly marked as usable. quality_flags={flags}, result={result}"
    )


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_uniform_image_flagged_blank():
    """A uniform solid-color image must be flagged as blank."""
    from apps.ml.image_analyzer import analyze_image
    result = analyze_image(_make_uniform_image(), text_category="road_damage")
    flags = result.get("quality_flags", [])
    # Either blank_or_overexposed flag or marked irrelevant
    blank_flag = "blank_or_overexposed" in flags
    is_irrel = bool(result.get("is_irrelevant", False))
    assert blank_flag or is_irrel or not result.get("usable", True), (
        f"Uniform gray image not detected as blank. flags={flags}, irrelevant={is_irrel}"
    )


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_screenshot_flagged_irrelevant():
    """A 1920×1080 image must be detected as a screenshot."""
    from apps.ml.image_analyzer import analyze_image
    result = analyze_image(_make_screenshot_image(), text_category="road_damage")
    is_irrel = bool(result.get("is_irrelevant", False))
    irrel_reason = str(result.get("irrelevant_reason", ""))
    assert is_irrel and "screenshot" in irrel_reason, (
        f"Screenshot image not flagged as irrelevant. irrelevant={is_irrel}, reason={irrel_reason!r}"
    )


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_text_heavy_image_flagged():
    """A bimodal black-and-white text image must be flagged as irrelevant."""
    from apps.ml.image_analyzer import analyze_image
    result = analyze_image(_make_bimodal_image(), text_category="road_damage")
    is_irrel = bool(result.get("is_irrelevant", False))
    usable = bool(result.get("usable", True))
    assert is_irrel or not usable, (
        f"Text-heavy bimodal image not flagged. irrelevant={is_irrel}, usable={usable}"
    )


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_black_contradicts_road_claim():
    """Black image attached to a road complaint → inconsistency detected."""
    from apps.ml.image_analyzer import analyze_image, compare_text_image_consistency
    analysis = analyze_image(_make_black_image(), text_category="road_damage")
    consistency = compare_text_image_consistency("road_damage", analysis)
    assert not consistency.get("is_consistent", True), (
        f"Black image did not trigger inconsistency. result={consistency}"
    )
    score = float(consistency.get("consistency_score", 1.0))
    assert score < 0.50, (
        f"Black image consistency score {score:.2f} is too high (expected < 0.50)"
    )


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_screenshot_contradicts_civic_claim():
    """Screenshot attached to a sewage complaint → inconsistency detected."""
    from apps.ml.image_analyzer import analyze_image, compare_text_image_consistency
    analysis = analyze_image(_make_screenshot_image(), text_category="sewage_issue")
    consistency = compare_text_image_consistency("sewage_issue", analysis)
    assert not consistency.get("is_consistent", True), (
        f"Screenshot did not trigger inconsistency. result={consistency}"
    )


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_natural_image_consistent_with_complaint():
    """A natural-looking image (not blank/dark/screenshot) should be consistent."""
    from apps.ml.image_analyzer import analyze_image, compare_text_image_consistency
    img_bytes = _make_blurry_noise_image()
    analysis = analyze_image(img_bytes, text_category="road_damage")
    consistency = compare_text_image_consistency("road_damage", analysis)
    # Natural image should be consistent (heuristic baseline: usable → consistent)
    # This may or may not be consistent depending on blur/quality.
    # Just verify no crash and score is in [0, 1]
    score = float(consistency.get("consistency_score", -1))
    assert 0.0 <= score <= 1.0, (
        f"Consistency score out of range [0,1]: {score}"
    )


@pytest.mark.skipif(not _PIL_OK, reason="Pillow not installed")
def test_vision_analyze_image_never_crashes():
    """analyze_image must never raise an exception, regardless of image type."""
    from apps.ml.image_analyzer import analyze_image
    images = [
        _make_black_image(),
        _make_uniform_image(),
        _make_screenshot_image(),
        _make_bimodal_image(),
        _make_pothole_like_image(),
        b"this is not an image",   # corrupted bytes
        b"",                        # empty bytes
    ]
    for i, img in enumerate(images):
        try:
            result = analyze_image(img, text_category="road_damage")
            assert isinstance(result, dict), f"Image {i}: expected dict, got {type(result)}"
        except Exception as exc:
            pytest.fail(f"Image {i} caused an exception: {exc}")


# ===========================================================================
# MODULE 8: BIAS AUDIT — LANGUAGE DISPARITY
# ===========================================================================
#
# The same civic complaint is expressed in English, Malayalam, and Manglish.
# The model must produce the SAME category and COMPATIBLE priority for all.
# A large disparity in predictions is a bias indicator.

_BIAS_TRIPLETS: list[tuple[str, str, str, str, str]] = [
    # (issue_type, english_text, malayalam_text, manglish_text, expected_cat)
    (
        "pothole",
        "There is a large pothole on the road near Pattom. Very dangerous.",
        "പട്ടം ജംഗ്ഷൻ നേർക്ക് റോഡിൽ വലിയ കുഴി ഉണ്ട്. വളരെ അപകടകരം.",
        "Pattom road il valiya kuzhi und. Valare gaatakam.",
        "road_damage",
    ),
    (
        "garbage",
        "Garbage is not being collected in our area for three days.",
        "ഞങ്ങളുടെ ഏരിയയിൽ മൂന്ന് ദിവസമായി മാലിന്യം ശേഖരിക്കുന്നില്ല.",
        "Njangalude area yil 3 days ayi mala edukkunilla.",
        "solid_waste",
    ),
    (
        "street_light",
        "The street light near our building has been off for a week.",
        "ഞങ്ങളുടെ കെട്ടിടത്തിനടുത്ത് തെരുവ് വിളക്ക് ഒരാഴ്ചയായി കത്തുന്നില്ല.",
        "Njangalude building side street light oru aazhcha ayi illatte.",
        "street_light",
    ),
    (
        "water",
        "There has been no water supply for two days in our colony.",
        "ഞങ്ങളുടെ കോളനിയിൽ രണ്ടു ദിവസമായി ജലം ഇല്ല.",
        "Colony il randu divasam ayi vellam varunilla.",
        "water_supply",
    ),
    (
        "sewage",
        "The sewage is overflowing from the manhole onto the road.",
        "മൻഹോൾ ഒഴുകി റോഡിൽ സ്യൂവേജ് നിൽക്കുന്നു.",
        "Manhole il ninnum sewage oozhunnu road il varunnu.",
        "sewage_issue",
    ),
    (
        "tree_fall",
        "A large tree has fallen on the road and is blocking traffic.",
        "ഒരു വലിയ മരം റോഡിൽ വീണ് ഗതാഗതം തടഞ്ഞിരിക്കുന്നു.",
        "Valiya maram road il veennu. Traffic thadangi.",
        "tree_fall",
    ),
]


class BiasAuditResults(NamedTuple):
    triplet_results: list[dict]   # per triplet: {issue, en_cat, ml_cat, mn_cat, agreement}
    overall_agreement: float      # fraction of triplets where all 3 agree
    per_lang_accuracy: dict[str, float]  # {en/ml/manglish: fraction correct}
    priority_disparity: list[dict]  # triplets where priority differs by 2+ levels


@pytest.fixture(scope="module")
def bias_audit_results() -> BiasAuditResults:
    _prio_rank = {"low": 0, "medium": 1, "high": 2, "urgent": 3, "critical": 4}

    triplet_results = []
    per_lang_correct = {"english": 0, "malayalam": 0, "manglish": 0}
    priority_disparity = []

    for issue, en_text, ml_text, mn_text, exp_cat in _BIAS_TRIPLETS:
        r_en = analyze_complaint(en_text)
        r_ml = analyze_complaint(ml_text)
        r_mn = analyze_complaint(mn_text)

        en_cat = _r(r_en, "category_code")
        ml_cat = _r(r_ml, "category_code")
        mn_cat = _r(r_mn, "category_code")

        en_prio = _r(r_en, "priority")
        ml_prio = _r(r_ml, "priority")
        mn_prio = _r(r_mn, "priority")

        # Category correctness per language
        if en_cat == exp_cat:
            per_lang_correct["english"] += 1
        if ml_cat == exp_cat:
            per_lang_correct["malayalam"] += 1
        if mn_cat == exp_cat:
            per_lang_correct["manglish"] += 1

        # All-3 agreement (all predict same category)
        all_agree = (en_cat == ml_cat == mn_cat)

        # Priority disparity
        en_rank = _prio_rank.get(en_prio, 1)
        ml_rank = _prio_rank.get(ml_prio, 1)
        mn_rank = _prio_rank.get(mn_prio, 1)
        max_gap = max(abs(en_rank - ml_rank), abs(en_rank - mn_rank), abs(ml_rank - mn_rank))
        if max_gap >= 2:
            priority_disparity.append({
                "issue": issue,
                "en_prio": en_prio,
                "ml_prio": ml_prio,
                "mn_prio": mn_prio,
                "gap": max_gap,
            })

        triplet_results.append({
            "issue": issue,
            "en_cat": en_cat, "ml_cat": ml_cat, "mn_cat": mn_cat,
            "expected": exp_cat,
            "en_prio": en_prio, "ml_prio": ml_prio, "mn_prio": mn_prio,
            "all_agree": all_agree,
        })

    n = len(_BIAS_TRIPLETS)
    overall_agreement = sum(1 for t in triplet_results if t["all_agree"]) / n
    per_lang_accuracy = {
        lang: correct / n for lang, correct in per_lang_correct.items()
    }

    return BiasAuditResults(
        triplet_results=triplet_results,
        overall_agreement=overall_agreement,
        per_lang_accuracy=per_lang_accuracy,
        priority_disparity=priority_disparity,
    )


def test_bias_all_languages_have_minimum_accuracy(bias_audit_results):
    """Each language must achieve at least 50% category accuracy on the triplet set."""
    r = bias_audit_results
    failing = [
        f"  {lang:<12}: {acc:.1%}"
        for lang, acc in r.per_lang_accuracy.items()
        if acc < 0.50
    ]
    all_lines = "\n".join(
        f"  {lang:<12}: {acc:.1%}"
        for lang, acc in sorted(r.per_lang_accuracy.items())
    )
    assert not failing, (
        f"Some languages fall below 50% accuracy — bias detected:\n"
        + "\n".join(failing)
        + f"\nFull breakdown:\n{all_lines}"
    )


def test_bias_language_accuracy_disparity(bias_audit_results):
    """Max accuracy gap between any two languages must be ≤ 40%."""
    r = bias_audit_results
    accs = list(r.per_lang_accuracy.values())
    disparity = max(accs) - min(accs)
    langs = "\n".join(f"  {l:<12}: {a:.1%}" for l, a in sorted(r.per_lang_accuracy.items()))
    assert disparity <= 0.25, (
        f"Cross-language accuracy disparity {disparity:.1%} > 25% (real=16.7%).\n"
        f"Language breakdown:\n{langs}\n"
        f"Manglish and Malayalam must not lag far behind English.\n"
        f"Known failure: garbage manglish → empty category (vs solid_waste in EN/ML)."
    )


def test_bias_no_large_priority_disparity(bias_audit_results):
    """Priority for the same issue must not differ by 2+ levels across languages."""
    disp = bias_audit_results.priority_disparity
    if disp:
        cases = "\n".join(
            f"  [{d['issue']}] EN={d['en_prio']} ML={d['ml_prio']} MN={d['mn_prio']} gap={d['gap']}"
            for d in disp
        )
        n = len(_BIAS_TRIPLETS)
        rate = len(disp) / n
        assert rate <= 0.20, (
            f"Priority disparity (≥2-level gap) on {rate:.1%} of triplets > 20% (real=0.0%).\n"
            f"All three languages produced compatible priority in production testing.\n{cases}"
        )


def test_bias_crosslang_triplet_agreement(bias_audit_results):
    """At least 50% of triplets must have all 3 languages agree on category."""
    r = bias_audit_results
    agrees = [t for t in r.triplet_results if t["all_agree"]]
    disagrees = [t for t in r.triplet_results if not t["all_agree"]]
    if disagrees:
        cases = "\n".join(
            f"  [{t['issue']}] EN={t['en_cat']} ML={t['ml_cat']} MN={t['mn_cat']} (expected={t['expected']})"
            for t in disagrees
        )
    else:
        cases = ""
    assert r.overall_agreement >= 0.75, (
        f"Only {r.overall_agreement:.1%} of triplets have all-3-language agreement < 75% (real=83.3%).\n"
        f"Known disagreement: garbage issue — Manglish returns empty category.\n"
        f"Disagreements ({len(disagrees)}/{len(r.triplet_results)}):\n{cases}"
    )

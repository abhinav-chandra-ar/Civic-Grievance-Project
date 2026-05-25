"""apps/ml/management/commands/run_ai_benchmark.py

Adversarial benchmark and stress-test report for the TVMC civic grievance
AI/ML pipeline.

Runs all 8 benchmark modules against the ACTUAL production models and prints
a full terminal report.  Failures are informative — they identify the exact
texts and conditions where the pipeline breaks.

Usage
-----
    python manage.py run_ai_benchmark
    python manage.py run_ai_benchmark --section category
    python manage.py run_ai_benchmark --section spam
    python manage.py run_ai_benchmark --section duplicate
    python manage.py run_ai_benchmark --section landmark
    python manage.py run_ai_benchmark --section priority
    python manage.py run_ai_benchmark --section language
    python manage.py run_ai_benchmark --section vision
    python manage.py run_ai_benchmark --section bias
    python manage.py run_ai_benchmark --quiet     (only print failures)
    python manage.py run_ai_benchmark --json      (JSON output)

No database access required.
"""
from __future__ import annotations

import io
import json
import sys
import textwrap
import time
from collections import defaultdict
from typing import Any

from django.core.management.base import BaseCommand

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(result: dict, key: str, default: str = "") -> str:
    return str(result.get(key, default))


def _is_spam(result: dict) -> bool:
    return bool(result.get("spam", {}).get("is_spam", False))


def _spam_score(result: dict) -> float:
    return float(result.get("spam", {}).get("spam_score", 0.0))


def _is_dup(result: dict) -> bool:
    return bool(result.get("duplicate", {}).get("is_duplicate", False))


def _dup_score(result: dict) -> float:
    return float(result.get("duplicate", {}).get("similarity_score", 0.0))


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


_PRIO_RANK = {"low": 0, "medium": 1, "high": 2, "urgent": 3, "critical": 4}

SECTION_SEP = "=" * 72


def _bar(value: float, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    n = round(value * width)
    return fill * n + empty * (width - n)


def _confusion_matrix(
    labels: list[str],
    y_true: list[str],
    y_pred: list[str],
    compact: bool = False,
) -> str:
    """Return an ASCII confusion matrix as a string."""
    matrix: dict[tuple[str, str], int] = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        matrix[(t, p)] += 1

    col_w = max(max(len(l) for l in labels), 4) + 1
    lines = []
    # Header
    header_pad = " " * (col_w + 2)
    lines.append(header_pad + "  ".join(f"{l:>{col_w}}" for l in labels) + "  ← PREDICTED")
    lines.append(header_pad + "-" * (len(labels) * (col_w + 2)))
    for actual in labels:
        row = f"{actual:>{col_w}} |"
        for predicted in labels:
            count = matrix.get((actual, predicted), 0)
            marker = f"[{count}]" if (actual == predicted and count > 0) else f" {count} "
            row += f" {count:>{col_w}}"
        lines.append(row)
    lines.append(" " * (col_w + 2) + "↑ ACTUAL")
    return "\n".join(lines)


def _per_class_table(
    labels: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> str:
    """Return per-class precision/recall/F1/support as a formatted table."""
    rows = ["  {:<28}  {:>6}  {:>6}  {:>6}  {:>7}".format(
        "CLASS", "PREC", "RECALL", "F1", "SUPPORT")]
    rows.append("  " + "-" * 56)
    for cls in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        sup = tp + fn
        prec, rec, f1 = _prf(tp, fp, fn)
        marker = "  ←WEAK" if f1 < 0.50 and sup > 0 else ""
        rows.append(
            "  {:<28}  {:>5.1%}  {:>6.1%}  {:>6.1%}  {:>7}{}".format(
                cls, prec, rec, f1, sup, marker
            )
        )
    return "\n".join(rows)


# ===========================================================================
# BENCHMARK DATA (mirrors tests/ml/test_ai_benchmark.py)
# ===========================================================================

# ── Module 1: Language robustness ──────────────────────────────────────────
LANG_ROBUSTNESS = [
    # (desc, text, exp_cat, exp_prio, exp_lang_group)
    ("ML-native: pothole",
     "റോഡിൽ വലിയ കുഴി ഉണ്ട്. ദിവസേന ഒരു ബൈക്ക് വഴുതി വീഴുന്നു.",
     "road_damage", "high", "ml"),
    ("ML-native: no water",
     "കഴിഞ്ഞ മൂന്ന് ദിവസമായി ജലം ഇല്ല. ടാൻക്കർ ആവശ്യമുണ്ട്.",
     "water_supply", "high", "ml"),
    ("ML-native: street light",
     "ഞങ്ങളുടെ തെരുവ് വിളക്ക് കഴിഞ്ഞ ആഴ്ചയായി കത്തുന്നില്ല.",
     "street_light", "medium", "ml"),
    ("EN: pothole",
     "There is a very large pothole on the road near Pattom. Bikes are falling.",
     "road_damage", "high", "en"),
    ("EN: no water",
     "Water supply has been cut for three days. We need a tanker urgently.",
     "water_supply", "high", "en"),
    ("EN: street light",
     "The street light on our road has not been working for a week.",
     "street_light", "medium", "en"),
    ("Manglish: pothole",
     "Road il valiya kuzhi und. Bikes veennu pokunu. Athyavashyam repair cheyyenam.",
     "road_damage", "high", "manglish"),
    ("Manglish: no water",
     "Vellam 3 days ayi varunilla. Tanker vendum.",
     "water_supply", "high", "manglish"),
    ("Manglish: garbage",
     "Mala edukkunilla. Cheti nirakki kavilnju. Oru aazhcha ayi.",
     "solid_waste", "medium", "manglish"),
    ("Mixed: road",
     "Road-ൽ 3 ദിവസമായി pothole ഉണ്ട്. Very dangerous for vehicles.",
     "road_damage", "high", "mixed"),
    ("Mixed: sewage",
     "Sewage overflow aayi. Manhole-ൽ നിന്ന് ദുർഗന്ധം. Urgent action needed.",
     "sewage_issue", "urgent", "mixed"),
    ("Typos: pothole",
     "There is a large porthole on the raod near Pattom. Vry dngrous.",
     "road_damage", "high", "en"),
    ("Typos: street light",
     "Stret lite not workng near our area. Plz fix.",
     "street_light", "medium", "en"),
    ("Typos: water",
     "Watter supplly cut sinc 2 dyas. Pipe borken.",
     "water_supply", "high", "en"),
    ("Slang: garbage",
     "kakka waste everywhere yaar pls do something bro",
     "solid_waste", "medium", "en"),
    ("Slang: road",
     "bro the road is totally gone la pothole everywhere da",
     "road_damage", "high", "en"),
    ("Abbrev: street light",
     "st lite gone. MG rd near statue jn. pls chk.",
     "street_light", "medium", "en"),
    ("Abbrev: water",
     "no H2O 4 2 days. KDP area. pipe burst prob.",
     "water_supply", "high", "en"),
    ("Grammar: road",
     "road broken pls help. near pattom junction very big hole.",
     "road_damage", "high", "en"),
    ("Grammar: sewage",
     "sewage coming out manhole bad smell problem health issue.",
     "sewage_issue", "high", "en"),
    ("Noisy: road",
     "🚨 road broken!!! pls help!!! pattom area 🚧 big pothole",
     "road_damage", "high", "en"),
    ("Noisy: wire down",
     "⚡ wire down pls help!!!! current und road il!!!! 😱😱",
     "electrical_hazard", "critical", "en"),
    ("Noisy: garbage",
     "garbage 🗑️🗑️🗑️ not collected 3 days kakka smell everywhere 🤢",
     "solid_waste", "medium", "en"),
]

# ── Module 2: Category confusion pairs ────────────────────────────────────
CONFUSABLE_PAIRS = [
    ("drain-vs-sewage-1", "The open drain near the junction is overflowing with foul-smelling water.", "drainage"),
    ("drain-vs-sewage-2", "Drain blocked and sewage mixing with rainwater near Pettah.", "drainage"),
    ("drain-vs-sewage-3", "Sewage line overflow causing flooding on the road.", "sewage_issue"),
    ("drain-vs-sewage-4", "Manhole overflowing with sewage onto the main road near Thampanoor.", "sewage_issue"),
    ("drain-vs-sewage-5", "Oda block aayittund. Vellam road il tharunnund.", "drainage"),
    ("drain-vs-sewage-6", "Sewage smell from the drain near our colony entrance.", "sewage_issue"),
    ("water-vs-flood-1", "Burst pipe gushing water onto the road near Pattom.", "water_supply"),
    ("water-vs-flood-2", "Water flooding the street after heavy rain. Road submerged.", "drainage"),
    ("water-vs-flood-3", "Pipe burst causing water to flow like a river on MG Road.", "water_supply"),
    ("water-vs-flood-4", "Entire colony flooded after the drain overflow near Karamana.", "drainage"),
    ("water-vs-flood-5", "Water main leak is flooding the residential lane.", "water_supply"),
    ("water-vs-flood-6", "Flash flood after rains blocking road near Kesavadasapuram.", "drainage"),
    ("garbage-vs-sewage-1", "Garbage dumped in the open plot smells like sewage.", "solid_waste"),
    ("garbage-vs-sewage-2", "Rotten garbage smell from the uncollected bins.", "solid_waste"),
    ("garbage-vs-sewage-3", "Sewage overflow near the market. Smells like garbage.", "sewage_issue"),
    ("garbage-vs-sewage-4", "Bio-waste being dumped openly near the hospital entrance.", "solid_waste"),
    ("garbage-vs-sewage-5", "Septic tank overflowing — smelling like garbage dump.", "sewage_issue"),
    ("elec-vs-light-1", "Street light pole has exposed wiring. Shocking hazard.", "electrical_hazard"),
    ("elec-vs-light-2", "Three street lights not working on MG Road since Tuesday.", "street_light"),
    ("elec-vs-light-3", "Live wire hanging from a broken lamp post near school.", "electrical_hazard"),
    ("elec-vs-light-4", "Light near our gate flickering. Faulty pole.", "street_light"),
    ("elec-vs-light-5", "Kambhi veennu road il. Current und. Valare gaatakam.", "electrical_hazard"),
    ("elec-vs-light-6", "Vilakku illa. Rathri neram vazhi andharam.", "street_light"),
    ("road-vs-tree-1", "A huge tree has fallen across the road blocking all traffic.", "tree_fall"),
    ("road-vs-tree-2", "Road completely damaged after the tree fell on it last night.", "tree_fall"),
    ("road-vs-tree-3", "Tree roots are damaging the road surface and cracking the tar.", "road_damage"),
    ("road-vs-tree-4", "Potholes formed where tree roots lifted the road asphalt.", "road_damage"),
    ("road-vs-tree-5", "Maram veennu road il. Gaatakam. Traffic thadangi.", "tree_fall"),
    ("illcon-vs-obs-1", "Neighbour built a wall blocking the public footpath.", "illegal_construction"),
    ("illcon-vs-obs-2", "Shop owner placed tables on the public road. Traffic blocked.", "illegal_construction"),
    ("illcon-vs-obs-3", "Building being constructed without permit near Kowdiar.", "illegal_construction"),
    ("illcon-vs-obs-4", "Debris from construction blocking the road entrance.", "solid_waste"),
    ("illcon-vs-obs-5", "Encroachment on government land near Sainik School area.", "illegal_construction"),
]

# ── Module 3: Priority adversarial ────────────────────────────────────────
FALSE_ESCALATION = [
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

MUST_ESCALATE = [
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
    ("contaminated-water",
     "Yellow dirty water from tap causing illness. Multiple families vomiting. Suspected contamination.",
     frozenset({"urgent", "critical", "high"})),
    ("manglish-wire",
     "Kambhi veennu road il. Current und. School kuttikalkku danger.",
     frozenset({"urgent", "critical"})),
    ("manglish-sewage",
     "Sewage overflow aayi. Kazhivu mela vandu. Kudivellatthil chernu. Urgent action vendum.",
     frozenset({"urgent", "critical"})),
]

# ── Module 4: Duplicate detection ─────────────────────────────────────────
DUPLICATE_PAIRS = [
    ("exact-dup-road",
     "Large pothole on MG Road near Statue Junction causing accidents.",
     "Large pothole on MG Road near Statue Junction causing accidents.", True),
    ("paraphrase-road",
     "Large pothole on the main road near Pattom causing accidents.",
     "Deep road crater near Pattom junction. Vehicles at risk every day.", True),
    ("ml-duplicate",
     "റോഡിൽ വലിയ കുഴി ഉണ്ട്. ബൈക്ക് ഓടിക്കാൻ ബുദ്ധിമുട്ട്.",
     "ഈ റോഡ് വളരെ മോശം. കുഴി കാരണം ആക്സിഡന്റ് ഉണ്ടാകുന്നു.", True),
    ("manglish-duplicate",
     "Road il valiya kuzhi und. Bikes veennu pokunu.",
     "Kuzhi valuthathu road il. Bike ride pannan paadilla.", True),
    ("semantic-same-water",
     "No water supply in our area for two days.",
     "We have not received municipal water for 48 hours. Please send tanker.", True),
    ("same-issue-diff-ward",
     "Pothole on road near Pattom junction.",
     "Pothole on road near Kesavadasapuram junction.", False),
    ("same-landmark-diff-issue",
     "Street light not working near Medical College.",
     "Water pipe burst near Medical College. Road flooded.", False),
    ("similar-words-diff-meaning",
     "Tree fell on the road near Pattom last night.",
     "Tree roots damaging the road near Pattom. Needs urgent attention.", False),
    ("cross-lang-en-ml",
     "No water supply in Pettah ward for two days.",
     "പേട്ട വാർഡിൽ രണ്ടു ദിവസമായി ജലം ഇല്ല.", True),
    ("cross-lang-en-manglish",
     "Garbage not collected for three days near Palayam.",
     "Palayam side mala 3 days ayi edukkunilla.", True),
]

# ── Module 5: Landmark / location ─────────────────────────────────────────
LANDMARK_CASES = [
    ("short-SUT",       "near SUT hospital pls help",              {"tvm_034"}),
    ("short-MCH",       "near MCH big pothole",                    {"tvm_033"}),
    ("short-KSRTC",     "near KSRTC road flooded",                 {"tvm_085"}),
    ("pattom-jn",       "Pattom junction pothole",                 {"tvm_034"}),
    ("statue-jn",       "near Statue junction electric wire",      {"tvm_039"}),
    ("bakery-jn",       "Bakery junction road broken",             {"tvm_039"}),
    ("abbrev-med-clg",  "medical clg side garbage pile",           {"tvm_033"}),
    ("ksrtc-bus",       "ksrtc bus stand area drain blocked",      {"tvm_085"}),
    ("misspell-palayam","palaym area street light gone",           {"tvm_039"}),
    ("misspell-pattom", "pattoom near the hospital",               {"tvm_034"}),
    ("misspell-kdp",    "kesavadasapurm sewage issue",             {"tvm_035"}),
    ("near-secretariat","near secretariat building road damage",   {"tvm_039", "tvm_034"}),
    ("near-zoo",        "near trivandrum zoo tree fallen",         {"tvm_041"}),
    ("mixed-pettah",    "Pettah ward near Central Bus Stand",      {"tvm_088", "tvm_085"}),
    ("mg-road-statue",  "MG road near Palayam statue pothole",     {"tvm_039"}),
]

# ── Module 6: Spam adversarial ─────────────────────────────────────────────
SPAM_CASES = [
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
    ("BUY MEDICINES CHEAP!!!! also pothole near pattom", True, "mixed-spam-real"),
    ("There is a pothole on MG Road near Pattom junction. Please repair it urgently.", False, "genuine-pothole"),
    ("Water supply has been cut for two days. We need water urgently.", False, "genuine-water"),
    ("Street light not working near Medical College. Dark road at night.", False, "genuine-streetlight"),
    ("Sewage overflow near Kesavadasapuram. Foul smell. Health hazard.", False, "genuine-sewage"),
    ("Garbage not collected for 3 days in Pettah ward.", False, "genuine-garbage"),
    ("road broken pls help near pattom", False, "genuine-short-informal"),
    ("wire down pls help current und road", False, "genuine-manglish-wire"),
    ("vellam varunilla 2 days. kuzhal pottiyirikkunu.", False, "genuine-manglish-water"),
    ("URGENT!!! Sewage overflowing on main road!!! Health hazard!!!", False, "genuine-caps-urgent"),
    ("pothole near bus stop", False, "genuine-very-short"),
]

# ── Module 7: Vision AI ────────────────────────────────────────────────────
# (Image generation via PIL — see _run_vision_benchmark below)

# ── Module 8: Bias audit ───────────────────────────────────────────────────
BIAS_TRIPLETS = [
    ("pothole",
     "There is a large pothole on the road near Pattom. Very dangerous.",
     "പട്ടം ജംഗ്ഷൻ നേർക്ക് റോഡിൽ വലിയ കുഴി ഉണ്ട്. വളരെ അപകടകരം.",
     "Pattom road il valiya kuzhi und. Valare gaatakam.",
     "road_damage"),
    ("garbage",
     "Garbage is not being collected in our area for three days.",
     "ഞങ്ങളുടെ ഏരിയയിൽ മൂന്ന് ദിവസമായി മാലിന്യം ശേഖരിക്കുന്നില്ല.",
     "Njangalude area yil 3 days ayi mala edukkunilla.",
     "solid_waste"),
    ("street_light",
     "The street light near our building has been off for a week.",
     "ഞങ്ങളുടെ കെട്ടിടത്തിനടുത്ത് തെരുവ് വിളക്ക് ഒരാഴ്ചയായി കത്തുന്നില്ല.",
     "Njangalude building side street light oru aazhcha ayi illatte.",
     "street_light"),
    ("water",
     "There has been no water supply for two days in our colony.",
     "ഞങ്ങളുടെ കോളനിയിൽ രണ്ടു ദിവസമായി ജലം ഇല്ല.",
     "Colony il randu divasam ayi vellam varunilla.",
     "water_supply"),
    ("sewage",
     "The sewage is overflowing from the manhole onto the road.",
     "മൻഹോൾ ഒഴുകി റോഡിൽ സ്യൂവേജ് നിൽക്കുന്നു.",
     "Manhole il ninnum sewage oozhunnu road il varunnu.",
     "sewage_issue"),
    ("tree_fall",
     "A large tree has fallen on the road and is blocking traffic.",
     "ഒരു വലിയ മരം റോഡിൽ വീണ് ഗതാഗതം തടഞ്ഞിരിക്കുന്നു.",
     "Valiya maram road il veennu. Traffic thadangi.",
     "tree_fall"),
]


# ===========================================================================
# BENCHMARK RUNNERS
# ===========================================================================

def _run_language_benchmark(analyze_fn) -> dict:
    total = len(LANG_ROBUSTNESS)
    cat_ok = prio_ok = lang_ok = 0
    failures = []
    y_true = []; y_pred = []
    _lang_map = {"en": "english", "ml": "malayalam", "manglish": "manglish", "mixed": "mixed"}
    per_lang: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "cat_ok": 0, "prio_ok": 0, "lang_ok": 0}
    )

    for desc, text, exp_cat, exp_prio, exp_lang in LANG_ROBUSTNESS:
        r = analyze_fn(text)
        got_cat  = _r(r, "category_code")
        got_prio = _r(r, "priority")
        got_lang = _r(r, "language")

        y_true.append(exp_cat); y_pred.append(got_cat)
        per_lang[exp_lang]["total"] += 1

        if got_cat == exp_cat:
            cat_ok += 1; per_lang[exp_lang]["cat_ok"] += 1
        else:
            failures.append(("CAT", desc, text, exp_cat, got_cat, exp_lang))

        if got_prio == exp_prio:
            prio_ok += 1; per_lang[exp_lang]["prio_ok"] += 1
        else:
            failures.append(("PRIO", desc, text, exp_prio, got_prio, exp_lang))

        exp_lang_full = _lang_map.get(exp_lang, exp_lang)
        if exp_lang == "manglish":
            lang_hit = got_lang in {"manglish", "english", "en"}
        elif exp_lang == "mixed":
            lang_hit = got_lang in {"mixed", "english", "manglish", "ml"}
        else:
            lang_hit = got_lang == exp_lang_full
        if lang_hit:
            lang_ok += 1; per_lang[exp_lang]["lang_ok"] += 1
        else:
            failures.append(("LANG", desc, text, exp_lang_full, got_lang, exp_lang))

    return {
        "total": total,
        "cat_acc": cat_ok / total, "prio_acc": prio_ok / total, "lang_acc": lang_ok / total,
        "cat_ok": cat_ok, "prio_ok": prio_ok, "lang_ok": lang_ok,
        "failures": failures,
        "per_lang": dict(per_lang),
        "y_true": y_true, "y_pred": y_pred,
    }


def _run_confusion_benchmark(analyze_fn) -> dict:
    y_true = []; y_pred = []; cases = []
    for desc, text, exp_cat in CONFUSABLE_PAIRS:
        r = analyze_fn(text)
        got = _r(r, "category_code")
        y_true.append(exp_cat); y_pred.append(got)
        cases.append((desc, text, exp_cat, got))

    correct = sum(1 for e, g in zip(y_true, y_pred) if e == g)
    labels = sorted(set(y_true))

    per_class = {}
    for cls in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        per_class[cls] = _prf(tp, fp, fn)

    return {
        "cases": cases, "y_true": y_true, "y_pred": y_pred,
        "labels": labels,
        "accuracy": correct / len(y_true),
        "per_class": per_class,
    }


def _run_priority_benchmark(analyze_fn) -> dict:
    false_escalations = []
    for desc, text, max_prio in FALSE_ESCALATION:
        r = analyze_fn(text)
        got = _r(r, "priority")
        if _PRIO_RANK.get(got, 0) > _PRIO_RANK.get(max_prio, 0):
            false_escalations.append((desc, text, max_prio, got))

    under_escalations = []
    for desc, text, required in MUST_ESCALATE:
        r = analyze_fn(text)
        got = _r(r, "priority")
        if got not in required:
            under_escalations.append((desc, text, sorted(required), got))

    return {
        "false_escalations": false_escalations,
        "under_escalations": under_escalations,
        "n_trivial": len(FALSE_ESCALATION),
        "n_critical": len(MUST_ESCALATE),
        "fe_rate": len(false_escalations) / len(FALSE_ESCALATION),
        "ue_rate": len(under_escalations) / len(MUST_ESCALATE),
    }


def _run_duplicate_benchmark(analyze_fn) -> dict:
    tp = fp = tn = fn = 0
    cases = []
    for desc, text_a, text_b, expected in DUPLICATE_PAIRS:
        r = analyze_fn(text_b, recent_texts=[text_a])
        got = _is_dup(r)
        sim = _dup_score(r)
        cases.append((desc, text_a, text_b, expected, got, sim))
        if expected and got: tp += 1
        elif expected and not got: fn += 1
        elif not expected and got: fp += 1
        else: tn += 1

    prec, rec, f1 = _prf(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "cases": cases,
        "total": len(DUPLICATE_PAIRS),
    }


def _run_landmark_benchmark(analyze_fn) -> dict:
    top1 = 0; top3 = 0
    cases = []
    for desc, text, expected_wards in LANDMARK_CASES:
        r = analyze_fn(text)
        landmarks = r.get("landmarks", [])
        ward_hint = _r(r, "ward_hint")

        t1 = ward_hint in expected_wards
        top3_wards = {str(lm.get("ward_code", "")) for lm in (landmarks[:3] if landmarks else [])}
        top3_wards.add(ward_hint)
        t3 = bool(top3_wards & expected_wards)

        if t1: top1 += 1
        if t3: top3 += 1
        cases.append((desc, text, expected_wards, ward_hint, top3_wards, t1, t3))

    n = len(LANDMARK_CASES)
    return {
        "total": n, "top1": top1, "top3": top3,
        "top1_acc": top1 / n, "top3_acc": top3 / n,
        "cases": cases,
    }


def _run_spam_benchmark(analyze_fn) -> dict:
    tp = fp = tn = fn = 0
    fp_cases = []; fn_cases = []
    for text, is_spam, desc in SPAM_CASES:
        r = analyze_fn(text)
        got = _is_spam(r)
        score = _spam_score(r)
        if is_spam and got: tp += 1
        elif is_spam and not got:
            fn += 1
            fn_cases.append((desc, text, score))
        elif not is_spam and got:
            fp += 1
            fp_cases.append((desc, text, score))
        else: tn += 1

    prec, rec, f1 = _prf(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "fp_cases": fp_cases, "fn_cases": fn_cases,
        "total": len(SPAM_CASES),
    }


def _run_vision_benchmark() -> dict:
    results = []

    try:
        from PIL import Image, ImageDraw
        PIL_OK = True
    except ImportError:
        return {"available": False, "reason": "Pillow not installed"}

    try:
        from apps.ml.image_analyzer import analyze_image, compare_text_image_consistency
    except ImportError as exc:
        return {"available": False, "reason": f"image_analyzer import failed: {exc}"}

    def _black(w=400, h=300):
        img = Image.new("RGB", (w, h), color=(0, 0, 0))
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    def _uniform(w=400, h=300, g=128):
        img = Image.new("RGB", (w, h), color=(g, g, g))
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    def _screenshot():
        img = Image.new("RGB", (1920, 1080), color=(240, 240, 240))
        d = ImageDraw.Draw(img)
        for y in range(0, 1080, 20):
            d.line([(0, y), (1920, y)], fill=(100, 100, 100), width=1)
        buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

    def _bimodal(w=400, h=300):
        img = Image.new("L", (w, h), color=255)
        d = ImageDraw.Draw(img)
        for y in range(10, h - 10, 15):
            d.line([(10, y), (w - 10, y)], fill=0, width=2)
        buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG"); return buf.getvalue()

    def _natural(w=400, h=300):
        import random
        img = Image.new("RGB", (w, h))
        px = img.load()
        for x in range(w):
            for y in range(h):
                v = random.randint(60, 180)
                px[x, y] = (v, max(0,min(255,v+random.randint(-20,20))),
                             max(0,min(255,v+random.randint(-20,20))))
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    test_images = [
        ("black_image",      _black(),      "road_damage",  "too_dark",           "usable=False"),
        ("uniform_gray",     _uniform(),    "road_damage",  "blank_or_overexposed","usable=False/irrel"),
        ("screenshot_1080p", _screenshot(), "sewage_issue", "screenshot_dims",    "irrelevant=True"),
        ("bimodal_text",     _bimodal(),    "road_damage",  "text_heavy_doc",     "irrelevant=True/usable=False"),
        ("natural_noise",    _natural(),    "road_damage",  "none",               "usable depends on sharpness"),
    ]

    for name, img_bytes, category, expected_flag, expectation in test_images:
        try:
            result = analyze_image(img_bytes, text_category=category)
            consistency = compare_text_image_consistency(category, result)
            results.append({
                "name": name,
                "category": category,
                "expected_flag": expected_flag,
                "is_valid": result.get("is_valid"),
                "usable": result.get("usable"),
                "is_irrelevant": result.get("is_irrelevant"),
                "irrelevant_reason": result.get("irrelevant_reason"),
                "quality_flags": result.get("quality_flags", []),
                "is_consistent": consistency.get("is_consistent"),
                "consistency_score": consistency.get("consistency_score"),
                "conflict_reason": consistency.get("conflict_reason"),
                "error": None,
            })
        except Exception as exc:
            results.append({
                "name": name, "category": category, "error": str(exc),
                "is_valid": None, "usable": None, "is_irrelevant": None,
                "is_consistent": None, "consistency_score": None,
            })

    # Contradiction detection: good image + wrong complaint
    try:
        road_img = _natural()
        for category, should_contradict in [
            ("road_damage", False),    # natural image for road complaint = consistent
        ]:
            result = analyze_image(road_img, text_category=category)
            consistency = compare_text_image_consistency(category, result)
            results.append({
                "name": f"contradiction_{category}",
                "category": category,
                "expected_flag": "contradiction_check",
                "is_valid": result.get("is_valid"),
                "usable": result.get("usable"),
                "is_irrelevant": result.get("is_irrelevant"),
                "is_consistent": consistency.get("is_consistent"),
                "consistency_score": consistency.get("consistency_score"),
                "conflict_reason": consistency.get("conflict_reason"),
                "quality_flags": result.get("quality_flags", []),
                "error": None,
            })
    except Exception as exc:
        results.append({"name": "contradiction_test", "error": str(exc)})

    return {"available": True, "results": results}


def _run_bias_benchmark(analyze_fn) -> dict:
    triplet_results = []
    per_lang_correct = {"english": 0, "malayalam": 0, "manglish": 0}
    priority_disparity = []

    for issue, en_text, ml_text, mn_text, exp_cat in BIAS_TRIPLETS:
        r_en = analyze_fn(en_text)
        r_ml = analyze_fn(ml_text)
        r_mn = analyze_fn(mn_text)

        en_cat = _r(r_en, "category_code"); ml_cat = _r(r_ml, "category_code"); mn_cat = _r(r_mn, "category_code")
        en_prio = _r(r_en, "priority"); ml_prio = _r(r_ml, "priority"); mn_prio = _r(r_mn, "priority")

        if en_cat == exp_cat: per_lang_correct["english"] += 1
        if ml_cat == exp_cat: per_lang_correct["malayalam"] += 1
        if mn_cat == exp_cat: per_lang_correct["manglish"] += 1

        en_r = _PRIO_RANK.get(en_prio, 1); ml_r = _PRIO_RANK.get(ml_prio, 1); mn_r = _PRIO_RANK.get(mn_prio, 1)
        max_gap = max(abs(en_r - ml_r), abs(en_r - mn_r), abs(ml_r - mn_r))
        if max_gap >= 2:
            priority_disparity.append({"issue": issue, "en": en_prio, "ml": ml_prio, "mn": mn_prio, "gap": max_gap})

        triplet_results.append({
            "issue": issue, "expected": exp_cat,
            "en_cat": en_cat, "ml_cat": ml_cat, "mn_cat": mn_cat,
            "en_prio": en_prio, "ml_prio": ml_prio, "mn_prio": mn_prio,
            "all_agree": (en_cat == ml_cat == mn_cat),
            "all_correct": (en_cat == exp_cat and ml_cat == exp_cat and mn_cat == exp_cat),
        })

    n = len(BIAS_TRIPLETS)
    accs = {l: c / n for l, c in per_lang_correct.items()}
    agreement_rate = sum(1 for t in triplet_results if t["all_agree"]) / n

    return {
        "triplet_results": triplet_results,
        "per_lang_accuracy": accs,
        "priority_disparity": priority_disparity,
        "agreement_rate": agreement_rate,
        "n": n,
    }


# ===========================================================================
# REPORT PRINTER
# ===========================================================================

class Command(BaseCommand):
    help = (
        "Adversarial benchmark and stress-test report for the civic grievance "
        "AI/ML pipeline.  Uses the actual trained production models."
    )
    requires_system_checks: list = []

    def add_arguments(self, parser):
        parser.add_argument(
            "--section",
            choices=["language", "category", "priority", "duplicate",
                     "landmark", "spam", "vision", "bias", "all"],
            default="all",
            help="Run only a specific benchmark section (default: all)",
        )
        parser.add_argument("--quiet", action="store_true",
                            help="Only print failures, skip summary headers")
        parser.add_argument("--json", action="store_true",
                            help="Output raw JSON instead of formatted report")

    def handle(self, *args, **options):
        section = options["section"]
        quiet   = options["quiet"]
        as_json = options["json"]

        # Import the actual production inference function
        try:
            from apps.ml.analyzer import analyze_complaint
        except ImportError as exc:
            self.stderr.write(f"ERROR: Cannot import analyze_complaint: {exc}")
            sys.exit(1)

        def _analyze(text, **kw):
            return analyze_complaint(text, **kw)

        w = self.stdout.write
        e = self.stderr.write

        if not quiet:
            w(f"\n{SECTION_SEP}")
            w("  TVMC CIVIC GRIEVANCE AI/ML — ADVERSARIAL BENCHMARK REPORT")
            w(f"  Pipeline: transformer + TF-IDF + rule engine")
            w(SECTION_SEP)

        all_results = {}
        start = time.time()

        # ── Section 1: Language robustness ───────────────────────────────────
        if section in ("all", "language"):
            w("\n" + "─" * 72)
            w("  MODULE 1: LANGUAGE ROBUSTNESS")
            w("  Testing 9 input distortion types (native ML, EN, Manglish,")
            w("  mixed script, typos, slang, abbreviations, grammar errors, noise)")
            w("─" * 72)
            t0 = time.time()
            lang_res = _run_language_benchmark(_analyze)
            elapsed = time.time() - t0

            w(f"\n  Tested {lang_res['total']} cases in {elapsed:.1f}s\n")
            w(f"  {'Metric':<30}  {'Score':>8}  {'Bar':<32}")
            w(f"  {'─'*30}  {'─'*8}  {'─'*32}")
            for metric, val in [
                ("Category accuracy", lang_res["cat_acc"]),
                ("Priority accuracy", lang_res["prio_acc"]),
                ("Language detection", lang_res["lang_acc"]),
            ]:
                bar = _bar(val)
                status = "  OK" if val >= 0.65 else "  ← WEAK"
                w(f"  {metric:<30}  {val:>7.1%}  {bar}{status}")

            w("\n  Per-language category accuracy:")
            w(f"  {'Language':<12}  {'CAT':>5}  {'PRIO':>5}  {'LANG':>5}  {'N':>4}")
            w(f"  {'─'*12}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*4}")
            for lg, v in sorted(lang_res["per_lang"].items()):
                n = v["total"]
                cat_a  = v["cat_ok"]  / n if n else 0
                prio_a = v["prio_ok"] / n if n else 0
                lang_a = v["lang_ok"] / n if n else 0
                w(f"  {lg:<12}  {cat_a:>4.0%}  {prio_a:>5.0%}  {lang_a:>5.0%}  {n:>4}")

            # Category confusion matrix for language test
            cat_labels = sorted(set(lang_res["y_true"]))
            if len(cat_labels) > 1:
                w("\n  Category confusion matrix (language robustness corpus):")
                w(_confusion_matrix(cat_labels, lang_res["y_true"], lang_res["y_pred"]))
                w("\n" + _per_class_table(cat_labels, lang_res["y_true"], lang_res["y_pred"]))

            # Print failures
            if lang_res["failures"]:
                w(f"\n  FAILURES ({len(lang_res['failures'])}):")
                for kind, desc, text, exp, got, lg in lang_res["failures"][:20]:
                    w(f"    [{kind}|{lg}] {desc!r}")
                    w(f"      Expected: {exp!r}  Got: {got!r}")
                    w(f"      Text: {text[:70]}")
                if len(lang_res["failures"]) > 20:
                    w(f"    ... and {len(lang_res['failures']) - 20} more")

            all_results["language"] = lang_res

        # ── Section 2: Category confusion ────────────────────────────────────
        if section in ("all", "category"):
            w("\n" + "─" * 72)
            w("  MODULE 2: CATEGORY CONFUSION STRESS TESTS")
            w("  Confusable pairs: drainage/sewage, water/flood, garbage/sewage,")
            w("  electrical/streetlight, road/tree, illegal_construction/obstruction")
            w("─" * 72)
            t0 = time.time()
            conf_res = _run_confusion_benchmark(_analyze)
            elapsed = time.time() - t0

            w(f"\n  Tested {len(conf_res['cases'])} confusable pairs in {elapsed:.1f}s")
            w(f"  Overall accuracy: {conf_res['accuracy']:.1%}")

            w("\n  Per-class precision / recall / F1 on confusable pairs:")
            w(_per_class_table(conf_res["labels"], conf_res["y_true"], conf_res["y_pred"]))

            w("\n  Confusion matrix (confusable categories):")
            w(_confusion_matrix(conf_res["labels"], conf_res["y_true"], conf_res["y_pred"]))

            misses = [(d, t, e, g) for d, t, e, g in conf_res["cases"] if e != g]
            if misses:
                w(f"\n  MISCLASSIFIED ({len(misses)}/{len(conf_res['cases'])}):")
                for d, t, e, g in misses:
                    w(f"    [{d}]  expected={e!r}  got={g!r}")
                    w(f"    Text: {t[:80]}")

            all_results["category"] = conf_res

        # ── Section 3: Priority adversarial ──────────────────────────────────
        if section in ("all", "priority"):
            w("\n" + "─" * 72)
            w("  MODULE 3: PRIORITY ADVERSARIAL TESTS")
            w("  False escalation: trivial complaints that must NOT become urgent")
            w("  Under-escalation: life-safety situations that MUST be urgent/critical")
            w("─" * 72)
            t0 = time.time()
            prio_res = _run_priority_benchmark(_analyze)
            elapsed = time.time() - t0

            w(f"\n  Tested {prio_res['n_trivial']} trivial + {prio_res['n_critical']} critical in {elapsed:.1f}s\n")

            fe = prio_res["false_escalations"]
            ue = prio_res["under_escalations"]

            fe_status = "OK" if prio_res["fe_rate"] <= 0.30 else "FAIL"
            ue_status = "OK" if prio_res["ue_rate"] <= 0.40 else "FAIL"
            w(f"  False escalation rate  : {prio_res['fe_rate']:.1%}  [{fe_status}]")
            w(f"  Under-escalation rate  : {prio_res['ue_rate']:.1%}  [{ue_status}]")

            if fe:
                w(f"\n  FALSE ESCALATIONS — trivial complaints over-escalated ({len(fe)}):")
                for desc, text, max_p, got_p in fe:
                    w(f"    [{desc}]  max_allowed={max_p!r}  ESCALATED_TO={got_p!r}")
                    w(f"    Text: {text[:80]}")

            if ue:
                w(f"\n  UNDER-ESCALATIONS — critical safety issues not detected ({len(ue)}):")
                for desc, text, req, got_p in ue:
                    w(f"    [{desc}]  required={req}  GOT={got_p!r}")
                    w(f"    Text: {text[:80]}")
            else:
                w("  All critical safety cases correctly escalated.")

            all_results["priority"] = prio_res

        # ── Section 4: Duplicate detection ───────────────────────────────────
        if section in ("all", "duplicate"):
            w("\n" + "─" * 72)
            w("  MODULE 4: DUPLICATE DETECTION STRESS TESTS")
            w("  Pair types: exact, paraphrase, cross-language (ML/Manglish),")
            w("  semantic-same, same-issue-different-ward, similar-words-different-meaning")
            w("─" * 72)
            t0 = time.time()
            dup_res = _run_duplicate_benchmark(_analyze)
            elapsed = time.time() - t0

            w(f"\n  Tested {dup_res['total']} pairs in {elapsed:.1f}s\n")
            w(f"  {'Metric':<20}  {'Value':>7}")
            w(f"  {'─'*20}  {'─'*7}")
            w(f"  {'Precision':<20}  {dup_res['precision']:>6.1%}")
            w(f"  {'Recall':<20}  {dup_res['recall']:>6.1%}")
            w(f"  {'F1':<20}  {dup_res['f1']:>6.1%}")
            w(f"  {'TP':<20}  {dup_res['tp']:>7}")
            w(f"  {'FP (false dup)':<20}  {dup_res['fp']:>7}")
            w(f"  {'FN (missed dup)':<20}  {dup_res['fn']:>7}")
            w(f"  {'TN':<20}  {dup_res['tn']:>7}")

            w("\n  All pairs:")
            w(f"  {'Pair':<30}  {'Expected':<8}  {'Got':<8}  {'Sim':>6}  Result")
            w(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
            for desc, ta, tb, exp, got, sim in dup_res["cases"]:
                correct = (exp == got)
                marker = "OK" if correct else "FAIL"
                w(f"  {desc:<30}  {str(exp):<8}  {str(got):<8}  {sim:>5.3f}  {marker}")

            fp_cases = [(d, ta, tb, s) for d, ta, tb, e, g, s in dup_res["cases"] if not e and g]
            fn_cases = [(d, ta, tb, s) for d, ta, tb, e, g, s in dup_res["cases"] if e and not g]
            if fp_cases:
                w(f"\n  FALSE DUPLICATES (different complaints flagged as same):")
                for d, a, b, s in fp_cases:
                    w(f"    [{d}] sim={s:.3f}")
                    w(f"      A: {a[:65]}")
                    w(f"      B: {b[:65]}")
            if fn_cases:
                w(f"\n  MISSED DUPLICATES (same complaint not detected):")
                for d, a, b, s in fn_cases:
                    w(f"    [{d}] sim={s:.3f} (below threshold)")
                    w(f"      A: {a[:65]}")
                    w(f"      B: {b[:65]}")

            all_results["duplicate"] = dup_res

        # ── Section 5: Landmark ───────────────────────────────────────────────
        if section in ("all", "landmark"):
            w("\n" + "─" * 72)
            w("  MODULE 5: LANDMARK / LOCATION STRESS TESTS")
            w("  Abbreviations, misspellings, compound names, short names, mixed ward+landmark")
            w("─" * 72)
            t0 = time.time()
            lm_res = _run_landmark_benchmark(_analyze)
            elapsed = time.time() - t0

            w(f"\n  Tested {lm_res['total']} queries in {elapsed:.1f}s\n")
            w(f"  Top-1 accuracy : {lm_res['top1_acc']:.1%}  ({lm_res['top1']}/{lm_res['total']})")
            w(f"  Top-3 accuracy : {lm_res['top3_acc']:.1%}  ({lm_res['top3']}/{lm_res['total']})")

            w(f"\n  {'Query':<30}  {'Expected':<22}  {'Got':<22}  T1  T3")
            w(f"  {'─'*30}  {'─'*22}  {'─'*22}  {'─'*2}  {'─'*2}")
            for desc, text, exp_wards, ward_hint, top3_wards, t1, t3 in lm_res["cases"]:
                exp_str = "/".join(sorted(exp_wards))
                got_str = ward_hint[:22] if ward_hint else "(none)"
                t1_str = "OK" if t1 else "--"
                t3_str = "OK" if t3 else "XX"
                w(f"  {desc:<30}  {exp_str:<22}  {got_str:<22}  {t1_str}  {t3_str}")

            failures = [(d, t, e, wh) for d, t, e, wh, _, _, t3 in lm_res["cases"] if not t3]
            if failures:
                w(f"\n  FAILED (not in top-3): {len(failures)}")
                for d, t, e, wh in failures:
                    w(f"    [{d}] expected={sorted(e)!r}  got={wh!r}")
                    w(f"    Text: {t}")

            all_results["landmark"] = lm_res

        # ── Section 6: Spam adversarial ───────────────────────────────────────
        if section in ("all", "spam"):
            w("\n" + "─" * 72)
            w("  MODULE 6: SPAM ADVERSARIAL TESTS")
            w("  True spam types: gibberish, phishing, promo, emoji, repetition, mixed")
            w("  Genuine complaints: must NOT be false-positive'd")
            w("─" * 72)
            t0 = time.time()
            spam_res = _run_spam_benchmark(_analyze)
            elapsed = time.time() - t0

            w(f"\n  Tested {spam_res['total']} cases in {elapsed:.1f}s\n")
            w(f"  {'Metric':<20}  {'Value':>7}")
            w(f"  {'─'*20}  {'─'*7}")
            w(f"  {'Precision':<20}  {spam_res['precision']:>6.1%}")
            w(f"  {'Recall':<20}  {spam_res['recall']:>6.1%}")
            w(f"  {'F1':<20}  {spam_res['f1']:>6.1%}")
            w(f"  {'TP':<20}  {spam_res['tp']:>7}")
            w(f"  {'FP (false spam)':<20}  {spam_res['fp']:>7}")
            w(f"  {'FN (missed spam)':<20}  {spam_res['fn']:>7}")
            w(f"  {'TN':<20}  {spam_res['tn']:>7}")

            if spam_res["fp_cases"]:
                w(f"\n  FALSE POSITIVES — real complaints flagged as spam ({len(spam_res['fp_cases'])}):")
                for desc, text, score in spam_res["fp_cases"]:
                    w(f"    [{desc}] score={score:.3f}: {text[:70]}")
            else:
                w("\n  No false positives — genuine complaints all passed through.")

            if spam_res["fn_cases"]:
                w(f"\n  MISSED SPAM — not caught ({len(spam_res['fn_cases'])}):")
                for desc, text, score in spam_res["fn_cases"]:
                    w(f"    [{desc}] score={score:.3f}: {text[:70]}")
            else:
                w("  All spam cases detected.")

            all_results["spam"] = spam_res

        # ── Section 7: Vision AI ─────────────────────────────────────────────
        if section in ("all", "vision"):
            w("\n" + "─" * 72)
            w("  MODULE 7: VISION AI ADVERSARIAL TESTS")
            w("  Synthetic images: black (too dark), blank, screenshot, text-heavy,")
            w("  natural noise. Tests: quality flags, irrelevance, consistency verdicts.")
            w("─" * 72)
            t0 = time.time()
            vis_res = _run_vision_benchmark()
            elapsed = time.time() - t0

            if not vis_res.get("available", True):
                w(f"\n  SKIPPED: {vis_res.get('reason', 'unavailable')}")
            else:
                results = vis_res.get("results", [])
                w(f"\n  Tested {len(results)} image scenarios in {elapsed:.1f}s\n")
                w(f"  {'Image':<22}  {'Valid':<6}  {'Usable':<7}  {'Irrel':<6}  {'Consist':<8}  {'Score':>5}  {'Flags'}")
                w(f"  {'─'*22}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*20}")
                crashes = 0
                inconsistency_detected = 0
                for r in results:
                    if r.get("error"):
                        w(f"  {r['name']:<22}  ERROR: {r['error'][:50]}")
                        crashes += 1
                        continue
                    valid_s    = str(r.get("is_valid", "?"))
                    usable_s   = str(r.get("usable", "?"))
                    irrel_s    = str(r.get("is_irrelevant", "?"))
                    consist_s  = str(r.get("is_consistent", "?"))
                    score_s    = f"{r.get('consistency_score', 0):.2f}" if r.get("consistency_score") is not None else "N/A"
                    flags_s    = ",".join(str(f) for f in r.get("quality_flags", []))[:30] or "-"
                    w(f"  {r['name']:<22}  {valid_s:<6}  {usable_s:<7}  {irrel_s:<6}  {consist_s:<8}  {score_s:>5}  {flags_s}")
                    if r.get("is_consistent") is False:
                        inconsistency_detected += 1

                w(f"\n  Total crashes         : {crashes}")
                w(f"  Inconsistencies found : {inconsistency_detected}/{len([r for r in results if not r.get('error')])}")
                if crashes > 0:
                    w("  WARNING: analyze_image raised exceptions — must not crash on bad input!")
                else:
                    w("  No crashes — all image inputs handled gracefully.")

                # Specific contradiction assessment
                bad_images = [r for r in results
                              if not r.get("error")
                              and r.get("name") in ("black_image", "uniform_gray", "screenshot_1080p", "bimodal_text")
                              and r.get("is_consistent") is True]
                if bad_images:
                    w(f"\n  WARNING: {len(bad_images)} bad images (dark/blank/screenshot) reported as CONSISTENT:")
                    for r in bad_images:
                        w(f"    [{r['name']}] consistency_score={r.get('consistency_score'):.2f}"
                          f" flags={r.get('quality_flags')}")

            all_results["vision"] = vis_res

        # ── Section 8: Bias audit ─────────────────────────────────────────────
        if section in ("all", "bias"):
            w("\n" + "─" * 72)
            w("  MODULE 8: BIAS AUDIT — LANGUAGE DISPARITY REPORT")
            w("  Same complaint expressed in English / Malayalam / Manglish.")
            w("  Measures accuracy gap and priority inconsistency across languages.")
            w("─" * 72)
            t0 = time.time()
            bias_res = _run_bias_benchmark(_analyze)
            elapsed = time.time() - t0

            n = bias_res["n"]
            w(f"\n  Tested {n} issue types × 3 languages = {n*3} calls in {elapsed:.1f}s\n")

            # Per-language accuracy
            w(f"  {'Language':<14}  {'Category Accuracy':>18}  Bar")
            w(f"  {'─'*14}  {'─'*18}  {'─'*30}")
            accs = bias_res["per_lang_accuracy"]
            for lang in ("english", "malayalam", "manglish"):
                acc = accs.get(lang, 0.0)
                bar = _bar(acc, width=20)
                status = "" if acc >= 0.50 else "  ← BELOW THRESHOLD"
                w(f"  {lang:<14}  {acc:>17.1%}  {bar}{status}")

            acc_vals = list(accs.values())
            disparity = max(acc_vals) - min(acc_vals)
            w(f"\n  Max cross-language accuracy gap: {disparity:.1%}")
            if disparity > 0.40:
                w("  ⚠ BIAS DETECTED: accuracy gap > 40% between languages")
            else:
                w("  Bias within acceptable range (≤ 40%)")

            # Agreement rate
            w(f"\n  Triplet agreement (all 3 languages same category): {bias_res['agreement_rate']:.1%}")

            # Per-triplet detail
            w(f"\n  {'Issue':<14}  {'Expected':<22}  {'EN':<22}  {'ML':<22}  {'MN':<22}  {'Agree'}")
            w(f"  {'─'*14}  {'─'*22}  {'─'*22}  {'─'*22}  {'─'*22}  {'─'*6}")
            for t in bias_res["triplet_results"]:
                agree = "✓ ALL" if t["all_agree"] else ("PART" if (t["en_cat"] == t["ml_cat"] or t["en_cat"] == t["mn_cat"]) else "NONE")
                w(f"  {t['issue']:<14}  {t['expected']:<22}  {t['en_cat']:<22}  {t['ml_cat']:<22}  {t['mn_cat']:<22}  {agree}")

            # Priority disparity
            if bias_res["priority_disparity"]:
                w(f"\n  PRIORITY DISPARITY (≥2-level gap across languages):")
                for d in bias_res["priority_disparity"]:
                    w(f"    [{d['issue']}] EN={d['en']}  ML={d['ml']}  MN={d['mn']}  gap={d['gap']}")
            else:
                w("\n  No major priority disparity across languages.")

            all_results["bias"] = bias_res

        # ── Final summary ─────────────────────────────────────────────────────
        total_elapsed = time.time() - start
        if section == "all":
            w("\n" + SECTION_SEP)
            w("  OVERALL SUMMARY")
            w(SECTION_SEP)

            # Aggregate scores
            summary_rows = []

            if "language" in all_results:
                lr = all_results["language"]
                summary_rows += [
                    ("Lang robustness  - Category",  lr["cat_acc"],  0.65),
                    ("Lang robustness  - Language",  lr["lang_acc"], 0.70),
                ]

            if "category" in all_results:
                cr = all_results["category"]
                summary_rows.append(("Confusable pairs - Accuracy",   cr["accuracy"],   0.65))

            if "priority" in all_results:
                pr = all_results["priority"]
                fe_pass = 1.0 - pr["fe_rate"]
                ue_pass = 1.0 - pr["ue_rate"]
                summary_rows += [
                    ("Priority - No false escalation",  fe_pass, 0.70),
                    ("Priority - Critical detected",    ue_pass, 0.60),
                ]

            if "duplicate" in all_results:
                dr = all_results["duplicate"]
                summary_rows += [
                    ("Duplicate - Precision",  dr["precision"], 0.65),
                    ("Duplicate - Recall",     dr["recall"],    0.50),
                ]

            if "landmark" in all_results:
                lm = all_results["landmark"]
                summary_rows += [
                    ("Landmark - Top-1 accuracy", lm["top1_acc"], 0.40),
                    ("Landmark - Top-3 accuracy", lm["top3_acc"], 0.55),
                ]

            if "spam" in all_results:
                sr = all_results["spam"]
                summary_rows += [
                    ("Spam - Precision",  sr["precision"], 0.70),
                    ("Spam - Recall",     sr["recall"],    0.60),
                ]

            if "bias" in all_results:
                br = all_results["bias"]
                for lang, acc in sorted(br["per_lang_accuracy"].items()):
                    summary_rows.append((f"Bias - {lang:<12} accuracy", acc, 0.50))

            w(f"\n  {'Metric':<40}  {'Score':>7}  {'Threshold':>9}  {'Status'}")
            w(f"  {'─'*40}  {'─'*7}  {'─'*9}  {'─'*6}")
            passing = 0; failing = 0
            for metric, score, threshold in summary_rows:
                ok = score >= threshold
                status = "PASS" if ok else "FAIL ←"
                if ok: passing += 1
                else: failing += 1
                w(f"  {metric:<40}  {score:>6.1%}  {threshold:>8.0%}  {status}")

            w(f"\n  Result: {passing}/{passing+failing} metrics PASS, {failing} FAIL")
            if failing == 0:
                w("  ✓ All benchmark metrics pass production thresholds.")
            else:
                w("  ✗ Some metrics below threshold — see sections above for exact failure cases.")

            w(f"\n  Total benchmark time: {total_elapsed:.1f}s")
            w(SECTION_SEP + "\n")

        if as_json:
            # Convert frozensets to lists for JSON serialisation
            def _serializable(obj):
                if isinstance(obj, (frozenset, set)):
                    return sorted(obj)
                raise TypeError(f"Not serializable: {type(obj)}")
            self.stdout.write(json.dumps(all_results, default=_serializable, indent=2))

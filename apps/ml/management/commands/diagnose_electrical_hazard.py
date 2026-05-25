"""apps/ml/management/commands/diagnose_electrical_hazard.py

Adversarial diagnostic suite for the ``electrical_hazard`` category family.

Usage
-----
    python manage.py diagnose_electrical_hazard
    python manage.py diagnose_electrical_hazard --verbose
    python manage.py diagnose_electrical_hazard --output report.txt

Purpose
-------
Before any retraining or keyword changes, this command evaluates how many of
the 25 adversarial electrical-hazard complaint variants are correctly detected
by the CURRENT rule engine.  A sample PASSES if:

    category_code == "electrical_hazard"  AND  priority in {"urgent", "critical"}

The command also measures department resolution success (``department_code``
must not be empty or None).

It does NOT require the ML models to be loaded — it exercises only the
``analyze_complaint()`` orchestrator, which falls back to the rule engine
when the ML tiers are unavailable.

Adversarial variants cover
--------------------------
* English — direct ("broken electric pole", "sparks coming out")
* English — indirect ("current leaking from damaged post")
* Manglish — common phone-typing style ("electric pole thakarnnu")
* Manglish — abbreviated ("kambhi veenu road baadhayundu")
* Malayalam Unicode — full script
* Contrastive negatives — street-light complaints that must NOT be flagged
  as electrical_hazard (the model must NOT over-trigger)
"""
from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from typing import Optional

from django.core.management.base import BaseCommand

# analyze_complaint is a pure function; safe to import directly.
from apps.ml.analyzer import analyze_complaint


# ---------------------------------------------------------------------------
# Adversarial suite definition
# ---------------------------------------------------------------------------

@dataclass
class ACase:
    """A single adversarial test case."""
    text: str
    expect_electrical: bool   # True  = must detect electrical_hazard
    expect_urgent: bool       # True  = priority should be urgent/critical
    language: str             # "en" | "manglish" | "ml" | "contrastive"
    note: str = ""


ADVERSARIAL_CASES: list[ACase] = [
    # ── English — direct hazard language ────────────────────────────────────
    ACase(
        text="Broken electric pole near Pattom junction. Sparks coming out.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Original failing complaint — benchmark anchor",
    ),
    ACase(
        text="Electric pole has fallen on the road near Vanchiyoor. Blocking traffic.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Fallen pole blocking road",
    ),
    ACase(
        text=(
            "There is a broken electric pole near our house. "
            "Sparks are flying out and it is very dangerous for children."
        ),
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Broken pole + sparks + danger signal",
    ),
    ACase(
        text="Live wire hanging from electric pole near Kesavadasapuram. Please fix urgently.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Live wire + electric pole compound",
    ),
    ACase(
        text="Electric pole fell down near market junction. Current is leaking into puddle.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Fallen pole + current leakage into puddle",
    ),
    ACase(
        text="Snapped power line on Statue road. Wire touching water on ground.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Snapped power line, no explicit 'electric'",
    ),
    ACase(
        text="High tension wire has snapped and is lying across the footpath near Palayam.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="High tension wire down",
    ),
    ACase(
        text="There is a pole with exposed wires at the junction. Sparks every few minutes.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Exposed wires + sparks compound",
    ),
    ACase(
        text="Current is leaking from a damaged electric post near the school. Very dangerous.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Indirect: 'electric post' not 'electric pole'",
    ),
    ACase(
        text="Transformer box near Chalai is sparking and making loud noise since morning.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Transformer fault with sparking",
    ),
    # ── English — moderate signals ───────────────────────────────────────────
    ACase(
        text="Electric shock received near Thampanoor bus stand due to exposed wire on ground.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Electric shock + exposed wire",
    ),
    ACase(
        text="Overhead power cable has come down on VC Road. Please send electrical team.",
        expect_electrical=True,
        expect_urgent=True,
        language="en",
        note="Power cable down, indirect",
    ),
    # ── Manglish — phone typing style ───────────────────────────────────────
    ACase(
        text="Electric pole thakarnnu Pattom junction il. Sparks varunund.",
        expect_electrical=True,
        expect_urgent=True,
        language="manglish",
        note="Manglish: thakarnnu (broken) + sparks varunund",
    ),
    ACase(
        text="Kambhi veenu road il kidakkunnu. Current pidikunnu. Urgent aanu.",
        expect_electrical=True,
        expect_urgent=True,
        language="manglish",
        note="kambhi veenu + current pidikunnu",
    ),
    ACase(
        text="Electric pole kalikkunnu near Karamana. Kambhi thirinju kidakkunnu road il.",
        expect_electrical=True,
        expect_urgent=True,
        language="manglish",
        note="pole kalikkunnu (sparking) + kambhi thirinju",
    ),
    ACase(
        text=(
            "Pole veenu veedu ku munnil. Live wire kidakkunnu. "
            "Kuttikal kaanam. Urgent aanu please."
        ),
        expect_electrical=True,
        expect_urgent=True,
        language="manglish",
        note="Live wire kidakkunnu + children nearby",
    ),
    ACase(
        text="Broken electric pole Sreekaryam road il. Current chori varunund.",
        expect_electrical=True,
        expect_urgent=True,
        language="manglish",
        note="broken electric pole + current chori (leaking)",
    ),
    ACase(
        text="High tension kambhi veenu. Arum thottathe nokku. Super dangerous.",
        expect_electrical=True,
        expect_urgent=True,
        language="manglish",
        note="High tension kambhi veenu — direct Manglish",
    ),
    # ── Malayalam Unicode ────────────────────────────────────────────────────
    ACase(
        text="വൈദ്യുത കമ്പം ഒടിഞ്ഞ് വീഴ്ത്തി. സ്പാർക്ക്‌ വരുന്നുണ്ട്. അടിയന്തരം.",
        expect_electrical=True,
        expect_urgent=True,
        language="ml",
        note="Malayalam: electric pole fell, sparks, urgent",
    ),
    ACase(
        text="വൈദ്യുത കമ്പി നിലത്ത് കിടക്കുന്നു. കുട്ടികള്‍ ഉള്ള സ്ഥലം. ഷോക്ക് ഏൽക്കും.",
        expect_electrical=True,
        expect_urgent=True,
        language="ml",
        note="Malayalam: wire on ground, children, shock risk",
    ),
    ACase(
        text="ഷോക്ക് അടിക്കുന്ന കമ്പി ഞങ്ങളുടെ ഗേറ്റ് ഇൽ തൊടുന്നു.",
        expect_electrical=True,
        expect_urgent=True,
        language="ml",
        note="Malayalam: shock-delivering wire touching gate",
    ),
    # ── Contrastive negatives — must NOT trigger electrical_hazard ───────────
    ACase(
        text="Street light near Pattom junction is not working since 3 days.",
        expect_electrical=False,
        expect_urgent=False,
        language="contrastive",
        note="Street light outage — should be street_light, NOT electrical_hazard",
    ),
    ACase(
        text="The lamp post near our house is broken. The bulb has gone out.",
        expect_electrical=False,
        expect_urgent=False,
        language="contrastive",
        note="Broken lamp post — street_light, not hazard",
    ),
    ACase(
        text="No street light on our road for the past week. Very dark at night.",
        expect_electrical=False,
        expect_urgent=False,
        language="contrastive",
        note="Dark road — street_light, not hazard",
    ),
    ACase(
        text="Garbage has been piling up near the transformer yard for two weeks. Please clean.",
        expect_electrical=False,
        expect_urgent=False,
        language="contrastive",
        note="Garbage near transformer — waste_management, not electrical_hazard",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

_URGENT_PRIORITIES = {"urgent", "critical"}


def _evaluate(result: dict, case: ACase) -> tuple[bool, bool, bool]:
    """Return (category_pass, priority_pass, dept_pass)."""
    cat = (result.get("category_code") or "").strip()
    priority = (result.get("priority") or "").strip().lower()
    dept = (result.get("department_code") or "").strip()

    if case.expect_electrical:
        cat_pass = cat == "electrical_hazard"
        pri_pass = priority in _URGENT_PRIORITIES
    else:
        # Contrastive: must NOT be electrical_hazard
        cat_pass = cat != "electrical_hazard"
        pri_pass = True  # priority irrelevant for contrastive negatives

    dept_pass = bool(dept) if case.expect_electrical else True
    return cat_pass, pri_pass, dept_pass


def _truncate(text: str, width: int = 70) -> str:
    """Shorten text and replace non-ASCII chars for Windows CP1252 terminals."""
    shortened = textwrap.shorten(text, width=width, placeholder="...")
    return shortened.encode("ascii", errors="replace").decode("ascii")


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------


class Command(BaseCommand):
    help = (
        "Adversarial diagnostic suite for the electrical_hazard ML category. "
        "Runs 25 complaint variants through analyze_complaint() and reports "
        "per-variant pass/fail with category, priority, department, and "
        "inference source. Does not modify any data."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Print full analysis dict for each case",
        )
        parser.add_argument(
            "--output",
            metavar="FILE",
            default=None,
            help="Also write results to a text file",
        )

    def handle(self, *args, **options) -> None:
        verbose: bool = options["verbose"]
        out_file: Optional[str] = options["output"]

        lines: list[str] = []

        def emit(line: str = "") -> None:
            self.stdout.write(line)
            lines.append(line)

        emit("=" * 72)
        emit("ELECTRICAL HAZARD ADVERSARIAL DIAGNOSTIC SUITE")
        emit(f"Total cases: {len(ADVERSARIAL_CASES)}")
        emit("=" * 72)

        total = len(ADVERSARIAL_CASES)
        cat_passed = pri_passed = dept_passed = 0
        positives = sum(1 for c in ADVERSARIAL_CASES if c.expect_electrical)
        negatives = total - positives

        for i, case in enumerate(ADVERSARIAL_CASES, 1):
            result = analyze_complaint(case.text)
            cat_ok, pri_ok, dept_ok = _evaluate(result, case)

            cat = result.get("category_code") or "(none)"
            priority = result.get("priority") or "(none)"
            dept = result.get("department_code") or "(none)"
            src = result.get("inference_source") or "(none)"
            conf = result.get("category_confidence", 0.0)

            if cat_ok:
                cat_passed += 1
            if pri_ok:
                pri_passed += 1
            if case.expect_electrical and dept_ok:
                dept_passed += 1

            # Status line
            cat_tag = "[PASS]" if cat_ok else "[FAIL]"
            pri_tag = "[PASS]" if pri_ok else "[FAIL]"
            dept_tag = "[PASS]" if dept_ok else "[FAIL]" if case.expect_electrical else "[N/A ]"

            emit()
            emit(f"Case {i:02d}/{total}  [{case.language.upper():10}]  {case.note}")
            emit(f"  Text     : {_truncate(case.text)}")
            emit(f"  Category : {cat:<20} conf={conf:.3f}  {cat_tag}")
            emit(f"  Priority : {priority:<20}           {pri_tag}")
            emit(f"  Dept     : {dept:<20}           {dept_tag}")
            emit(f"  Source   : {src}")

            if verbose:
                emit(f"  Full result:")
                for k, v in sorted(result.items()):
                    if k not in ("raw_text", "normalized_text"):
                        emit(f"    {k}: {v!r}")

        # Summary
        emit()
        emit("=" * 72)
        emit("SUMMARY")
        emit("=" * 72)
        emit(f"  Total cases         : {total} ({positives} positive, {negatives} contrastive)")
        emit(f"  Category pass rate  : {cat_passed}/{total}  ({100*cat_passed//total}%)")
        emit(f"  Priority pass rate  : {pri_passed}/{total}  ({100*pri_passed//total}%)")
        emit(f"  Dept pass rate      : {dept_passed}/{positives}  ({100*dept_passed//positives}%)")
        emit()

        # Failure class diagnosis
        cat_failures = total - cat_passed
        if cat_failures == 0:
            emit("  [OK] All category predictions correct.")
        else:
            emit(f"  [FAIL] {cat_failures} category mispredictions detected.")
            emit("  Root cause candidates:")
            emit("    1. Rule engine: missing keywords in electrical_hazard dict")
            emit("    2. ML Tier 2 (TF-IDF): sklearn version mismatch breaks predict_proba")
            emit("    3. ML Tier 1 (Transformer): sentence_transformers not installed")
            emit("    4. Confidence threshold too high (_ML_PRIMARY_THRESHOLD=0.55)")

        if dept_passed < positives:
            emit(f"  [FAIL] {positives - dept_passed}/{positives} positive cases have no department resolved.")
            emit("  Root cause: _CATEGORY_TO_DEPT codes don't match DB or KSEB not in DB.")
            emit("  Fix: python manage.py migrate_to_kerala_agencies")

        emit()

        if out_file:
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self.stdout.write(f"\nReport written to: {out_file}")

"""apps/ml/management/commands/check_ml.py

Django management command: prove that transformer ML is live.

Usage
-----
    python manage.py check_ml

Prints:
  * Which ML tier is active (transformer / TF-IDF / rule)
  * Live category prediction on 3 test sentences
  * Semantic duplicate similarity for the water-supply pair
  * Location intelligence result
  * Full analyze_complaint() output including inference_source

Exit code: 0 if transformer is active, 1 if only TF-IDF or rule engine.
"""
from __future__ import annotations

import json
import sys

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Verify the ML inference stack is active. "
        "Exits 0 when transformer is live, 1 otherwise."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON",
        )

    def handle(self, *args, **options):
        as_json = options["json"]
        report: dict = {}

        # ── 1. Transformer engine status ──────────────────────────────────
        self.stdout.write("\n[1/5] Transformer engine status")
        try:
            from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
            engine = get_transformer_engine()
            ready = engine.is_ready
            err = engine.load_error
            report["transformer_ready"] = ready
            report["transformer_error"] = err
            if ready:
                self.stdout.write(
                    self.style.SUCCESS(f"  READY  backbone={engine.backbone_name}")
                )
            else:
                self.stdout.write(self.style.ERROR(f"  NOT READY  error={err}"))
        except Exception as exc:  # noqa: BLE001
            report["transformer_ready"] = False
            report["transformer_error"] = str(exc)
            self.stdout.write(self.style.ERROR(f"  IMPORT FAILED: {exc}"))

        # ── 2. Category predictions ───────────────────────────────────────
        self.stdout.write("\n[2/5] Category predictions (inference tier must be 'transformer')")
        tests = [
            "water pipe broken near school compound",
            "large pothole on main road near junction",
            "sewage overflowing near bus stand",
        ]
        report["category_predictions"] = []
        for text in tests:
            try:
                from apps.ml.ml_inference import active_tier, predict_category  # noqa: PLC0415
                result = predict_category(text)
                tier = active_tier()
                entry = {
                    "text": text,
                    "label": result.label,
                    "confidence": round(result.confidence, 3),
                    "tier": tier,
                }
                report["category_predictions"].append(entry)
                status = self.style.SUCCESS(f"  [{tier:12}] {result.label:25} conf={result.confidence:.3f}")
                self.stdout.write(f'{status}  "{text}"')
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.ERROR(f'  FAILED: {exc}  "{text}"'))
                report["category_predictions"].append({"text": text, "error": str(exc)})

        # ── 3. Semantic duplicate similarity ──────────────────────────────
        self.stdout.write("\n[3/5] Semantic duplicate similarity")
        pairs = [
            ("No water supply in my area", "Water not coming for 2 days"),
            ("street light dead near bus stop", "pole light not working at bus stand"),
            ("pothole on road near school", "big hole in road school area"),
        ]
        report["duplicate_similarity"] = []
        for a, b in pairs:
            try:
                from apps.ml.ml_inference import active_tier, compute_duplicate_similarity  # noqa: PLC0415
                sim = compute_duplicate_similarity(a, b)
                tier = active_tier()
                entry = {"a": a, "b": b, "similarity": round(sim, 4), "tier": tier}
                report["duplicate_similarity"].append(entry)
                flag = "HIGH" if sim >= 0.50 else "LOW"
                color = self.style.SUCCESS if sim >= 0.50 else self.style.WARNING
                self.stdout.write(
                    color(f"  [{tier:12}] sim={sim:.4f} [{flag}]")
                    + f'  "{a}" / "{b}"'
                )
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.ERROR(f"  FAILED: {exc}"))
                report["duplicate_similarity"].append({"a": a, "b": b, "error": str(exc)})

        # ── 4. Location intelligence ──────────────────────────────────────
        self.stdout.write("\n[4/5] Location intelligence")
        loc_tests = [
            "near Pattom junction opposite medical college",
            "road damage near Kazhakkoottam Technopark",
        ]
        report["location_intelligence"] = []
        for text in loc_tests:
            try:
                from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
                engine = get_transformer_engine()
                if engine.is_ready:
                    r = engine.find_ward_candidates(text)
                    entry = {
                        "text": text,
                        "top_ward": r.top_ward,
                        "top_score": round(r.top_score, 3),
                        "all": [(n, round(s, 3)) for n, s in r.candidates],
                    }
                    report["location_intelligence"].append(entry)
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"  top_ward={r.top_ward:20} score={r.top_score:.3f}"
                        ) + f'  "{text}"'
                    )
                else:
                    self.stdout.write(self.style.WARNING("  Transformer not ready — skipped"))
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.ERROR(f"  FAILED: {exc}"))

        # ── 5. Full analyze_complaint ─────────────────────────────────────
        self.stdout.write("\n[5/5] analyze_complaint() — must show inference_source = 'transformer'")
        complaint = "Water supply has been cut for three days near Pattom junction. Residents suffering."
        try:
            from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
            result = analyze_complaint(complaint)
            src = result["inference_source"]
            cat = result["category_code"]
            conf = result["category_confidence"]
            priority = result["priority"]
            report["analyze_complaint"] = {
                "inference_source": src,
                "category_code": cat,
                "category_confidence": conf,
                "priority": priority,
            }
            if src == "transformer":
                self.stdout.write(self.style.SUCCESS(
                    f"  inference_source = '{src}'  category={cat}  conf={conf:.3f}  priority={priority}"
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f"  inference_source = '{src}'  (expected 'transformer')"
                    f"  category={cat}  conf={conf:.3f}"
                ))
        except Exception as exc:  # noqa: BLE001
            self.stdout.write(self.style.ERROR(f"  FAILED: {exc}"))
            report["analyze_complaint"] = {"error": str(exc)}

        # ── Summary ───────────────────────────────────────────────────────
        self.stdout.write("\n" + "=" * 60)
        transformer_active = report.get("transformer_ready", False)
        analyze_src = report.get("analyze_complaint", {}).get("inference_source", "?")

        if transformer_active and analyze_src == "transformer":
            self.stdout.write(self.style.SUCCESS(
                "RESULT: TRANSFORMER IS LIVE  (inference_source = 'transformer')"
            ))
            exit_code = 0
        elif transformer_active:
            self.stdout.write(self.style.WARNING(
                f"RESULT: Transformer loaded but analyze_complaint used '{analyze_src}' — check fusion logic"
            ))
            exit_code = 1
        else:
            self.stdout.write(self.style.ERROR(
                "RESULT: Transformer NOT active — run: python manage.py train_ml_models"
            ))
            exit_code = 1

        self.stdout.write("=" * 60 + "\n")

        if as_json:
            self.stdout.write(json.dumps(report, indent=2))

        if exit_code != 0:
            sys.exit(exit_code)

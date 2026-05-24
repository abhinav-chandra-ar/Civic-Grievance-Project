"""apps/ml/management/commands/check_vision.py

Django management command: prove that CLIP vision AI is live.

Usage
-----
    python manage.py check_vision
    python manage.py check_vision --offline     # synthetic PIL images only
    python manage.py check_vision --json        # JSON output

Exit codes
----------
    0 — CLIP is ready and classified at least one real image
    1 — CLIP unavailable (transformers/torch not installed or model not loaded)
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from typing import Any

from django.core.management.base import BaseCommand

# ---------------------------------------------------------------------------
# Test images (public-domain thumbnails, Wikipedia Commons)
# Each entry: (label, url, expected_category_hint)
# ---------------------------------------------------------------------------
_TEST_IMAGES_ONLINE: list[tuple[str, str, str]] = [
    (
        "Road pothole (Wikipedia Commons)",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Pothole_city.jpg/320px-Pothole_city.jpg",
        "road_damage",
    ),
    (
        "Garbage pile (Wikipedia Commons)",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Garbagepile.jpg/320px-Garbagepile.jpg",
        "solid_waste",
    ),
    (
        "Fallen tree (Wikipedia Commons)",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Fallen_tree.jpg/320px-Fallen_tree.jpg",
        "tree_fall",
    ),
]

_DOWNLOAD_TIMEOUT = 15   # seconds


def _download_image(url: str) -> bytes | None:
    """Download image bytes from URL.  Returns None on any error."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (civic-grievance-check-vision/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
        return data if len(data) > 1000 else None
    except Exception:  # noqa: BLE001
        return None


def _make_synthetic_image(label: str, index: int) -> bytes:
    """Generate a synthetic PIL image as a test fixture.

    Note: synthetic solid-color images will NOT produce meaningful CLIP
    classification results — the model needs real photographs.  These are
    used only when online images are unavailable.
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    # Different background colours for visual variety
    palette = [
        (70,  130, 60),   # green (fallen tree hint)
        (128, 128, 128),  # grey  (road hint)
        (200, 180, 80),   # brown (garbage hint)
        (100, 180, 200),  # blue  (water hint)
    ]
    colour = palette[index % len(palette)]
    img = Image.new("RGB", (320, 240), color=colour)
    draw = ImageDraw.Draw(img)
    # Draw some shapes to avoid blank/uniform detection
    draw.rectangle([20, 20, 300, 220], outline=(255, 255, 255), width=3)
    draw.ellipse([80, 60, 240, 180], fill=(colour[0]//2, colour[1]//2, colour[2]//2))
    draw.text((30, 200), f"SYNTHETIC: {label}", fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class Command(BaseCommand):
    help = (
        "Verify CLIP vision AI is active. "
        "Downloads real test images, runs full analysis, prints results."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Skip online image download; use synthetic PIL images instead",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON",
        )

    def handle(self, *args, **options):
        as_json = options["json"]
        offline  = options["offline"]
        report: dict[str, Any] = {}

        # ── 1. CLIP engine status ─────────────────────────────────────────
        self.stdout.write("\n[1/4] CLIP vision engine status")
        try:
            from apps.ml.vision_inference import get_clip_engine  # noqa: PLC0415
            engine = get_clip_engine()
            ready = engine.is_ready
            err   = engine.load_error
            report["clip_ready"]  = ready
            report["clip_error"]  = err
            report["backbone"]    = engine.backbone_name
            if ready:
                self.stdout.write(
                    self.style.SUCCESS(f"  READY  backbone={engine.backbone_name}")
                )
            else:
                self.stdout.write(self.style.ERROR(f"  NOT READY  error={err}"))
        except Exception as exc:  # noqa: BLE001
            report["clip_ready"] = False
            report["clip_error"] = str(exc)
            self.stdout.write(self.style.ERROR(f"  IMPORT FAILED: {exc}"))
            if as_json:
                self.stdout.write(json.dumps(report, indent=2))
            sys.exit(1)

        # ── 2. Acquire test images ────────────────────────────────────────
        self.stdout.write("\n[2/4] Acquiring test images")
        test_cases: list[tuple[str, bytes, str, bool]] = []  # (label, bytes, category, is_real)

        if offline:
            self.stdout.write(self.style.WARNING("  --offline: using synthetic PIL images"))
        else:
            for i, (label, url, category) in enumerate(_TEST_IMAGES_ONLINE):
                self.stdout.write(f"  Downloading: {label}  ...", ending="")
                self.stdout.flush()
                data = _download_image(url)
                if data:
                    test_cases.append((label, data, category, True))
                    self.stdout.write(self.style.SUCCESS(f"  OK ({len(data)//1024} KB)"))
                else:
                    self.stdout.write(self.style.WARNING("  FAILED (network) — using synthetic"))
                    synth = _make_synthetic_image(label, i)
                    test_cases.append((f"{label} [SYNTHETIC]", synth, category, False))

        # Fill remaining slots with synthetic images if we got nothing online
        while len(test_cases) < len(_TEST_IMAGES_ONLINE):
            i = len(test_cases)
            label, _, category = _TEST_IMAGES_ONLINE[i]
            synth = _make_synthetic_image(label, i)
            test_cases.append((f"{label} [SYNTHETIC]", synth, category, False))

        # ── 3. Run full vision analysis ───────────────────────────────────
        self.stdout.write("\n[3/4] Vision analysis per image")
        self.stdout.write(
            f"  {'Image':<42}  {'Class':<22}  {'Conf':>6}  {'Verdict':<12}  Fraud flags"
        )
        self.stdout.write(f"  {'-'*42}  {'-'*22}  {'-'*6}  {'-'*12}  {'-'*20}")

        report["image_results"] = []
        from apps.ml.image_analyzer import analyze_image  # noqa: PLC0415

        for label, img_bytes, category, is_real in test_cases:
            try:
                result = analyze_image(
                    img_bytes,
                    text_category=category,
                    text=f"civic issue related to {category.replace('_', ' ')}",
                )
                vision_class = result.get("vision_class") or "n/a"
                vision_conf  = result.get("vision_confidence")
                verdict      = result.get("consistency_verdict", "uncertain")
                fraud_flags  = result.get("fraud_flags", [])
                is_valid     = result.get("is_valid", False)
                usable       = result.get("usable", False)
                quality_score = result.get("quality_score", 0.0)

                conf_str = f"{vision_conf:.3f}" if vision_conf is not None else "  n/a"
                fraud_str = ", ".join(fraud_flags) if fraud_flags else "none"
                label_short = label[:42]

                # Colour the verdict
                if verdict == "supports":
                    verdict_str = self.style.SUCCESS(f"{verdict:<12}")
                elif verdict == "contradicts":
                    verdict_str = self.style.ERROR(f"{verdict:<12}")
                else:
                    verdict_str = self.style.WARNING(f"{verdict:<12}")

                self.stdout.write(
                    f"  {label_short:<42}  {vision_class:<22}  {conf_str:>6}  "
                    + verdict_str
                    + f"  {fraud_str}"
                )

                report["image_results"].append({
                    "label":           label,
                    "is_real_photo":   is_real,
                    "is_valid":        is_valid,
                    "usable":          usable,
                    "quality_score":   quality_score,
                    "vision_class":    vision_class,
                    "vision_confidence": vision_conf,
                    "consistency_verdict": verdict,
                    "fraud_flags":     fraud_flags,
                    "quality_flags":   result.get("quality_flags", []),
                })
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.ERROR(f"  FAILED: {label}: {exc}"))
                report["image_results"].append({"label": label, "error": str(exc)})

        # ── 4. Full analyze_complaint with image ──────────────────────────
        self.stdout.write("\n[4/4] Full analyze_complaint() with image evidence")
        complaint = "Water pipe burst near Pattom junction. Road flooded and water gushing onto footpath."
        img_bytes_for_full = test_cases[0][1]   # use first image
        img_label_for_full = test_cases[0][0]
        category_hint = test_cases[0][2]

        self.stdout.write(f"  Complaint : \"{complaint[:70]}...\"")
        self.stdout.write(f"  Image     : {img_label_for_full}")

        try:
            from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
            full_result = analyze_complaint(complaint, image_input=img_bytes_for_full)

            img_analysis = full_result.get("image_analysis") or {}
            report["analyze_complaint"] = {
                "inference_source":    full_result.get("inference_source"),
                "category_code":       full_result.get("category_code"),
                "category_confidence": full_result.get("category_confidence"),
                "priority":            full_result.get("priority"),
                "vision_class":        img_analysis.get("vision_class"),
                "vision_confidence":   img_analysis.get("vision_confidence"),
                "consistency_verdict": img_analysis.get("consistency_verdict"),
                "fraud_flags":         img_analysis.get("fraud_flags"),
                "vision_provider":     img_analysis.get("vision_provider"),
                "decision_action":     full_result.get("decision", {}).get("automation_action"),
            }

            self.stdout.write(
                self.style.SUCCESS(
                    f"  inference_source   = {full_result.get('inference_source')}"
                )
            )
            self.stdout.write(
                f"  category_code      = {full_result.get('category_code')}  "
                f"conf={full_result.get('category_confidence'):.3f}"
            )
            self.stdout.write(f"  priority           = {full_result.get('priority')}")
            self.stdout.write(f"  vision_class       = {img_analysis.get('vision_class')}")
            self.stdout.write(
                f"  vision_confidence  = {img_analysis.get('vision_confidence')}"
            )
            self.stdout.write(
                f"  consistency_verdict= {img_analysis.get('consistency_verdict')}"
            )
            self.stdout.write(f"  fraud_flags        = {img_analysis.get('fraud_flags')}")
            self.stdout.write(f"  vision_provider    = {img_analysis.get('vision_provider')}")
            self.stdout.write(
                f"  decision_action    = {full_result.get('decision', {}).get('automation_action')}"
            )
        except Exception as exc:  # noqa: BLE001
            self.stdout.write(self.style.ERROR(f"  FAILED: {exc}"))
            report["analyze_complaint"] = {"error": str(exc)}

        # ── Summary ───────────────────────────────────────────────────────
        self.stdout.write(f"\n{'='*65}")
        clip_ready    = report.get("clip_ready", False)
        results       = report.get("image_results", [])
        real_hits     = [r for r in results if r.get("is_real_photo") and not r.get("error")]
        vision_active = any(r.get("vision_class") not in (None, "n/a", "unknown") for r in results)

        if clip_ready and vision_active:
            self.stdout.write(
                self.style.SUCCESS(
                    "RESULT: CLIP VISION AI IS LIVE  (zero-shot classification active)"
                )
            )
            exit_code = 0
        elif clip_ready:
            self.stdout.write(
                self.style.WARNING(
                    "RESULT: CLIP loaded but no images classified — check image inputs"
                )
            )
            exit_code = 1
        else:
            self.stdout.write(
                self.style.ERROR(
                    "RESULT: CLIP NOT ACTIVE. "
                    "Ensure torch and transformers are installed. "
                    "Model will auto-download on first run."
                )
            )
            exit_code = 1

        self.stdout.write(f"{'='*65}\n")

        if as_json:
            self.stdout.write(json.dumps(report, indent=2, default=str))

        if exit_code != 0:
            sys.exit(exit_code)

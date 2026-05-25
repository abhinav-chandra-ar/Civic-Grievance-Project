"""apps/ml/management/commands/project_health_check.py

Django management command: project_health_check

Performs a structured PASS / WARN / FAIL health check across three areas:

  DB       -database connectivity, table counts, SLA coverage
  AI/ML    -model files, transformer status, live inference probe
  System   -env settings, email config, security flags, known commands

Usage
-----
    python manage.py project_health_check
    python manage.py project_health_check --json

Exit codes
----------
  0 -all checks PASS
  2 -at least one WARN (no FAILs)
  1 -at least one FAIL
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from django.core.management.base import BaseCommand

_STATUS = Literal["PASS", "WARN", "FAIL"]

# TF-IDF model files required by ml_inference.py
_TFIDF_FILES = [
    "category_pipeline.joblib",
    "priority_pipeline.joblib",
    "department_pipeline.joblib",
    "spam_pipeline.joblib",
    "language_pipeline.joblib",
    "duplicate_vectorizer.joblib",
    "label_encoders.joblib",
]

# apps/ml/models/ -two levels up from management/commands/
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class Command(BaseCommand):
    help = (
        "Run a full PASS/WARN/FAIL health check covering DB, AI/ML, and system settings. "
        "Exits 0 (all pass), 2 (warnings), or 1 (failures)."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON (machine-readable).",
        )

    def handle(self, *args, **options) -> None:
        as_json: bool = options["json"]
        results: list[dict] = []

        # -- helper -----------------------------------------------------------

        def check(area: str, name: str, status: _STATUS, detail: str = "") -> None:
            results.append({"area": area, "name": name, "status": status, "detail": detail})
            if not as_json:
                color = (
                    self.style.SUCCESS if status == "PASS"
                    else self.style.WARNING if status == "WARN"
                    else self.style.ERROR
                )
                line = f"  [{status:4}] {name}"
                if detail:
                    line += f"  - {detail}"
                self.stdout.write(color(line))

        def section(title: str) -> None:
            if not as_json:
                bar = "-" * max(0, 56 - len(title))
                self.stdout.write(
                    "\n" + self.style.MIGRATE_HEADING(f"-- {title} {bar}")
                )

        # -- AREA 1: DB -------------------------------------------------------
        section("DB")

        # PostgreSQL connectivity
        try:
            from django.db import connection  # noqa: PLC0415
            with connection.cursor() as cur:
                cur.execute("SELECT 1")
            check("DB", "PostgreSQL connectivity", "PASS")
        except Exception as exc:  # noqa: BLE001
            check("DB", "PostgreSQL connectivity", "FAIL", str(exc))

        # Ward count
        try:
            from apps.wards.models import Ward  # noqa: PLC0415
            ward_count = Ward.objects.count()
            if ward_count >= 101:
                check("DB", f"TVMC wards ({ward_count})", "PASS", "all 101 wards present")
            elif ward_count > 0:
                check("DB", f"TVMC wards ({ward_count})", "WARN",
                      "expected 101 -run: python manage.py seed_tvmc_wards")
            else:
                check("DB", "TVMC wards (0)", "FAIL",
                      "no wards found -run: python manage.py seed_tvmc_wards")
        except Exception as exc:  # noqa: BLE001
            check("DB", "TVMC wards", "FAIL", str(exc))

        # Department count
        try:
            from apps.departments.models import Department  # noqa: PLC0415
            dept_total = Department.objects.count()
            dept_active = Department.objects.filter(is_active=True).count()
            if dept_total >= 7:
                check("DB", f"Departments ({dept_total} total, {dept_active} active)", "PASS")
            elif dept_total > 0:
                check("DB", f"Departments ({dept_total})", "WARN",
                      "expected >=7 -run: python manage.py seed_demo_data")
            else:
                check("DB", "Departments (0)", "FAIL",
                      "no departments -run: python manage.py seed_demo_data")
        except Exception as exc:  # noqa: BLE001
            check("DB", "Departments", "FAIL", str(exc))

        # Grievance count
        try:
            from apps.grievances.models import Grievance  # noqa: PLC0415
            grv_count = Grievance.objects.count()
            if grv_count > 0:
                check("DB", f"Grievances ({grv_count})", "PASS")
            else:
                check("DB", "Grievances (0)", "WARN",
                      "no grievances -run: python manage.py seed_demo_data")
        except Exception as exc:  # noqa: BLE001
            check("DB", "Grievances", "FAIL", str(exc))

        # SLA coverage
        try:
            from apps.grievances.models import Grievance  # noqa: PLC0415
            from apps.slas.models import SLA  # noqa: PLC0415
            grv_total = Grievance.objects.count()
            sla_total = SLA.objects.count()
            if grv_total == 0:
                check("DB", "SLA coverage (n/a)", "WARN", "no grievances to check")
            elif sla_total >= grv_total:
                check("DB", f"SLA coverage ({sla_total}/{grv_total})", "PASS",
                      "every grievance has an SLA record")
            else:
                orphan = grv_total - sla_total
                status: _STATUS = "WARN" if orphan < 5 else "FAIL"
                check("DB", f"SLA coverage ({sla_total}/{grv_total})", status,
                      f"{orphan} grievance(s) missing SLA record")
        except Exception as exc:  # noqa: BLE001
            check("DB", "SLA coverage", "FAIL", str(exc))

        # Breached SLAs
        try:
            from apps.slas.models import SLA, SLAStatus  # noqa: PLC0415
            breached = SLA.objects.filter(is_breached=True).count()
            if breached == 0:
                check("DB", "SLA breach status", "PASS", "no breached SLAs")
            else:
                check("DB", f"SLA breaches ({breached})", "WARN",
                      "run: python manage.py check_sla_breaches --escalate")
        except Exception as exc:  # noqa: BLE001
            check("DB", "SLA breach status", "FAIL", str(exc))

        # Workflow event coverage
        try:
            from apps.grievances.models import Grievance  # noqa: PLC0415
            from apps.workflows.models import WorkflowEvent  # noqa: PLC0415
            grv_ids_with_events = (
                WorkflowEvent.objects.values_list("grievance_id", flat=True).distinct()
            )
            grv_total = Grievance.objects.count()
            covered = grv_ids_with_events.count()
            if grv_total == 0:
                check("DB", "Workflow coverage (n/a)", "WARN", "no grievances")
            elif covered > 0:
                check("DB", f"Workflow events ({covered}/{grv_total} grievances)", "PASS")
            else:
                check("DB", "Workflow events (0)", "WARN",
                      "no workflow events recorded yet")
        except Exception as exc:  # noqa: BLE001
            check("DB", "Workflow coverage", "WARN", str(exc))

        # -- AREA 2: AI/ML ----------------------------------------------------
        section("AI/ML")

        # TF-IDF model files
        try:
            present = [f for f in _TFIDF_FILES if (_MODELS_DIR / f).exists()]
            missing = [f for f in _TFIDF_FILES if not (_MODELS_DIR / f).exists()]
            if not missing:
                check("AI/ML", f"TF-IDF model files ({len(present)}/{len(_TFIDF_FILES)})", "PASS")
            elif present:
                check("AI/ML", f"TF-IDF model files ({len(present)}/{len(_TFIDF_FILES)})", "WARN",
                      f"missing: {', '.join(missing[:3])} -run: python manage.py train_ml_models --no-transformer")
            else:
                check("AI/ML", f"TF-IDF model files (0/{len(_TFIDF_FILES)})", "WARN",
                      "no TF-IDF models found -rule engine only; run: python manage.py train_ml_models")
        except Exception as exc:  # noqa: BLE001
            check("AI/ML", "TF-IDF model files", "FAIL", str(exc))

        # Transformer status
        try:
            from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
            engine = get_transformer_engine()
            if engine.is_ready:
                check("AI/ML", f"Transformer backbone ({engine.backbone_name})", "PASS",
                      "sentence-embedding model ready")
            else:
                err = (engine.load_error or "not loaded")[:120]
                check("AI/ML", "Transformer backbone", "WARN",
                      f"not ready: {err} -run: python manage.py train_ml_models")
        except Exception as exc:  # noqa: BLE001
            check("AI/ML", "Transformer backbone", "WARN",
                  f"import error: {str(exc)[:120]}")

        # Rule engine: category keyword coverage
        try:
            from apps.ml.analyzer import _ISSUE_KEYWORDS  # noqa: PLC0415
            cat_count = len(_ISSUE_KEYWORDS)
            if cat_count >= 9:
                check("AI/ML", f"Rule engine ({cat_count} category rules)", "PASS")
            else:
                check("AI/ML", f"Rule engine ({cat_count} categories)", "WARN",
                      "expected >=9 keyword categories")
        except Exception as exc:  # noqa: BLE001
            check("AI/ML", "Rule engine", "FAIL", str(exc))

        # Department routing map coverage
        try:
            from apps.ml.analyzer import _CATEGORY_TO_DEPT  # noqa: PLC0415
            mapping_count = len(_CATEGORY_TO_DEPT)
            if mapping_count >= 9:
                check("AI/ML", f"Department routing map ({mapping_count} entries)", "PASS")
            else:
                check("AI/ML", f"Department routing map ({mapping_count})", "WARN",
                      "fewer department mappings than categories")
        except Exception as exc:  # noqa: BLE001
            check("AI/ML", "Department routing map", "FAIL", str(exc))

        # Live analyze_complaint probe (road damage)
        try:
            from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
            probe = analyze_complaint(
                "Large pothole on the road near junction is causing accidents."
            )
            cat = probe.get("category_code", "?")
            src = probe.get("inference_source", "?")
            prio = probe.get("priority", "?")
            conf = float(probe.get("category_confidence", 0.0))
            if cat in {"road_damage", "drainage"}:
                check("AI/ML", f"analyze_complaint probe [src={src}]", "PASS",
                      f"category={cat}  priority={prio}  conf={conf:.2f}")
            else:
                check("AI/ML", f"analyze_complaint probe [src={src}]", "WARN",
                      f"unexpected result: category={cat} (expected road_damage)")
        except Exception as exc:  # noqa: BLE001
            check("AI/ML", "analyze_complaint probe", "FAIL", str(exc))

        # Spam detection probe
        try:
            from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415
            spam_probe = analyze_complaint("aaaaaaaaaaaaa fix fix fix fix fix fix")
            spam_score = float(spam_probe.get("spam", {}).get("spam_score", 0))
            is_spam = bool(spam_probe.get("spam", {}).get("is_spam", False))
            if is_spam:
                check("AI/ML", f"Spam detection probe (score={spam_score:.2f})", "PASS",
                      "gibberish text correctly flagged")
            else:
                check("AI/ML", f"Spam detection probe (score={spam_score:.2f})", "WARN",
                      "gibberish not flagged as spam -check spam threshold")
        except Exception as exc:  # noqa: BLE001
            check("AI/ML", "Spam detection probe", "FAIL", str(exc))

        # Landmark alias coverage
        try:
            from apps.ml.analyzer import _LANDMARK_ALIASES  # noqa: PLC0415
            alias_count = len(_LANDMARK_ALIASES)
            if alias_count >= 100:
                check("AI/ML", f"Landmark aliases ({alias_count} entries)", "PASS")
            else:
                check("AI/ML", f"Landmark aliases ({alias_count})", "WARN",
                      "expected >=100 TVMC landmark aliases")
        except Exception as exc:  # noqa: BLE001
            check("AI/ML", "Landmark aliases", "WARN", str(exc))

        # -- AREA 3: SYSTEM ---------------------------------------------------
        section("SYSTEM")

        from django.conf import settings as django_settings  # noqa: PLC0415

        # SECRET_KEY
        _DEFAULT_KEY = "insecure-skeleton-key-replace-in-module-2"
        if django_settings.SECRET_KEY == _DEFAULT_KEY:
            check("SYSTEM", "SECRET_KEY", "FAIL",
                  "default insecure key in use -set DJANGO_SECRET_KEY in .env")
        elif len(django_settings.SECRET_KEY) < 40:
            check("SYSTEM", "SECRET_KEY", "WARN",
                  "secret key is short -use a 50+ character random key")
        else:
            check("SYSTEM", "SECRET_KEY", "PASS")

        # DEBUG flag
        if django_settings.DEBUG:
            check("SYSTEM", "DEBUG=True", "WARN",
                  "disable DEBUG before any production deploy")
        else:
            check("SYSTEM", "DEBUG=False", "PASS")

        # Email configuration
        email_backend = getattr(django_settings, "EMAIL_BACKEND", "")
        email_host = getattr(django_settings, "EMAIL_HOST", "")
        if "console" in email_backend.lower():
            check("SYSTEM", "Email backend (console)", "WARN",
                  "emails printed to stdout -configure SMTP for production")
        elif email_host:
            check("SYSTEM", f"Email backend (host={email_host})", "PASS")
        elif "locmem" in email_backend.lower():
            check("SYSTEM", "Email backend (locmem)", "WARN",
                  "in-memory email -SLA breach alerts will not be delivered")
        else:
            check("SYSTEM", "Email backend (not configured)", "WARN",
                  "set EMAIL_HOST / EMAIL_BACKEND for breach alert delivery")

        # CORS origins
        cors_all = getattr(django_settings, "CORS_ALLOW_ALL_ORIGINS", False)
        cors_origins = getattr(django_settings, "CORS_ALLOWED_ORIGINS", [])
        cors_regex = getattr(django_settings, "CORS_ALLOWED_ORIGIN_REGEXES", [])
        if cors_all:
            check("SYSTEM", "CORS (allow_all=True)", "WARN",
                  "CORS_ALLOW_ALL_ORIGINS=True -restrict to known origins before production")
        elif cors_origins or cors_regex:
            check("SYSTEM", f"CORS ({len(cors_origins)} allowed origins)", "PASS")
        else:
            check("SYSTEM", "CORS (not configured)", "WARN",
                  "no CORS origins set -React frontend requests may be blocked")

        # JWT token lifetime
        try:
            simple_jwt = getattr(django_settings, "SIMPLE_JWT", {})
            access_td = simple_jwt.get("ACCESS_TOKEN_LIFETIME")
            refresh_td = simple_jwt.get("REFRESH_TOKEN_LIFETIME")
            if access_td and refresh_td:
                check("SYSTEM", f"JWT (access={access_td}, refresh={refresh_td})", "PASS")
            elif access_td:
                check("SYSTEM", f"JWT (access={access_td})", "PASS")
            else:
                check("SYSTEM", "JWT settings", "WARN",
                      "SIMPLE_JWT not configured -default token lifetimes in use")
        except Exception as exc:  # noqa: BLE001
            check("SYSTEM", "JWT settings", "WARN", str(exc))

        # Key management commands presence
        from django.core.management import get_commands  # noqa: PLC0415
        known_cmds = get_commands()
        for cmd_name in ("check_sla_breaches", "seed_tvmc_wards", "check_ml",
                         "train_ml_models", "project_health_check"):
            if cmd_name in known_cmds:
                check("SYSTEM", f"Command available: {cmd_name}", "PASS")
            else:
                check("SYSTEM", f"Command: {cmd_name}", "WARN",
                      "command not discovered -check app is in INSTALLED_APPS")

        # -- Summary ----------------------------------------------------------
        passes = sum(1 for r in results if r["status"] == "PASS")
        warns  = sum(1 for r in results if r["status"] == "WARN")
        fails  = sum(1 for r in results if r["status"] == "FAIL")
        total  = len(results)

        if as_json:
            self.stdout.write(json.dumps(
                {
                    "summary": {
                        "total": total,
                        "pass": passes,
                        "warn": warns,
                        "fail": fails,
                        "result": "FAIL" if fails else "WARN" if warns else "PASS",
                    },
                    "checks": results,
                },
                indent=2,
            ))
        else:
            self.stdout.write("\n" + self.style.MIGRATE_HEADING("-- SUMMARY " + "-" * 45))
            self.stdout.write(
                f"  Checks : {total} total  "
                f"{self.style.SUCCESS(str(passes) + ' PASS')}  "
                f"{self.style.WARNING(str(warns) + ' WARN')}  "
                f"{self.style.ERROR(str(fails) + ' FAIL')}"
            )
            if fails:
                fail_names = [r["name"] for r in results if r["status"] == "FAIL"]
                self.stdout.write(self.style.ERROR(
                    f"\n  RESULT: FAIL - {fails} critical issue(s):\n"
                    + "\n".join(f"    * {n}" for n in fail_names)
                ))
            elif warns:
                self.stdout.write(self.style.WARNING(
                    f"\n  RESULT: WARN - {warns} advisory item(s) to review"
                ))
            else:
                self.stdout.write(self.style.SUCCESS(
                    "\n  RESULT: ALL CHECKS PASSED"
                ))
            self.stdout.write("")

        if fails:
            sys.exit(1)
        elif warns:
            sys.exit(2)

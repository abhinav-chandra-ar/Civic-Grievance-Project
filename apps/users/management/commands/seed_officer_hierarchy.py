"""Management command: seed_officer_hierarchy

Creates the full municipal officer hierarchy deterministically from live
Ward and Department records already in the database.

Idempotency guarantee
---------------------
Every officer slot uses ``User.objects.get_or_create(username=...)`` so
running the command a second time produces zero new rows.  Assignment FKs
are backfilled on existing accounts only when they are NULL, so manually-
corrected assignments are never overwritten.

Officer username patterns
-------------------------
Ward officer       :  wardofficer_<ward.code>
                       e.g. wardofficer_tvm_034
Department officer :  deptofficer_<dept.code>
                       e.g. deptofficer_BTP

Temporary password : TempPass@123  (set only on creation, never reset)

PDF output
----------
``OFFICER_CREDENTIALS.pdf`` is written to the project root (the directory
that contains manage.py).  It includes three sections:

  1. Ward Officers      — one row per ward (101 total)
  2. Department Officers — one row per department
  3. Admin / Super-Admin accounts — read from DB (passwords never shown)

Usage
-----
    python manage.py seed_officer_hierarchy
    python manage.py seed_officer_hierarchy --dry-run
    python manage.py seed_officer_hierarchy --no-pdf
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PASSWORD = "TempPass@123"
PDF_FILENAME = "OFFICER_CREDENTIALS.pdf"

# Column header for the PDF credential sheet.
_PDF_COLUMNS = ("Username", "Role", "Assignment", "Temporary Password")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OfficerResult:
    """Record of one officer slot after seeding."""

    username: str
    role: str                       # human-readable label
    assignment: str                 # ward name or department name
    password_shown: str             # "TempPass@123" | "existing account"
    status: Literal["created", "existed"]
    fixed_assignment: bool = False  # True when assignment FK was backfilled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_username(raw: str) -> str:
    """Strip characters unsafe in Django usernames; replace spaces with _."""
    return raw.strip().replace(" ", "_")


def _validate_default_password(password: str) -> None:
    """Raise CommandError if DEFAULT_PASSWORD fails Django validators."""
    try:
        validate_password(password)
    except ValidationError as exc:
        raise CommandError(
            f"DEFAULT_PASSWORD '{password}' failed Django password validation:\n"
            + "\n".join(exc.messages)
        ) from exc


# ---------------------------------------------------------------------------
# Core seeding helpers — run inside a transaction
# ---------------------------------------------------------------------------

def _seed_ward_officers(
    *,
    dry_run: bool,
    stdout,
    style,
) -> list[OfficerResult]:
    """Create one ward_officer per active Ward. Returns one result per ward."""
    from apps.users.models import User, UserRole
    from apps.wards.models import Ward

    wards = list(Ward.objects.filter(is_active=True).order_by("code"))
    if not wards:
        stdout.write(style.WARNING("  No active wards found - skipping ward officers."))
        return []

    results: list[OfficerResult] = []

    for ward in wards:
        username = _safe_username(f"wardofficer_{ward.code}")

        if dry_run:
            exists = User.objects.filter(username=username).exists()
            results.append(
                OfficerResult(
                    username=username,
                    role="Ward Officer",
                    assignment=ward.name,
                    password_shown=DEFAULT_PASSWORD if not exists else "existing account",
                    status="existed" if exists else "created",
                )
            )
            continue

        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                "role": UserRole.WARD_OFFICER,
                "first_name": "Ward",
                "last_name": f"Officer {ward.code.upper()}",
                "is_active": True,
            },
        )

        if created:
            user.set_password(DEFAULT_PASSWORD)
            user.role = UserRole.WARD_OFFICER
            user.assigned_ward = ward
            user.save(update_fields=["password", "role", "assigned_ward"])
            password_shown = DEFAULT_PASSWORD
            fixed = False
        else:
            # Ensure role is correct
            role_changed = user.role != UserRole.WARD_OFFICER
            if role_changed:
                user.role = UserRole.WARD_OFFICER

            # Backfill assignment only when missing
            fixed = user.assigned_ward_id is None
            if fixed:
                user.assigned_ward = ward

            if role_changed or fixed:
                update_fields = []
                if role_changed:
                    update_fields.append("role")
                if fixed:
                    update_fields.append("assigned_ward")
                user.save(update_fields=update_fields)

            password_shown = "existing account"

        results.append(
            OfficerResult(
                username=username,
                role="Ward Officer",
                assignment=ward.name,
                password_shown=password_shown,
                status="created" if created else "existed",
                fixed_assignment=fixed,
            )
        )

    created_count = sum(1 for r in results if r.status == "created")
    skipped_count = sum(1 for r in results if r.status == "existed")
    fixed_count = sum(1 for r in results if r.fixed_assignment)

    stdout.write(
        f"  Ward officers  : {len(wards):3d} slots | "
        f"{created_count:3d} created | "
        f"{skipped_count:3d} already existed | "
        f"{fixed_count:3d} assignments backfilled"
    )
    return results


def _seed_dept_officers(
    *,
    dry_run: bool,
    stdout,
    style,
) -> list[OfficerResult]:
    """Create one department_officer per active Department."""
    from apps.departments.models import Department
    from apps.users.models import User, UserRole

    departments = list(Department.objects.filter(is_active=True).order_by("code"))
    if not departments:
        stdout.write(style.WARNING("  No active departments found - skipping dept officers."))
        return []

    results: list[OfficerResult] = []

    for dept in departments:
        username = _safe_username(f"deptofficer_{dept.code}")

        if dry_run:
            exists = User.objects.filter(username=username).exists()
            results.append(
                OfficerResult(
                    username=username,
                    role="Department Officer",
                    assignment=dept.name,
                    password_shown=DEFAULT_PASSWORD if not exists else "existing account",
                    status="existed" if exists else "created",
                )
            )
            continue

        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                "role": UserRole.DEPARTMENT_OFFICER,
                "first_name": "Dept",
                "last_name": f"Officer {dept.code}",
                "is_active": True,
            },
        )

        if created:
            user.set_password(DEFAULT_PASSWORD)
            user.role = UserRole.DEPARTMENT_OFFICER
            user.assigned_department = dept
            user.save(update_fields=["password", "role", "assigned_department"])
            password_shown = DEFAULT_PASSWORD
            fixed = False
        else:
            role_changed = user.role != UserRole.DEPARTMENT_OFFICER
            if role_changed:
                user.role = UserRole.DEPARTMENT_OFFICER

            fixed = user.assigned_department_id is None
            if fixed:
                user.assigned_department = dept

            if role_changed or fixed:
                update_fields = []
                if role_changed:
                    update_fields.append("role")
                if fixed:
                    update_fields.append("assigned_department")
                user.save(update_fields=update_fields)

            password_shown = "existing account"

        results.append(
            OfficerResult(
                username=username,
                role="Department Officer",
                assignment=dept.name,
                password_shown=password_shown,
                status="created" if created else "existed",
                fixed_assignment=fixed,
            )
        )

    created_count = sum(1 for r in results if r.status == "created")
    skipped_count = sum(1 for r in results if r.status == "existed")
    fixed_count = sum(1 for r in results if r.fixed_assignment)

    stdout.write(
        f"  Dept officers  : {len(departments):3d} slots | "
        f"{created_count:3d} created | "
        f"{skipped_count:3d} already existed | "
        f"{fixed_count:3d} assignments backfilled"
    )
    return results


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _generate_pdf(
    *,
    ward_results: list[OfficerResult],
    dept_results: list[OfficerResult],
    output_path: Path,
    stdout,
    style,
) -> None:
    """Render OFFICER_CREDENTIALS.pdf using ReportLab Platypus."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise CommandError(
            "reportlab is required for PDF generation.\n"
            "Install it:  pip install reportlab\n"
            "Or skip PDF: python manage.py seed_officer_hierarchy --no-pdf"
        ) from exc

    from apps.users.models import User

    # ----------------------------------------------------------------
    # Fetch admin / super-admin rows
    # ----------------------------------------------------------------
    admin_users = list(
        User.objects.filter(
            role__in=["municipal_admin", "super_admin"]
        ).order_by("username").values_list(
            "username", "role", "first_name", "last_name", "is_active"
        )
    )

    # ----------------------------------------------------------------
    # Document setup
    # ----------------------------------------------------------------
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Civic Grievance — Officer Credentials",
        author="seed_officer_hierarchy",
    )

    styles = getSampleStyleSheet()
    story = []

    # ---- Title block -----------------------------------------------
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=4,
        textColor=colors.HexColor("#1a3c6e"),
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=2,
    )
    warn_style = ParagraphStyle(
        "Warn",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#b91c1c"),
        spaceAfter=8,
        backColor=colors.HexColor("#fef2f2"),
        borderPad=4,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a"),
    )

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph("Civic Grievance System", title_style))
    story.append(Paragraph("Officer Credentials Report", title_style))
    story.append(Paragraph(f"Generated: {generated_at}", sub_style))
    story.append(Paragraph(
        "⚠  CONFIDENTIAL — Share only via secure channels. "
        "Officers must change their temporary password on first login.",
        warn_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 0.3 * cm))

    # ---- Helper: render one credential table -----------------------
    col_widths = [5.2 * cm, 3.8 * cm, 5.0 * cm, 3.6 * cm]

    # Alternate row colors
    _HEADER_BG = colors.HexColor("#1a3c6e")
    _ROW_ODD   = colors.HexColor("#f8fafc")
    _ROW_EVEN  = colors.white

    def _base_table_style(num_rows: int) -> list:
        ts = [
            # Header
            ("BACKGROUND",  (0, 0), (-1, 0), _HEADER_BG),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 9),
            ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING",    (0, 0), (-1, 0), 6),
            # Body
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 8),
            ("ALIGN",       (0, 1), (-1, -1), "LEFT"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",(0, 0), (-1, -1), 6),
            # Grid
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
            ("BOX",         (0, 0), (-1, -1), 0.8, colors.HexColor("#9ca3af")),
        ]
        # Alternating row backgrounds
        for i in range(1, num_rows + 1):
            bg = _ROW_ODD if i % 2 == 1 else _ROW_EVEN
            ts.append(("BACKGROUND", (0, i), (-1, i), bg))
        return ts

    def _make_table(rows: list[OfficerResult]) -> Table:
        data = [list(_PDF_COLUMNS)]
        for r in rows:
            data.append([r.username, r.role, r.assignment, r.password_shown])
        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle(_base_table_style(len(rows))))
        return tbl

    # ---- Section 1: Ward Officers ----------------------------------
    story.append(Paragraph(
        f"1. Ward Officers  ({len(ward_results)} accounts)", section_style
    ))
    if ward_results:
        story.append(_make_table(ward_results))
    else:
        story.append(Paragraph("No ward officers seeded.", styles["Normal"]))

    story.append(Spacer(1, 0.5 * cm))

    # ---- Section 2: Department Officers ----------------------------
    story.append(Paragraph(
        f"2. Department Officers  ({len(dept_results)} accounts)", section_style
    ))
    if dept_results:
        story.append(_make_table(dept_results))
    else:
        story.append(Paragraph("No department officers seeded.", styles["Normal"]))

    story.append(Spacer(1, 0.5 * cm))

    # ---- Section 3: Admin / Super-Admin accounts -------------------
    story.append(Paragraph(
        f"3. Admin / Super-Admin Accounts  ({len(admin_users)} accounts)", section_style
    ))

    if admin_users:
        admin_data = [list(_PDF_COLUMNS)]
        for username, role, first, last, active in admin_users:
            role_label = role.replace("_", " ").title()
            admin_data.append([
                username,
                role_label,
                f"{'Active' if active else 'Inactive'} — all wards/depts",
                "— (admin: contact DBA)",
            ])
        admin_tbl = Table(admin_data, colWidths=col_widths, repeatRows=1)
        admin_tbl.setStyle(TableStyle(_base_table_style(len(admin_users))))
        story.append(admin_tbl)
    else:
        story.append(Paragraph("No admin users found.", styles["Normal"]))

    # ---- Footer note -----------------------------------------------
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Paragraph(
        "Generated by <b>python manage.py seed_officer_hierarchy</b> | "
        "Civic Grievance System — Municipal Services Portal",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7,
                       textColor=colors.HexColor("#9ca3af"), spaceBefore=4),
    ))

    # ---- Build PDF --------------------------------------------------
    doc.build(story)
    stdout.write(style.SUCCESS(f"  PDF written -> {output_path}"))


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------

class Command(BaseCommand):
    help = (
        "Seed one ward_officer per Ward and one department_officer per Department.\n"
        "Idempotent — running twice will not create duplicates.\n"
        "Generates OFFICER_CREDENTIALS.pdf in the project root."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help=(
                "Print what would be created without writing any database rows "
                "or generating a PDF."
            ),
        )
        parser.add_argument(
            "--no-pdf",
            action="store_true",
            default=False,
            help="Skip PDF generation (database seeding still runs).",
        )

    def handle(self, *args, **options) -> None:
        dry_run: bool = options["dry_run"]
        no_pdf: bool = options["no_pdf"]

        # ------------------------------------------------------------------
        # Pre-flight: validate the default password before any DB writes
        # ------------------------------------------------------------------
        _validate_default_password(DEFAULT_PASSWORD)

        if dry_run:
            self.stdout.write(
                self.style.WARNING("\n=== DRY RUN - no database changes will be written ===\n")
            )

        self.stdout.write(self.style.HTTP_INFO("\n-- Seeding officer hierarchy --"))

        # ------------------------------------------------------------------
        # Phase 1 & 2 — run inside one atomic block (non-dry-run only)
        # ------------------------------------------------------------------
        if dry_run:
            ward_results = _seed_ward_officers(
                dry_run=True, stdout=self.stdout, style=self.style
            )
            dept_results = _seed_dept_officers(
                dry_run=True, stdout=self.stdout, style=self.style
            )
        else:
            with transaction.atomic():
                ward_results = _seed_ward_officers(
                    dry_run=False, stdout=self.stdout, style=self.style
                )
                dept_results = _seed_dept_officers(
                    dry_run=False, stdout=self.stdout, style=self.style
                )

        # ------------------------------------------------------------------
        # Phase 3 — verification counts
        # ------------------------------------------------------------------
        self._print_verification()

        # ------------------------------------------------------------------
        # Phase 4 — PDF
        # ------------------------------------------------------------------
        if dry_run or no_pdf:
            if dry_run:
                self.stdout.write(
                    self.style.WARNING("\n  PDF skipped in dry-run mode.\n")
                )
            else:
                self.stdout.write("\n  PDF generation skipped (--no-pdf).\n")
        else:
            self.stdout.write(self.style.HTTP_INFO("\n-- Generating credentials PDF --"))
            project_root = Path(__file__).resolve().parents[5]
            pdf_path = project_root / PDF_FILENAME
            _generate_pdf(
                ward_results=ward_results,
                dept_results=dept_results,
                output_path=pdf_path,
                stdout=self.stdout,
                style=self.style,
            )

        if dry_run:
            self.stdout.write(
                self.style.WARNING("\nDry run complete. No changes were made.\n")
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "\n[OK] seed_officer_hierarchy complete.\n"
                )
            )

    # ------------------------------------------------------------------

    def _print_verification(self) -> None:
        """Query live counts and print a verification table."""
        from apps.departments.models import Department
        from apps.users.models import User, UserRole
        from apps.wards.models import Ward

        total_wards = Ward.objects.filter(is_active=True).count()
        total_depts = Department.objects.filter(is_active=True).count()

        wo_total = User.objects.filter(role=UserRole.WARD_OFFICER).count()
        wo_assigned = User.objects.filter(
            role=UserRole.WARD_OFFICER, assigned_ward__isnull=False
        ).count()
        wo_unassigned = wo_total - wo_assigned

        do_total = User.objects.filter(role=UserRole.DEPARTMENT_OFFICER).count()
        do_assigned = User.objects.filter(
            role=UserRole.DEPARTMENT_OFFICER, assigned_department__isnull=False
        ).count()
        do_unassigned = do_total - do_assigned

        # Ward-level coverage: does every active ward have ≥1 officer?
        wards_covered = (
            Ward.objects.filter(
                is_active=True,
                assigned_officers__role=UserRole.WARD_OFFICER,
            )
            .distinct()
            .count()
        )

        # Dept-level coverage
        depts_covered = (
            Department.objects.filter(
                is_active=True,
                assigned_officers__role=UserRole.DEPARTMENT_OFFICER,
            )
            .distinct()
            .count()
        )

        self.stdout.write(self.style.HTTP_INFO("\n-- Verification --"))
        self.stdout.write(f"  Active wards           : {total_wards}")
        self.stdout.write(f"  Ward officers total    : {wo_total}")
        self.stdout.write(f"  Ward officers assigned : {wo_assigned}")
        if wo_unassigned:
            self.stdout.write(
                self.style.WARNING(f"  Ward officers missing assignment : {wo_unassigned}")
            )
        self.stdout.write(f"  Wards with >=1 officer : {wards_covered} / {total_wards}")

        self.stdout.write("")
        self.stdout.write(f"  Active departments          : {total_depts}")
        self.stdout.write(f"  Dept officers total         : {do_total}")
        self.stdout.write(f"  Dept officers assigned      : {do_assigned}")
        if do_unassigned:
            self.stdout.write(
                self.style.WARNING(f"  Dept officers missing assignment : {do_unassigned}")
            )
        self.stdout.write(f"  Departments with >=1 officer: {depts_covered} / {total_depts}")

        # Pass / fail summary
        all_ok = (
            wo_assigned == total_wards
            and wards_covered == total_wards
            and do_assigned == total_depts
            and depts_covered == total_depts
        )
        if all_ok:
            self.stdout.write(
                self.style.SUCCESS(
                    "\n  [OK] All wards and departments have exactly one assigned officer."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "\n  [WARN] Some coverage gaps remain -- see counts above."
                )
            )

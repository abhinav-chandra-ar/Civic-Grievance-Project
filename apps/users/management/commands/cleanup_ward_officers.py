"""apps/users/management/commands/cleanup_ward_officers.py

Enforce the canonical ward-officer hierarchy.

Canonical rule
--------------
Exactly ONE active ward_officer per active ward.
The canonical username is:  wardofficer_<ward_code>

e.g.  wardofficer_tvm_001, wardofficer_tvm_034, wardofficer_tvm_101

All other active ward_officer users are deactivated — NOT deleted.
Their rows and FK references are preserved.
A note is written to additional_translations["legacy_reason"].

Safety constraints
------------------
* Only touches role=ward_officer users.
* Never deletes rows.
* Never touches citizen, department_officer, or admin accounts.
* Idempotent — re-running produces no further changes.

Usage
-----
    python manage.py cleanup_ward_officers
    python manage.py cleanup_ward_officers --dry-run
"""
from __future__ import annotations

import re

from django.core.management.base import BaseCommand

_TEST_EMAIL_RE = re.compile(r"@(test|example|localhost|invalid)\.", re.IGNORECASE)
_TEST_NAME_RE  = re.compile(r"^(val_|test_|debug_)", re.IGNORECASE)


def _classify(username: str, email: str, ward_code: str | None, canonical_codes: set[str]) -> str:
    """Return 'canonical', 'orphan', or 'test'."""
    if username in canonical_codes:
        return "canonical"
    if ward_code is None:
        return "orphan"
    if _TEST_EMAIL_RE.search(email) or _TEST_NAME_RE.match(username):
        return "test"
    return "legacy"


class Command(BaseCommand):
    help = (
        "Deactivate non-canonical ward_officer accounts. "
        "Keeps exactly one active officer per active ward "
        "(username = wardofficer_<ward_code>). "
        "Never deletes rows. Idempotent."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Print planned actions without writing to the database.",
        )

    def handle(self, *args, **options) -> None:
        dry_run: bool = options["dry_run"]

        from apps.users.models import User, UserRole
        from apps.wards.models import Ward

        if dry_run:
            self.stdout.write(self.style.WARNING("[DRY RUN] No changes will be written.\n"))

        # ── Build canonical username set ────────────────────────────────────
        active_wards = {
            w.code: w
            for w in Ward.objects.filter(is_active=True).order_by("code")
        }
        canonical_usernames = {f"wardofficer_{code}" for code in active_wards}

        # ── Fetch all ward_officer users ────────────────────────────────────
        all_ward_officers = list(
            User.objects.filter(role=UserRole.WARD_OFFICER)
            .select_related("assigned_ward")
            .order_by("username")
        )

        kept: list[str]               = []
        deactivated_legacy: list[str] = []
        deactivated_orphan: list[str] = []
        deactivated_test: list[str]   = []
        already_inactive: list[str]   = []

        for user in all_ward_officers:
            username  = user.username
            ward      = user.assigned_ward
            ward_code = ward.code if ward else None

            if not user.is_active:
                already_inactive.append(username)
                continue

            kind = _classify(username, user.email, ward_code, canonical_usernames)

            if kind == "canonical":
                kept.append(username)
                continue

            # Deactivate — decide bucket and reason.
            if kind == "orphan":
                reason = "orphan: no ward assigned"
                bucket = deactivated_orphan
            elif kind == "test":
                reason = (
                    f"test/validation account "
                    f"(username={username!r}, email={user.email!r})"
                )
                bucket = deactivated_test
            else:
                reason = (
                    f"legacy: non-canonical username "
                    f"(username={username!r}, ward={ward_code!r})"
                )
                bucket = deactivated_legacy

            bucket.append(username)

            if not dry_run:
                user.is_active = False
                notes = dict(user.additional_translations or {})
                notes["legacy_reason"] = reason
                user.additional_translations = notes
                user.save(update_fields=["is_active", "additional_translations"])

        # ── Report deactivated ──────────────────────────────────────────────
        deactivated_all = deactivated_legacy + deactivated_orphan + deactivated_test

        self.stdout.write(self.style.MIGRATE_HEADING("Deactivated ward officers:"))
        if deactivated_all:
            for uname in deactivated_all:
                label = (
                    "[LEGACY]"  if uname in deactivated_legacy else
                    "[ORPHAN]"  if uname in deactivated_orphan else
                    "[TEST]  "
                )
                self.stdout.write(f"  {label}  {uname}")
        else:
            self.stdout.write("  (none — already clean)")

        # ── Summary ─────────────────────────────────────────────────────────
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Summary"))
        self.stdout.write(f"  Kept active (canonical)  : {len(kept)}")
        self.stdout.write(f"  Deactivated (legacy)     : {len(deactivated_legacy)}")
        self.stdout.write(f"  Deactivated (orphan)     : {len(deactivated_orphan)}")
        self.stdout.write(f"  Deactivated (test)       : {len(deactivated_test)}")
        self.stdout.write(f"  Already inactive (skip)  : {len(already_inactive)}")

        if dry_run:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(
                "[DRY RUN] No changes written. Remove --dry-run to apply."
            ))
            return

        # ── Post-cleanup verification ────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nPost-cleanup verification"))

        active_officers = list(
            User.objects.filter(role=UserRole.WARD_OFFICER, is_active=True)
            .select_related("assigned_ward")
            .order_by("username")
        )
        expected = len(active_wards)
        actual   = len(active_officers)

        self.stdout.write(f"  Active ward_officer count : {actual}  (expected: {expected})")

        # Check for non-canonical or duplicate ward assignments.
        seen_wards: set[str] = set()
        all_ok = True
        for u in active_officers:
            ward      = u.assigned_ward
            wcode     = ward.code if ward else None
            is_canon  = u.username in canonical_usernames and wcode in active_wards
            duplicate = wcode in seen_wards if wcode else False
            if wcode:
                seen_wards.add(wcode)
            if not is_canon or duplicate:
                all_ok = False
                tag = "[UNEXPECTED]" if not is_canon else "[DUPLICATE] "
                self.stdout.write(self.style.ERROR(
                    f"  {tag} {u.username} -> {wcode}"
                ))

        # First 10 canonical officers
        self.stdout.write(self.style.MIGRATE_HEADING("\nFirst 10 canonical ward officers:"))
        for u in active_officers[:10]:
            ward = u.assigned_ward
            self.stdout.write(f"  {u.username:<28} -> {ward.name if ward else '(none)'} ({ward.code if ward else '-'})")

        self.stdout.write("")
        if all_ok and actual == expected:
            self.stdout.write(self.style.SUCCESS(
                f"[OK] Exactly {actual} active ward officers. "
                f"One per active ward. Hierarchy is canonical."
            ))
        else:
            self.stdout.write(self.style.ERROR(
                "[FAIL] Unexpected state — manual inspection required."
            ))

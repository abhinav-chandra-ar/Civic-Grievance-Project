"""apps/users/management/commands/cleanup_dept_officers.py

Enforce the canonical department-officer hierarchy after the Kerala-agency
department migration.

Canonical rule
--------------
Exactly ONE active department_officer per active department.
The canonical username is:  deptofficer_<DEPARTMENT_CODE>

e.g.  deptofficer_KSEB, deptofficer_KWA, deptofficer_PWD, deptofficer_CENGG,
      deptofficer_PH, deptofficer_REV, deptofficer_TP, deptofficer_MADM

All other department_officer users (legacy migrated accounts, orphans, test
accounts) are deactivated — NOT deleted.  Their rows and any FK references
(submitted grievances, audit logs) are preserved.

Categories handled
------------------
1. CANONICAL   username == deptofficer_<active_dept_code>  → kept active  (no change)
2. LEGACY      username is deptofficer_<old_code> but pointing at an active dept
               → deactivated; additional_translations["legacy_reason"] note added
3. ORPHAN      assigned_department is NULL
               → deactivated
4. TEST        username matches a test/validation pattern (val_*, *_test, *@test.example)
               → deactivated; additional_translations["legacy_reason"] note added

Safety constraints
------------------
* Only touches role=department_officer users.
* Never deletes rows.
* Never touches citizen or ward_officer accounts.
* Idempotent — re-running produces no further changes.

Usage
-----
    python manage.py cleanup_dept_officers
    python manage.py cleanup_dept_officers --dry-run
"""
from __future__ import annotations

import re

from django.core.management.base import BaseCommand


# Patterns that identify test / validation accounts
_TEST_USERNAME_RE = re.compile(
    r"^(val_|test_|demo_dept|debug_)",
    re.IGNORECASE,
)
_TEST_EMAIL_RE = re.compile(
    r"@(test|example|localhost|invalid)\.",
    re.IGNORECASE,
)


def _is_test_user(username: str, email: str) -> bool:
    return bool(_TEST_USERNAME_RE.match(username) or _TEST_EMAIL_RE.search(email))


class Command(BaseCommand):
    help = (
        "Deactivate non-canonical department_officer accounts. "
        "Keeps exactly one active officer per active department "
        "(username = deptofficer_<DEPT_CODE>). "
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

        from apps.departments.models import Department
        from apps.users.models import User, UserRole

        if dry_run:
            self.stdout.write(self.style.WARNING("[DRY RUN] No changes will be written.\n"))

        # ── Build canonical username set ────────────────────────────────────
        active_depts = {
            d.code: d
            for d in Department.objects.filter(is_active=True)
        }
        canonical_usernames = {
            f"deptofficer_{code}" for code in active_depts
        }

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Canonical department officers ({len(canonical_usernames)}):"
        ))
        for uname in sorted(canonical_usernames):
            code = uname[len("deptofficer_"):]
            dept = active_depts[code]
            self.stdout.write(f"  {uname:<30} -> {dept.name}")

        # ── Fetch all department_officer users ──────────────────────────────
        all_dept_officers = list(
            User.objects.filter(role=UserRole.DEPARTMENT_OFFICER)
            .select_related("assigned_department")
            .order_by("username")
        )

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\nAll department_officer users found: {len(all_dept_officers)}"
        ))

        # ── Classify and act ────────────────────────────────────────────────
        kept: list[str] = []
        deactivated_legacy: list[str] = []
        deactivated_orphan: list[str] = []
        deactivated_test: list[str] = []
        already_inactive: list[str] = []

        for user in all_dept_officers:
            username = user.username
            dept = user.assigned_department
            dept_code = dept.code if dept else None

            # Already inactive — nothing to do.
            if not user.is_active:
                already_inactive.append(username)
                self.stdout.write(f"  [SKIP]      {username:<30}  already inactive")
                continue

            # Canonical: keep active.
            if username in canonical_usernames:
                kept.append(username)
                self.stdout.write(
                    f"  [KEEP]      {username:<30}  -> {dept.name if dept else '(none)'}"
                )
                continue

            # Decide deactivation reason.
            if dept is None:
                reason = "orphan: no department assigned"
                bucket = deactivated_orphan
                tag = "[ORPHAN]   "
            elif _is_test_user(username, user.email):
                reason = f"test/validation account (username={username!r}, email={user.email!r})"
                bucket = deactivated_test
                tag = "[TEST]     "
            else:
                reason = (
                    f"legacy: migrated from old dept code "
                    f"(username={username!r} -> dept={dept_code!r})"
                )
                bucket = deactivated_legacy
                tag = "[LEGACY]   "

            bucket.append(username)
            self.stdout.write(f"  {tag} {username:<30}  reason: {reason}")

            if not dry_run:
                # Mark inactive.
                user.is_active = False
                # Store a human-readable note in additional_translations
                # so the reason is never lost (no schema change needed).
                notes = dict(user.additional_translations or {})
                notes["legacy_reason"] = reason
                user.additional_translations = notes
                user.save(update_fields=["is_active", "additional_translations"])

        # ── Summary ─────────────────────────────────────────────────────────
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Summary"))
        self.stdout.write(f"  Kept active (canonical)  : {len(kept)}")
        self.stdout.write(f"  Deactivated (legacy)     : {len(deactivated_legacy)}")
        self.stdout.write(f"  Deactivated (orphan)     : {len(deactivated_orphan)}")
        self.stdout.write(f"  Deactivated (test)       : {len(deactivated_test)}")
        self.stdout.write(f"  Already inactive (skip)  : {len(already_inactive)}")

        total_deactivated = (
            len(deactivated_legacy) + len(deactivated_orphan) + len(deactivated_test)
        )

        if dry_run:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(
                "[DRY RUN] No changes written. Remove --dry-run to apply."
            ))
            return

        # ── Post-cleanup verification ────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nPost-cleanup verification"))
        active_officers = list(
            User.objects.filter(
                role=UserRole.DEPARTMENT_OFFICER,
                is_active=True,
            )
            .select_related("assigned_department")
            .order_by("username")
        )

        self.stdout.write(
            f"  Active department_officer users: {len(active_officers)}"
            f"  (expected: {len(canonical_usernames)})"
        )

        all_ok = True
        for u in active_officers:
            dept = u.assigned_department
            is_canonical = u.username in canonical_usernames and dept is not None and dept.is_active
            tag = "[OK]" if is_canonical else "[UNEXPECTED]"
            if not is_canonical:
                all_ok = False
            self.stdout.write(
                f"  {tag} {u.username:<30} -> {str(dept) if dept else '(none)'}"
            )

        # Check each active dept has exactly one officer.
        for code, dept in sorted(active_depts.items()):
            expected_user = f"deptofficer_{code}"
            exists = any(u.username == expected_user for u in active_officers)
            if not exists:
                self.stdout.write(
                    self.style.ERROR(f"  [MISSING] No active canonical officer for {code}!")
                )
                all_ok = False

        self.stdout.write("")
        if all_ok and len(active_officers) == len(canonical_usernames):
            self.stdout.write(self.style.SUCCESS(
                f"[OK] Exactly {len(active_officers)} active department officers. "
                f"One per active department. Hierarchy is canonical."
            ))
        else:
            self.stdout.write(self.style.ERROR(
                "[FAIL] Unexpected state — manual inspection required."
            ))

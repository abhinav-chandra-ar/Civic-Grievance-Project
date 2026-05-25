"""apps/departments/management/commands/migrate_to_kerala_agencies.py

One-time (idempotent) data migration:
  Replace generic placeholder departments with real Kerala civic agencies.

New department roster
---------------------
KSEB   Kerala State Electricity Board         electrical_hazard, street_light
KWA    Kerala Water Authority                 water_supply, sewage_issue
PWD    Kerala Public Works Department         (state roads — override via keyword)
CENGG  Corporation Engineering Department     road_damage, drainage, tree_fall
PH     Public Health / Sanitation Department  waste_management, solid_waste
REV    Revenue Department
TP     Town Planning Department               illegal_construction
MADM   Municipal Administration

Old → new code mapping
-----------------------
PWE  -> KSEB   (electrical engineering wing → KSEB)
SL   -> KSEB   (street lighting → KSEB, which operates Kerala street lights)
WS   -> KWA    (water supply → Kerala Water Authority)
SWM  -> PH     (sanitation/waste → Public Health; sewage_issue re-routed to KWA via ML)
RD   -> CENGG  (roads & drainage → Corporation Engineering)
ESW  -> CENGG  (environment/solid waste/tree fall → Corporation Engineering)
BTP  -> TP     (building & town planning → Town Planning)
GA   -> MADM   (general admin → Municipal Administration)
MA   -> MADM   (municipal admin → Municipal Administration, merge)
HPH  -> PH     (health/public health → Public Health, merge)
RTL  -> REV    (revenue/tax → Revenue Department)

Also maps legacy snake_case codes from seed_demo_data.py:
  roads_and_drainage  -> CENGG
  sanitation          -> PH
  water_authority     -> KWA
  street_lighting     -> KSEB
  parks_and_environment -> CENGG
  building_permit_office -> TP
  electrical_engineering -> KSEB

Idempotency
-----------
Safe to run multiple times.  New departments are created with get_or_create.
FK migrations check current FK values before updating.
Old departments are deactivated (not deleted) to preserve audit history.

Usage
-----
    python manage.py migrate_to_kerala_agencies
    python manage.py migrate_to_kerala_agencies --dry-run
"""
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction


# ---------------------------------------------------------------------------
# New Kerala civic agency definitions
# ---------------------------------------------------------------------------

_NEW_DEPARTMENTS: list[dict] = [
    {
        "code": "KSEB",
        "name": "Kerala State Electricity Board",
        "handled_categories": ["electrical_hazard", "street_light"],
        "translated_names": {"ml": "കേരള സ്റ്റേറ്റ് ഇലക്ട്രിസിറ്റി ബോർഡ്"},
    },
    {
        "code": "KWA",
        "name": "Kerala Water Authority",
        "handled_categories": ["water_supply", "sewage_issue"],
        "translated_names": {"ml": "കേരള ജല അതോറിറ്റി"},
    },
    {
        "code": "PWD",
        "name": "Kerala Public Works Department",
        "handled_categories": [],
        "translated_names": {"ml": "കേരള പൊതുമരാമത്ത് വകുപ്പ്"},
    },
    {
        "code": "CENGG",
        "name": "Corporation Engineering Department",
        "handled_categories": ["road_damage", "drainage", "tree_fall"],
        "translated_names": {"ml": "കോർപ്പറേഷൻ എഞ്ചിനീയറിംഗ് വിഭാഗം"},
    },
    {
        "code": "PH",
        "name": "Public Health / Sanitation Department",
        "handled_categories": ["waste_management", "solid_waste"],
        "translated_names": {"ml": "പൊതു ആരോഗ്യ / ശുചിത്വ വിഭാഗം"},
    },
    {
        "code": "REV",
        "name": "Revenue Department",
        "handled_categories": [],
        "translated_names": {"ml": "റവന്യൂ വകുപ്പ്"},
    },
    {
        "code": "TP",
        "name": "Town Planning Department",
        "handled_categories": ["illegal_construction"],
        "translated_names": {"ml": "ടൗൺ പ്ലാനിംഗ് വകുപ്പ്"},
    },
    {
        "code": "MADM",
        "name": "Municipal Administration",
        "handled_categories": [],
        "translated_names": {"ml": "മുനിസിപ്പൽ ഭരണം"},
    },
]

# Maps any old department code → new department code.
# Covers both short uppercase codes (from seed_officer_hierarchy) and
# long snake_case codes (from seed_demo_data, if ever run).
_OLD_TO_NEW: dict[str, str] = {
    # Short uppercase (current DB state)
    "PWE": "KSEB",
    "SL":  "KSEB",
    "WS":  "KWA",
    "SWM": "PH",
    "RD":  "CENGG",
    "ESW": "CENGG",
    "BTP": "TP",
    "GA":  "MADM",
    "MA":  "MADM",
    "HPH": "PH",
    "RTL": "REV",
    # Long snake_case (from seed_demo_data, if it was ever run)
    "roads_and_drainage":  "CENGG",
    "sanitation":          "PH",
    "water_authority":     "KWA",
    "street_lighting":     "KSEB",
    "parks_and_environment": "CENGG",
    "building_permit_office": "TP",
    "electrical_engineering": "KSEB",
}


class Command(BaseCommand):
    help = (
        "Migrate department records from generic placeholder names to real "
        "Kerala civic agencies (KSEB, KWA, PWD, CENGG, PH, REV, TP, MADM). "
        "Idempotent — safe to run multiple times."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Preview changes without writing to the database.",
        )

    def handle(self, *args, **options) -> None:
        dry_run: bool = options["dry_run"]

        from apps.departments.models import Department
        from apps.grievances.models import Grievance
        from apps.users.models import User

        if dry_run:
            self.stdout.write(self.style.WARNING("[DRY RUN] No changes will be written.\n"))

        # ------------------------------------------------------------------
        # 1. Create / verify new departments
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("-- Step 1: Create new Kerala agency departments"))
        new_dept_map: dict[str, Department] = {}

        for spec in _NEW_DEPARTMENTS:
            if not dry_run:
                obj, created = Department.objects.get_or_create(
                    code=spec["code"],
                    defaults={
                        "name": spec["name"],
                        "translated_names": spec["translated_names"],
                        "handled_categories": spec["handled_categories"],
                        "is_active": True,
                    },
                )
                if not created:
                    # Ensure name and categories are up to date even if already exists.
                    updated_fields = []
                    if obj.name != spec["name"]:
                        obj.name = spec["name"]
                        updated_fields.append("name")
                    if obj.handled_categories != spec["handled_categories"]:
                        obj.handled_categories = spec["handled_categories"]
                        updated_fields.append("handled_categories")
                    if obj.translated_names != spec["translated_names"]:
                        obj.translated_names = spec["translated_names"]
                        updated_fields.append("translated_names")
                    if not obj.is_active:
                        obj.is_active = True
                        updated_fields.append("is_active")
                    if updated_fields:
                        obj.save(update_fields=updated_fields)
                new_dept_map[spec["code"]] = obj
                tag = "[CREATED]" if created else "[EXISTS] "
            else:
                tag = "[DRY-RUN]"
                obj = None  # type: ignore[assignment]
            self.stdout.write(f"  {tag} {spec['code']:8} {spec['name']}")

        # ------------------------------------------------------------------
        # 2. Migrate Grievance.department FKs
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\n-- Step 2: Migrate Grievance.department FK references"))
        grievance_updates = 0

        if not dry_run:
            for old_code, new_code in _OLD_TO_NEW.items():
                new_dept = new_dept_map.get(new_code)
                if new_dept is None:
                    continue
                count = Grievance.objects.filter(
                    department__code=old_code
                ).update(department=new_dept)
                if count:
                    self.stdout.write(f"  Updated {count} grievance(s): dept {old_code} -> {new_code}")
                    grievance_updates += count

        if grievance_updates == 0:
            self.stdout.write("  No grievance FK updates needed (already clean or dry-run).")

        # ------------------------------------------------------------------
        # 3. Migrate User.assigned_department FKs
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\n-- Step 3: Migrate User.assigned_department FK references"))
        user_updates = 0

        if not dry_run:
            for old_code, new_code in _OLD_TO_NEW.items():
                new_dept = new_dept_map.get(new_code)
                if new_dept is None:
                    continue
                count = User.objects.filter(
                    assigned_department__code=old_code
                ).update(assigned_department=new_dept)
                if count:
                    self.stdout.write(
                        f"  Updated {count} user(s): assigned_department {old_code} -> {new_code}"
                    )
                    user_updates += count

        if user_updates == 0:
            self.stdout.write("  No user FK updates needed (already clean or dry-run).")

        # ------------------------------------------------------------------
        # 4. Deactivate old departments (preserves audit trail)
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\n-- Step 4: Deactivate old placeholder departments"))
        old_codes = list(_OLD_TO_NEW.keys())

        if not dry_run:
            deactivated = Department.objects.filter(
                code__in=old_codes,
                is_active=True,
            ).update(is_active=False)
            self.stdout.write(f"  Deactivated {deactivated} old department record(s).")
        else:
            count = Department.objects.filter(code__in=old_codes, is_active=True).count()
            self.stdout.write(f"  [DRY-RUN] Would deactivate {count} old department(s): {old_codes}")

        # ------------------------------------------------------------------
        # 5. Verification
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\n-- Step 5: Verification"))

        if not dry_run:
            active_depts = Department.objects.filter(is_active=True).order_by("code")
            self.stdout.write(f"\n  Active departments after migration ({active_depts.count()}):")
            for d in active_depts:
                self.stdout.write(f"    {d.code:8}  {d.name:<45}  cats={d.handled_categories}")

            orphan_g = Grievance.objects.filter(
                department__isnull=False,
                department__is_active=False,
            ).count()
            orphan_u = User.objects.filter(
                assigned_department__isnull=False,
                assigned_department__is_active=False,
            ).count()
            if orphan_g or orphan_u:
                self.stdout.write(
                    self.style.WARNING(
                        f"\n  [WARN] {orphan_g} grievance(s), {orphan_u} user(s) "
                        f"still point to INACTIVE departments. "
                        f"Re-run the command or fix manually."
                    )
                )
            else:
                self.stdout.write(
                    self.style.SUCCESS("\n  [OK] No FK references to inactive departments.")
                )

        self.stdout.write("")
        if dry_run:
            self.stdout.write(self.style.WARNING("[DRY RUN] No changes were written. Remove --dry-run to apply."))
        else:
            self.stdout.write(self.style.SUCCESS(
                "Migration complete. Run `python manage.py seed_officer_hierarchy` "
                "to regenerate department officer accounts for the new agencies."
            ))

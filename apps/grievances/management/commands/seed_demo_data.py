"""apps/grievances/management/commands/seed_demo_data.py

Django management command: seed_demo_data

Idempotently seeds the database with realistic TVMC civic grievance demo data:

    • 7 municipal departments (roads, sanitation, water, lighting, parks,
      building permits, electrical engineering)
    • 7 demo users  (3 citizens, 2 ward officers, 1 dept officer, 1 admin)
    • 15 demo grievances in English, Malayalam, and Manglish
    • 1 SLA record per grievance (if not already present)

Idempotency
-----------
All records are created with get_or_create keyed on unique natural keys
(department.code, user.username, grievance.tracking_code, sla.grievance).
Re-running the command never duplicates data.

Demo tracking codes use the GRV-2024-9XXXXX range to avoid clashing with
any real submissions in the same year.

Usage
-----
    python manage.py seed_demo_data
    python manage.py seed_demo_data --silent   (suppress per-object messages)
"""
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.utils import timezone


# ---------------------------------------------------------------------------
# Data tables
# ---------------------------------------------------------------------------

_DEPARTMENTS = [
    {
        "code": "roads_and_drainage",
        "name": "Roads & Drainage Department",
        "translated_names": {"ml": "റോഡ്, ഡ്രെയിനേജ് വിഭാഗം"},
        "handled_categories": ["road_damage", "drainage", "sewage_issue"],
    },
    {
        "code": "sanitation",
        "name": "Sanitation Department",
        "translated_names": {"ml": "ശുചിത്വ വിഭാഗം"},
        "handled_categories": ["waste_management", "solid_waste"],
    },
    {
        "code": "water_authority",
        "name": "Water Authority",
        "translated_names": {"ml": "ജല അതോറിറ്റി"},
        "handled_categories": ["water_supply"],
    },
    {
        "code": "street_lighting",
        "name": "Street Lighting Department",
        "translated_names": {"ml": "തെരുവ് വിളക്ക് വിഭാഗം"},
        "handled_categories": ["street_light"],
    },
    {
        "code": "parks_and_environment",
        "name": "Parks & Environment Department",
        "translated_names": {"ml": "പാർക്ക്, പരിസ്ഥിതി വിഭാഗം"},
        "handled_categories": ["tree_fall"],
    },
    {
        "code": "building_permit_office",
        "name": "Building Permit Office",
        "translated_names": {"ml": "കെട്ടിട അനുമതി ഓഫീസ്"},
        "handled_categories": ["illegal_construction"],
    },
    {
        "code": "electrical_engineering",
        "name": "Electrical Engineering Department",
        "translated_names": {"ml": "ഇലക്ട്രിക്കൽ എഞ്ചിനീയറിംഗ് വിഭാഗം"},
        "handled_categories": ["electrical_hazard"],
    },
]

_USERS = [
    # Citizens
    {
        "username": "demo_citizen_rajan",
        "password": "Demo@1234",
        "first_name": "Rajan",
        "last_name": "Kumar",
        "email": "rajan.demo@tvmc.gov.in",
        "role": "citizen",
        "preferred_language": "ml",
    },
    {
        "username": "demo_citizen_priya",
        "password": "Demo@1234",
        "first_name": "Priya",
        "last_name": "Nair",
        "email": "priya.demo@tvmc.gov.in",
        "role": "citizen",
        "preferred_language": "en",
    },
    {
        "username": "demo_citizen_ahmed",
        "password": "Demo@1234",
        "first_name": "Ahmed",
        "last_name": "Siddiqui",
        "email": "ahmed.demo@tvmc.gov.in",
        "role": "citizen",
        "preferred_language": "en",
    },
    # Ward officers
    {
        "username": "demo_officer_vishnu",
        "password": "Demo@1234",
        "first_name": "Vishnu",
        "last_name": "Menon",
        "email": "vishnu.officer@tvmc.gov.in",
        "role": "ward_officer",
        "preferred_language": "ml",
    },
    {
        "username": "demo_officer_deepa",
        "password": "Demo@1234",
        "first_name": "Deepa",
        "last_name": "Krishnan",
        "email": "deepa.officer@tvmc.gov.in",
        "role": "ward_officer",
        "preferred_language": "en",
    },
    # Department officer
    {
        "username": "demo_dept_officer_suresh",
        "password": "Demo@1234",
        "first_name": "Suresh",
        "last_name": "Babu",
        "email": "suresh.dept@tvmc.gov.in",
        "role": "department_officer",
        "preferred_language": "en",
    },
    # Municipal admin
    {
        "username": "demo_admin_anitha",
        "password": "Demo@1234",
        "first_name": "Anitha",
        "last_name": "Raghavan",
        "email": "anitha.admin@tvmc.gov.in",
        "role": "municipal_admin",
        "preferred_language": "en",
    },
]

# Grievance rows — tracking code, submitter username, text, priority, status, category
# Codes use GRV-2024-9XXXXX range (demo-reserved, won't clash with real submissions)
_GRIEVANCES = [
    # ── English ─────────────────────────────────────────────────────────────
    {
        "tracking_code": "GRV-2024-900001",
        "submitter": "demo_citizen_rajan",
        "raw_text": (
            "There is a large pothole on the main road near Pattom junction. "
            "It has been there for over two weeks and is causing accidents."
        ),
        "priority": "high",
        "status": "assigned",
        "category_code": "road_damage",
        "dept_code": "roads_and_drainage",
    },
    {
        "tracking_code": "GRV-2024-900002",
        "submitter": "demo_citizen_priya",
        "raw_text": (
            "No water supply in our locality near Vellayambalam for the past two days. "
            "Residents are suffering without drinking water."
        ),
        "priority": "high",
        "status": "in_progress",
        "category_code": "water_supply",
        "dept_code": "water_authority",
    },
    {
        "tracking_code": "GRV-2024-900003",
        "submitter": "demo_citizen_ahmed",
        "raw_text": (
            "Street light not working on MG Road near Statue junction for the past week. "
            "The area is completely dark at night and it is unsafe."
        ),
        "priority": "medium",
        "status": "triaged",
        "category_code": "street_light",
        "dept_code": "street_lighting",
    },
    {
        "tracking_code": "GRV-2024-900004",
        "submitter": "demo_citizen_rajan",
        "raw_text": (
            "A large tree has fallen on the road near Thampanoor blocking all traffic. "
            "Emergency clearance needed immediately."
        ),
        "priority": "urgent",
        "status": "in_progress",
        "category_code": "tree_fall",
        "dept_code": "parks_and_environment",
    },
    {
        "tracking_code": "GRV-2024-900005",
        "submitter": "demo_citizen_priya",
        "raw_text": (
            "Electric wire fell on the road near Bakery Junction. "
            "Wire is sparking and very dangerous. Children are playing nearby."
        ),
        "priority": "critical",
        "status": "assigned",
        "category_code": "electrical_hazard",
        "dept_code": "electrical_engineering",
    },
    {
        "tracking_code": "GRV-2024-900006",
        "submitter": "demo_citizen_ahmed",
        "raw_text": (
            "Sewage overflowing onto the road near Kesavadasapuram. "
            "Extremely foul smell and health hazard for local residents."
        ),
        "priority": "urgent",
        "status": "triaged",
        "category_code": "sewage_issue",
        "dept_code": "roads_and_drainage",
    },
    {
        "tracking_code": "GRV-2024-900007",
        "submitter": "demo_citizen_rajan",
        "raw_text": (
            "Drain blocked near Karamana junction. Severe water logging after rains. "
            "Mosquito breeding is a health concern."
        ),
        "priority": "high",
        "status": "submitted",
        "category_code": "drainage",
        "dept_code": "roads_and_drainage",
    },
    {
        "tracking_code": "GRV-2024-900008",
        "submitter": "demo_citizen_priya",
        "raw_text": (
            "Illegal construction happening near Kowdiar without any permit. "
            "The building encroaches onto the public road."
        ),
        "priority": "medium",
        "status": "submitted",
        "category_code": "illegal_construction",
        "dept_code": "building_permit_office",
    },
    # ── Malayalam ────────────────────────────────────────────────────────────
    {
        "tracking_code": "GRV-2024-900009",
        "submitter": "demo_citizen_rajan",
        "raw_text": (
            "റോഡിൽ വലിയ കുഴി ഉണ്ട്. വഴിമദ്ധ്യത്ത് വെള്ളം കെട്ടി നിൽക്കുന്നു. "
            "വാഹനങ്ങൾക്ക് കടന്നു പോകാൻ ബുദ്ധിമുട്ടാണ്."
        ),
        "priority": "high",
        "status": "triaged",
        "category_code": "road_damage",
        "dept_code": "roads_and_drainage",
    },
    {
        "tracking_code": "GRV-2024-900010",
        "submitter": "demo_citizen_rajan",
        "raw_text": (
            "കുടിവെള്ളം കിട്ടുന്നില്ല. രണ്ടു ദിവസമായി വെള്ളം ഇല്ല. "
            "ജലലഭ്യത ഉടൻ പുനഃസ്ഥാപിക്കണം."
        ),
        "priority": "high",
        "status": "submitted",
        "category_code": "water_supply",
        "dept_code": "water_authority",
    },
    {
        "tracking_code": "GRV-2024-900011",
        "submitter": "demo_citizen_rajan",
        "raw_text": (
            "ഓടയിൽ നിന്ന് ദുർഗന്ധം. ഓട ഒഴുകുന്നില്ല. "
            "ആരോഗ്യ പ്രശ്നം ഉണ്ടാകുന്നു. ഉടൻ ശ്രദ്ധിക്കണം."
        ),
        "priority": "urgent",
        "status": "submitted",
        "category_code": "sewage_issue",
        "dept_code": "roads_and_drainage",
    },
    {
        "tracking_code": "GRV-2024-900012",
        "submitter": "demo_citizen_rajan",
        "raw_text": (
            "തെരുവ് വിളക്ക് കത്തുന്നില്ല. ഒരാഴ്ചയായി ഇരുട്ടാണ്. "
            "രാത്രി നടക്കാൻ ബുദ്ധിമുട്ടാണ്."
        ),
        "priority": "medium",
        "status": "triaged",
        "category_code": "street_light",
        "dept_code": "street_lighting",
    },
    # ── Manglish ─────────────────────────────────────────────────────────────
    {
        "tracking_code": "GRV-2024-900013",
        "submitter": "demo_citizen_ahmed",
        "raw_text": (
            "Road il valiya kuzhi und. Bike pokunilla. "
            "Accident aavum ennu bhayam und. Utharam tharenam."
        ),
        "priority": "high",
        "status": "submitted",
        "category_code": "road_damage",
        "dept_code": "roads_and_drainage",
    },
    {
        "tracking_code": "GRV-2024-900014",
        "submitter": "demo_citizen_ahmed",
        "raw_text": (
            "Vilakku kattunilla. Rathri neram vazhi iruttu aanu. "
            "Aniyarayanu safety um illa."
        ),
        "priority": "medium",
        "status": "submitted",
        "category_code": "street_light",
        "dept_code": "street_lighting",
    },
    # ── Resolved / Closed (history demo) ────────────────────────────────────
    {
        "tracking_code": "GRV-2024-900015",
        "submitter": "demo_citizen_priya",
        "raw_text": (
            "Garbage pile near Palayam market not cleared for three days. "
            "Smell is unbearable and causing inconvenience to shoppers."
        ),
        "priority": "medium",
        "status": "resolved",
        "category_code": "waste_management",
        "dept_code": "sanitation",
    },
]

# SLA window deltas per priority (mirrors services.DEFAULT_SLA_DEADLINE_DELTAS)
_SLA_DAYS = {
    "low": 7,
    "medium": 5,
    "high": 3,
    "urgent": 1,
    "critical": 0,   # critical uses hours, handled below
}
_SLA_HOURS_CRITICAL = 12


class Command(BaseCommand):
    help = (
        "Idempotently seed the database with demo departments, users, grievances, "
        "and SLA records for development and demonstration purposes."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--silent",
            action="store_true",
            default=False,
            help="Suppress per-object creation messages.",
        )

    def handle(self, *args, **options) -> None:
        silent: bool = options["silent"]
        now = timezone.now()

        def log(msg: str) -> None:
            if not silent:
                self.stdout.write(msg)

        created_counts = {"departments": 0, "users": 0, "grievances": 0, "slas": 0}

        # ── 1. Departments ────────────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("━━ Departments"))
        from apps.departments.models import Department  # noqa: PLC0415

        dept_map: dict[str, Department] = {}
        for d in _DEPARTMENTS:
            obj, created = Department.objects.get_or_create(
                code=d["code"],
                defaults={
                    "name": d["name"],
                    "translated_names": d["translated_names"],
                    "handled_categories": d["handled_categories"],
                    "is_active": True,
                },
            )
            dept_map[d["code"]] = obj
            if created:
                created_counts["departments"] += 1
                log(f"  [CREATED] Department: {obj.name} ({obj.code})")
            else:
                log(f"  [EXISTS]  Department: {obj.name} ({obj.code})")

        # ── 2. Users ──────────────────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\n━━ Users"))
        from apps.users.models import User  # noqa: PLC0415

        user_map: dict[str, User] = {}
        for u in _USERS:
            obj, created = User.objects.get_or_create(
                username=u["username"],
                defaults={
                    "first_name": u["first_name"],
                    "last_name": u["last_name"],
                    "email": u["email"],
                    "role": u["role"],
                    "preferred_language": u["preferred_language"],
                    "is_active": True,
                },
            )
            if created:
                obj.set_password(u["password"])
                obj.save(update_fields=["password"])
                created_counts["users"] += 1
                log(f"  [CREATED] User: {obj.username}  role={obj.role}")
            else:
                log(f"  [EXISTS]  User: {obj.username}  role={obj.role}")
            user_map[u["username"]] = obj

        # ── 3. Grievances ─────────────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\n━━ Grievances"))
        from apps.grievances.models import Grievance  # noqa: PLC0415

        grievance_map: dict[str, Grievance] = {}
        for g in _GRIEVANCES:
            submitter = user_map[g["submitter"]]
            dept = dept_map.get(g["dept_code"])
            submitted_at = now - timezone.timedelta(days=_GRIEVANCES.index(g) + 1)
            obj, created = Grievance.objects.get_or_create(
                tracking_code=g["tracking_code"],
                defaults={
                    "submitter": submitter,
                    "raw_text": g["raw_text"],
                    "priority": g["priority"],
                    "status": g["status"],
                    "category_code": g["category_code"],
                    "department": dept,
                    "submitted_at": submitted_at,
                    "last_status_changed_at": submitted_at,
                },
            )
            grievance_map[g["tracking_code"]] = obj
            if created:
                created_counts["grievances"] += 1
                log(
                    f"  [CREATED] {obj.tracking_code}  "
                    f"cat={g['category_code']}  prio={g['priority']}  status={g['status']}"
                )
            else:
                log(
                    f"  [EXISTS]  {obj.tracking_code}  "
                    f"cat={g['category_code']}  prio={g['priority']}"
                )

        # ── 4. SLA records ────────────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\n━━ SLA records"))
        from apps.slas.models import SLA, SLAStatus  # noqa: PLC0415

        _SLA_SEQ_BASE = 900000  # demo-reserved sequence range

        for idx, g in enumerate(_GRIEVANCES, start=1):
            grievance = grievance_map[g["tracking_code"]]
            if SLA.objects.filter(grievance=grievance).exists():
                log(f"  [EXISTS]  SLA for {g['tracking_code']}")
                continue

            priority = g["priority"]
            submitted_at = grievance.submitted_at or now
            if priority == "critical":
                response_due = submitted_at + timezone.timedelta(hours=4)
                resolution_due = submitted_at + timezone.timedelta(hours=_SLA_HOURS_CRITICAL)
            else:
                days = _SLA_DAYS.get(priority, 5)
                response_due = submitted_at + timezone.timedelta(hours=24)
                resolution_due = submitted_at + timezone.timedelta(days=days)

            sla_code = f"SLA-2024-{_SLA_SEQ_BASE + idx:06d}"
            sla_status_val = (
                SLAStatus.SATISFIED if g["status"] in {"resolved", "closed"}
                else SLAStatus.ACTIVE
            )

            SLA.objects.create(
                sla_code=sla_code,
                grievance=grievance,
                response_due_at=response_due,
                resolution_due_at=resolution_due,
                sla_status=sla_status_val,
                is_breached=False,
            )
            created_counts["slas"] += 1
            log(f"  [CREATED] {sla_code}  grievance={g['tracking_code']}")

        # ── Summary ───────────────────────────────────────────────────────
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"Seed complete — "
            f"{created_counts['departments']} dept(s), "
            f"{created_counts['users']} user(s), "
            f"{created_counts['grievances']} grievance(s), "
            f"{created_counts['slas']} SLA record(s) created."
        ))
        self.stdout.write("")
        self.stdout.write("Demo login credentials (password: Demo@1234)")
        self.stdout.write("  citizen  : demo_citizen_rajan / demo_citizen_priya / demo_citizen_ahmed")
        self.stdout.write("  officer  : demo_officer_vishnu / demo_officer_deepa")
        self.stdout.write("  dept off : demo_dept_officer_suresh")
        self.stdout.write("  admin    : demo_admin_anitha")

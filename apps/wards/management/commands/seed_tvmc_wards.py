"""Management command: seed_tvmc_wards

Seed the full 101-ward Thiruvananthapuram Municipal Corporation (TVMC)
registry using the official Kerala State Election Commission 2025
delimitation ward list (published via LSG Kerala / tmc.lsgkerala.gov.in).

Ward codes
----------
All codes follow the WARD_CODE_VALIDATOR-compliant tvm_NNN format
(regex: ^[a-z][a-z0-9_]*$).  Uppercase codes (e.g. TVM01) are NOT used.

Boundaries
----------
Every ward receives the same placeholder polygon covering the approximate
TVMC bounding box (EPSG:4326).  Replace per-ward boundaries with real
GeoJSON data using the import_tvmc_wards command once authoritative
boundary data is available.

Two-phase execution
-------------------
Phase 1  — build and full_clean() all 101 Ward objects.  No DB writes.
Phase 2  — delete existing rows (--replace) and insert all validated wards
           inside a single transaction.atomic().

Usage
-----
    python manage.py seed_tvmc_wards                # fails if rows exist
    python manage.py seed_tvmc_wards --replace      # delete-all then re-seed
    python manage.py seed_tvmc_wards --dry-run      # validate only, no writes
"""
from __future__ import annotations

from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.wards.models import Ward

# ---------------------------------------------------------------------------
# Placeholder boundary
# A single rectangle covering the approximate TVMC bounding box (EPSG:4326).
# All 101 wards share this polygon until real boundaries are imported via
# the import_tvmc_wards command.
# ---------------------------------------------------------------------------

_PLACEHOLDER_WKT = (
    "POLYGON ((76.9 8.48, 77.02 8.48, 77.02 8.6, 76.9 8.6, 76.9 8.48))"
)

# ---------------------------------------------------------------------------
# Official TVMC ward registry
#
# Source : Kerala State Election Commission 2025 delimitation /
#          LSG Kerala official ward list (tmc.lsgkerala.gov.in)
#          Cross-verified against Wikipedia "Thiruvananthapuram Municipal
#          Corporation" ward table (retrieved 2026-05-23).
#
# Format : (code, english_name)
#   code = tvm_NNN  (three-digit zero-padded ward number, fully lowercase)
#   name = ward name in English as gazetted
#
# DO NOT edit ward names without an authoritative source reference.
# ---------------------------------------------------------------------------

TVMC_WARDS: tuple[tuple[str, str], ...] = (
    ("tvm_001", "Kazhakkoottam"),
    ("tvm_002", "Sainika School"),
    ("tvm_003", "Chanthavila"),
    ("tvm_004", "Kattaikonam"),
    ("tvm_005", "Njandoorkonam"),
    ("tvm_006", "Powdikonam"),
    ("tvm_007", "Chenkottukonam"),
    ("tvm_008", "Chempazhanthy"),
    ("tvm_009", "Kariavattom"),
    ("tvm_010", "Pangappara"),
    ("tvm_011", "Sreekariyam"),
    ("tvm_012", "Chellamangalam"),
    ("tvm_013", "Mannanthala"),
    ("tvm_014", "Pathirapalli"),
    ("tvm_015", "Ambalamukku"),
    ("tvm_016", "Kudappanakunnu"),
    ("tvm_017", "Thuruthummoola"),
    ("tvm_018", "Nettayam"),
    ("tvm_019", "Kachani"),
    ("tvm_020", "Vazhottukonam"),
    ("tvm_021", "Kodunganoor"),
    ("tvm_022", "Vattiyoorkavu"),
    ("tvm_023", "Kanjirampara"),
    ("tvm_024", "Peroorkada"),
    ("tvm_025", "Kowdiar"),
    ("tvm_026", "Kuravankonam"),
    ("tvm_027", "Muttada"),
    ("tvm_028", "Chettivilakam"),
    ("tvm_029", "Kinavoor"),
    ("tvm_030", "Nalanchira"),
    ("tvm_031", "Edavakode"),
    ("tvm_032", "Ulloor"),
    ("tvm_033", "Medical College"),
    ("tvm_034", "Pattom"),
    ("tvm_035", "Kesavadasapuram"),
    ("tvm_036", "Gowreeshapattom"),
    ("tvm_037", "Kunnukuzhy"),
    ("tvm_038", "Nanthancode"),
    ("tvm_039", "Palayam"),
    ("tvm_040", "Vazhuthacaud"),
    ("tvm_041", "Sasthamangalam"),
    ("tvm_042", "Pangode"),
    ("tvm_043", "Thirumala"),
    ("tvm_044", "Valiyavila"),
    ("tvm_045", "Thrikkannapuram"),
    ("tvm_046", "Punnakkamugal"),
    ("tvm_047", "Poojappura"),
    ("tvm_048", "Jagathy"),
    ("tvm_049", "Thycaud"),
    ("tvm_050", "Valiyasala"),
    ("tvm_051", "Arannoor"),
    ("tvm_052", "Mudavanmugal"),
    ("tvm_053", "Estate"),
    ("tvm_054", "Nemom"),
    ("tvm_055", "Ponnumangalam"),
    ("tvm_056", "Melamcode"),
    ("tvm_057", "Pappanamcode"),
    ("tvm_058", "Karamana"),
    ("tvm_059", "Nedumcaud"),
    ("tvm_060", "Kaladi"),
    ("tvm_061", "Karumom"),
    ("tvm_062", "Punchakkari"),
    ("tvm_063", "Poonkulam"),
    ("tvm_064", "Venganoor"),
    ("tvm_065", "Port"),
    ("tvm_066", "Vizhinjam"),
    ("tvm_067", "Harbour"),
    ("tvm_068", "Vellar"),
    ("tvm_069", "Thiruvallam"),
    ("tvm_070", "Poonthura"),
    ("tvm_071", "Puthenppalli"),
    ("tvm_072", "Ambalathara"),
    ("tvm_073", "Attukal"),
    ("tvm_074", "Kalippankulam"),
    ("tvm_075", "Kamaleswaram"),
    ("tvm_076", "Beemapalli"),
    ("tvm_077", "Valiyathura"),
    ("tvm_078", "Vallakkadavu"),
    ("tvm_079", "Sreevaraham"),
    ("tvm_080", "Manacaud"),
    ("tvm_081", "Chalai"),
    ("tvm_082", "Fort"),
    ("tvm_083", "Perunthanni"),
    ("tvm_084", "Sreekanteswaram"),
    ("tvm_085", "Thampanoor"),
    ("tvm_086", "Vanchiyoor"),
    ("tvm_087", "Kannammoola"),
    ("tvm_088", "Pettah"),
    ("tvm_089", "Chackai"),
    ("tvm_090", "Vettukadu"),
    ("tvm_091", "Karikkakam"),
    ("tvm_092", "Kadakampally"),
    ("tvm_093", "Anamugham"),
    ("tvm_094", "Akkulam"),
    ("tvm_095", "Cheruvaikkal"),
    ("tvm_096", "Alathara"),
    ("tvm_097", "Kuzhivila"),
    ("tvm_098", "Poundkadavu"),
    ("tvm_099", "Kulathoor"),
    ("tvm_100", "Attipra"),
    ("tvm_101", "Pallithura"),
)

# Guard: fail at import time if the list is ever accidentally truncated.
assert len(TVMC_WARDS) == 101, (
    f"TVMC_WARDS must contain exactly 101 entries; found {len(TVMC_WARDS)}."
)


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------


class Command(BaseCommand):
    help = (
        "Seed the full 101-ward TVMC registry using the official Kerala SEC\n"
        "2025 delimitation ward list.  All wards receive a shared placeholder\n"
        "polygon boundary — replace with real data via import_tvmc_wards."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--replace",
            action="store_true",
            default=False,
            help=(
                "Delete ALL existing ward rows and re-seed with the full "
                "101-ward registry.  Without this flag the command aborts "
                "if any wards already exist."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help=(
                "Run Phase 1 validation and print what would be written "
                "without making any database changes."
            ),
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle(self, *args, **options) -> None:
        replace: bool = options["replace"]
        dry_run: bool = options["dry_run"]

        existing_count: int = Ward.objects.count()

        # Abort early when wards already exist and the caller did not ask
        # for a replace.  Dry-run bypasses this guard so you can preview
        # the command output even when rows are present.
        if existing_count > 0 and not replace and not dry_run:
            raise CommandError(
                f"{existing_count} ward(s) already exist in the database.\n"
                "Use --replace to delete all existing wards and re-seed with "
                "the full 101-ward registry."
            )

        self.stdout.write(f"Registry : {len(TVMC_WARDS)} wards")
        self.stdout.write(f"Existing : {existing_count} ward(s) in DB")

        # --- Phase 1: validate all 101 wards (no DB writes) -----------------
        placeholder = GEOSGeometry(_PLACEHOLDER_WKT, srid=4326)
        validated: list[Ward] = []

        for code, name in TVMC_WARDS:
            ward = Ward(
                code=code,
                name=name,
                translated_names={},
                boundary=GEOSGeometry(_PLACEHOLDER_WKT, srid=4326),
                is_active=True,
                officer_assignment_metadata={},
                landmark_mapping_metadata={},
            )
            try:
                ward.full_clean()
            except ValidationError as exc:
                raise CommandError(
                    f"Ward '{code}' ({name}) failed full_clean(): {exc}\n"
                    "The ward registry constant contains invalid data — "
                    "fix TVMC_WARDS before re-running."
                ) from exc
            validated.append(ward)

        self.stdout.write(f"Validated: {len(validated)} wards passed full_clean()")

        # Unused variable — kept for clarity in the dry-run output path.
        del placeholder

        # --- Dry run: report and exit ----------------------------------------
        if dry_run:
            self.stdout.write(
                self.style.WARNING("\n--- DRY RUN — no database changes written ---")
            )
            if replace:
                self.stdout.write(f"  Would delete : {existing_count} existing ward(s)")
            self.stdout.write(
                self.style.SUCCESS(f"  Would insert : {len(validated)} wards")
            )
            return

        # --- Phase 2: single transactional write -----------------------------
        with transaction.atomic():
            deleted = 0
            if replace:
                deleted, _ = Ward.objects.all().delete()
                if deleted:
                    self.stdout.write(f"Deleted  : {deleted} existing ward(s)")

            for ward in validated:
                ward.save()

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone. Inserted {len(validated)} wards into the database."
            )
        )

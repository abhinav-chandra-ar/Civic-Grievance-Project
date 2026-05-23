"""Management command: import_tvmc_wards

Import Thiruvananthapuram ward boundaries from a GeoJSON file.

Designed to accept any of these sources (once available):
  - OpenSDI KSUDP WFS output (after ogr2ogr reprojection from EPSG:32643)
  - IKM/KSMART 2025 delimitation GeoPackage export
  - Any WGS84 GeoJSON with polygon ward boundaries

CRS requirement
---------------
Ward.boundary is PolygonField(srid=4326).  The input file MUST use
EPSG:4326 (WGS84).  If your source uses EPSG:32643 (UTM Zone 43N,
common for Kerala government datasets), reproject it first:

    ogr2ogr -f GeoJSON -t_srs EPSG:4326 wards_wgs84.geojson <input>

MultiPolygon policy
-------------------
MultiPolygon features are NEVER silently converted.  Administrative
boundaries must not be mutated without human review.  MultiPolygon
features are logged as warnings and skipped (--skip-invalid) or abort
the entire import (strict mode).

Two-phase execution
-------------------
Phase 1  — validate every feature, build prepared Ward objects, run
           full_clean().  No database writes.
Phase 2  — write all validated wards in a single transaction.atomic().
           Dry-run stops after Phase 1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.wards.models import Ward
from apps.wards.services import create_ward, update_ward

# ---------------------------------------------------------------------------
# CRS validation
# ---------------------------------------------------------------------------

_WGS84_CRS_NAMES: frozenset[str] = frozenset({
    "urn:ogc:def:crs:ogc:1.3:crs84",
    "urn:ogc:def:crs:epsg::4326",
    "epsg:4326",
    "wgs84",
    "wgs 84",
    "geographic crs wgs 84",
})

# ---------------------------------------------------------------------------
# Property field name candidates (tried in order of preference)
# ---------------------------------------------------------------------------

_NAME_CANDIDATES: tuple[str, ...] = (
    "name", "WARD_NAME", "ward_name", "ward_name_en", "wardname", "WardName",
)
_REF_CANDIDATES: tuple[str, ...] = (
    "ref", "WARD_NO", "ward_no", "ward", "ward_number", "WARD_NUMBER", "wardno",
)
_NAME_ML_CANDIDATES: tuple[str, ...] = (
    "name:ml", "name_ml", "WARD_NAME_ML", "ward_name_ml", "nameml", "name_malayalam",
)

# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Internal data holder
# ---------------------------------------------------------------------------


class _PreparedWard(NamedTuple):
    code: str
    name: str
    boundary: object          # GEOSGeometry Polygon, srid=4326
    translated_names: dict
    existing: Ward | None     # None → INSERT; Ward instance → UPDATE (--replace)


# ---------------------------------------------------------------------------
# Public pure helpers (importable for unit tests)
# ---------------------------------------------------------------------------


def validate_crs(geojson: dict) -> None:
    """Raise CommandError if the GeoJSON declares a non-WGS84 CRS.

    A missing 'crs' key is accepted: RFC 7946 mandates WGS84 as the
    implicit default and recommends omitting the 'crs' member.  Older
    tools (GeoNode, GDAL) still emit it, so we tolerate it when the
    value resolves to WGS84.
    """
    crs = geojson.get("crs")
    if crs is None:
        return
    name = (crs.get("properties", {}).get("name") or "").strip().lower()
    if not name or name in _WGS84_CRS_NAMES:
        return
    raise CommandError(
        f"Unsupported CRS '{name}'.\n"
        "Ward.boundary requires EPSG:4326 (WGS84).  Reproject the file first:\n"
        "  ogr2ogr -f GeoJSON -t_srs EPSG:4326 reprojected.geojson <input-file>\n"
        "Then re-run this command with the reprojected file."
    )


def extract_property(properties: dict, candidates: tuple[str, ...]) -> str | None:
    """Return the first non-blank string value matching any candidate key."""
    for key in candidates:
        value = properties.get(key)
        if value is not None:
            value = str(value).strip()
            if value:
                return value
    return None


def normalise_code(ref: str | None, name: str) -> str:
    """Return a WARD_CODE_VALIDATOR-safe lowercase code.

    Primary path:  ref is a valid integer in 1..999 → 'tvm_NNN'
                   (zero-padded to three digits, e.g. tvm_001).
    Fallback path: slugify the ward name            → 'tvm_<slug>'.

    Raises ValueError if neither path yields a code matching
    ``^[a-z][a-z0-9_]*$``.
    """
    if ref is not None:
        try:
            num = int(ref)
            if 1 <= num <= 999:
                candidate = f"tvm_{num:03d}"
                if _CODE_RE.match(candidate):
                    return candidate
        except ValueError:
            pass

    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    candidate = f"tvm_{slug}"
    if slug and _CODE_RE.match(candidate):
        return candidate

    raise ValueError(
        f"Cannot generate a valid ward code from ref={ref!r}, name={name!r}."
    )


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------


class Command(BaseCommand):
    help = (
        "Import Thiruvananthapuram ward boundaries from a GeoJSON file.\n\n"
        "Accepts output from OpenSDI KSUDP WFS (after ogr2ogr reprojection),\n"
        "IKM/KSMART 2025 GeoPackage exports, or any WGS84 polygon GeoJSON.\n\n"
        "MultiPolygon features are NEVER silently converted — they are skipped\n"
        "(--skip-invalid) or abort the import (strict mode)."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--file",
            required=True,
            metavar="PATH",
            help="Path to the input GeoJSON file.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help=(
                "Validate all features and print what would be imported "
                "without writing to the database."
            ),
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            default=False,
            help=(
                "Update wards that already exist in the database (matched by "
                "code) instead of treating them as conflicts."
            ),
        )
        parser.add_argument(
            "--skip-invalid",
            action="store_true",
            default=False,
            help=(
                "Skip features that cannot be imported (MultiPolygon geometry, "
                "missing name, validation errors) and continue processing the "
                "remaining features.  Without this flag any invalid feature "
                "aborts the entire import."
            ),
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle(self, *args, **options) -> None:
        path = Path(options["file"])
        dry_run: bool = options["dry_run"]
        replace: bool = options["replace"]
        skip_invalid: bool = options["skip_invalid"]

        # --- Load and parse --------------------------------------------------
        if not path.exists():
            raise CommandError(f"File not found: {path}")
        if not path.is_file():
            raise CommandError(f"Not a regular file: {path}")

        try:
            geojson = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CommandError(f"Cannot read GeoJSON file: {exc}") from exc

        if geojson.get("type") != "FeatureCollection":
            raise CommandError(
                f"Expected a GeoJSON FeatureCollection, "
                f"got type={geojson.get('type')!r}."
            )

        # --- CRS report and check -------------------------------------------
        crs_node = geojson.get("crs")
        if crs_node is None:
            self.stdout.write(
                "CRS    : not declared — RFC 7946 default (EPSG:4326) assumed"
            )
        else:
            crs_label = crs_node.get("properties", {}).get("name", "(unnamed)")
            self.stdout.write(f"CRS    : {crs_label}")

        validate_crs(geojson)  # raises CommandError for non-WGS84

        features: list[dict] = geojson.get("features") or []
        self.stdout.write(f"File   : {path}")
        self.stdout.write(f"Features in file: {len(features)}")

        if not features:
            self.stdout.write(self.style.WARNING("No features found — nothing to import."))
            return

        # --- Phase 1: validate and prepare (no DB writes) -------------------
        prepared: list[_PreparedWard] = []
        counts: dict[str, int] = {
            "multipolygon": 0,
            "unsupported_geom": 0,
            "no_name": 0,
            "bad_code": 0,
            "code_conflict": 0,
            "validation_error": 0,
            "duplicate_in_batch": 0,
        }
        seen_codes: set[str] = set()

        for idx, feature in enumerate(features, 1):
            result = self._prepare_feature(
                feature=feature,
                idx=idx,
                replace=replace,
                skip_invalid=skip_invalid,
                seen_codes=seen_codes,
                counts=counts,
            )
            if result is not None:
                prepared.append(result)
                seen_codes.add(result.code)

        # --- Totals ----------------------------------------------------------
        skip_total = sum(counts.values())
        would_insert = sum(1 for p in prepared if p.existing is None)
        would_update = sum(1 for p in prepared if p.existing is not None)

        if dry_run:
            self.stdout.write(self.style.WARNING("\n--- DRY RUN — no database changes written ---"))
            self._print_summary(
                would_insert, would_update, counts, skip_total, dry_run=True,
            )
            return

        # --- Phase 2: single transactional write ----------------------------
        inserted = updated = 0
        with transaction.atomic():
            for pw in prepared:
                if pw.existing is None:
                    create_ward(
                        code=pw.code,
                        name=pw.name,
                        boundary=pw.boundary,
                        translated_names=pw.translated_names,
                    )
                    inserted += 1
                else:
                    update_ward(
                        ward=pw.existing,
                        values={
                            "name": pw.name,
                            "boundary": pw.boundary,
                            "translated_names": pw.translated_names,
                        },
                    )
                    updated += 1

        self._print_summary(inserted, updated, counts, skip_total, dry_run=False)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare_feature(
        self,
        *,
        feature: dict,
        idx: int,
        replace: bool,
        skip_invalid: bool,
        seen_codes: set[str],
        counts: dict[str, int],
    ) -> _PreparedWard | None:
        """Validate and normalise one GeoJSON feature.

        Returns a _PreparedWard on success, None when the feature is
        skipped (--skip-invalid), or raises CommandError in strict mode.
        """
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        geom_type = geometry.get("type", "")

        # ---- geometry type -------------------------------------------------
        if geom_type == "MultiPolygon":
            return self._skip(
                f"[{idx}] MultiPolygon geometry — administrative boundaries "
                "must not be silently converted to Polygon.  Inspect the "
                "feature and supply a single Polygon, or split it manually.",
                skip_invalid=skip_invalid,
                counts=counts,
                key="multipolygon",
            )

        if geom_type != "Polygon":
            return self._skip(
                f"[{idx}] Unsupported geometry type '{geom_type}' "
                "(only Polygon is accepted).",
                skip_invalid=skip_invalid,
                counts=counts,
                key="unsupported_geom",
            )

        # ---- extract name --------------------------------------------------
        name = extract_property(properties, _NAME_CANDIDATES)
        if not name:
            return self._skip(
                f"[{idx}] Could not extract ward name.  "
                f"Tried: {_NAME_CANDIDATES}.  "
                f"Available properties: {sorted(properties)[:12]}",
                skip_invalid=skip_invalid,
                counts=counts,
                key="no_name",
            )

        ref = extract_property(properties, _REF_CANDIDATES)
        name_ml = extract_property(properties, _NAME_ML_CANDIDATES)

        # ---- generate code -------------------------------------------------
        try:
            code = normalise_code(ref, name)
        except ValueError as exc:
            return self._skip(
                f"[{idx}] {exc}",
                skip_invalid=skip_invalid,
                counts=counts,
                key="bad_code",
            )

        # ---- duplicate within this batch -----------------------------------
        if code in seen_codes:
            return self._skip(
                f"[{idx}] Code '{code}' appears more than once in this "
                "import batch — possible duplicate ward name or ref.",
                skip_invalid=skip_invalid,
                counts=counts,
                key="duplicate_in_batch",
            )

        # ---- existing DB row -----------------------------------------------
        existing: Ward | None = None
        try:
            existing = Ward.objects.get(code=code)
            if not replace:
                return self._skip(
                    f"[{idx}] Ward '{code}' ({name}) already exists in the "
                    "database.  Use --replace to update it.",
                    skip_invalid=skip_invalid,
                    counts=counts,
                    key="code_conflict",
                )
        except Ward.DoesNotExist:
            existing = None

        # ---- parse geometry ------------------------------------------------
        try:
            boundary = GEOSGeometry(json.dumps(geometry), srid=4326)
        except Exception as exc:
            return self._skip(
                f"[{idx}] Cannot parse geometry for '{name}': {exc}",
                skip_invalid=skip_invalid,
                counts=counts,
                key="unsupported_geom",
            )

        # ---- full_clean via a transient Ward instance ----------------------
        translated_names = {"ml": name_ml} if name_ml else {}
        probe = Ward(
            code=code,
            name=name,
            boundary=boundary,
            translated_names=translated_names,
        )
        try:
            probe.full_clean()
        except ValidationError as exc:
            detail = "; ".join(
                f"{field}: {', '.join(str(e) for e in errs)}"
                for field, errs in exc.message_dict.items()
            )
            return self._skip(
                f"[{idx}] Ward.full_clean() failed for '{code}' ({name}): {detail}",
                skip_invalid=skip_invalid,
                counts=counts,
                key="validation_error",
            )

        action = "update" if existing else "insert"
        self.stdout.write(f"[{idx:>3}] OK {action:6}  {code}  '{name}'")
        return _PreparedWard(
            code=code,
            name=name,
            boundary=boundary,
            translated_names=translated_names,
            existing=existing,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _skip(
        self,
        message: str,
        *,
        skip_invalid: bool,
        counts: dict[str, int],
        key: str,
    ) -> None:
        """Log a skip, increment the counter, continue or abort."""
        if skip_invalid:
            self.stderr.write(self.style.WARNING(f"SKIP  {message}"))
            counts[key] += 1
            return None
        raise CommandError(
            f"{message}\n"
            "Add --skip-invalid to skip bad features and continue."
        )

    def _print_summary(
        self,
        inserted: int,
        updated: int,
        counts: dict[str, int],
        skip_total: int,
        *,
        dry_run: bool,
    ) -> None:
        prefix = "Would " if dry_run else ""
        self.stdout.write("\n--- Summary ---")
        self.stdout.write(self.style.SUCCESS(f"  {prefix}Insert : {inserted}"))
        if updated:
            self.stdout.write(self.style.SUCCESS(f"  {prefix}Update : {updated}"))
        if skip_total:
            self.stdout.write(f"  Skipped        : {skip_total}")
            _labels = {
                "multipolygon":      "    MultiPolygon      ",
                "unsupported_geom":  "    Unsupported geom  ",
                "no_name":           "    No name           ",
                "bad_code":          "    Bad code          ",
                "code_conflict":     "    Code conflict     ",
                "validation_error":  "    Validation error  ",
                "duplicate_in_batch": "    Duplicate in batch",
            }
            for key, label in _labels.items():
                if counts.get(key):
                    self.stdout.write(f"{label}: {counts[key]}")

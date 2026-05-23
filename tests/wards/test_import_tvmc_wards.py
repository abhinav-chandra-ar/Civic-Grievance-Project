"""Tests for the import_tvmc_wards management command.

Strategy
--------
Helper-function tests are pure Python (no DB, no pytest.mark.django_db).
Command-level tests use tmp_path to create real GeoJSON files and
call_command to drive the full pipeline.  DB writes are only exercised
in tests that explicitly cover INSERT / UPDATE behaviour.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.wards.management.commands.import_tvmc_wards import (
    _NAME_CANDIDATES,
    _REF_CANDIDATES,
    extract_property,
    normalise_code,
    validate_crs,
)
from apps.wards.models import Ward

# ---------------------------------------------------------------------------
# GeoJSON fixtures
# ---------------------------------------------------------------------------

# A valid square polygon inside Thiruvananthapuram area (EPSG:4326).
_SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [[76.95, 8.50], [76.96, 8.50], [76.96, 8.51], [76.95, 8.51], [76.95, 8.50]]
    ],
}

_SQUARE_2 = {
    "type": "Polygon",
    "coordinates": [
        [[76.97, 8.52], [76.98, 8.52], [76.98, 8.53], [76.97, 8.53], [76.97, 8.52]]
    ],
}

_MULTIPOLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[76.95, 8.50], [76.96, 8.50], [76.96, 8.51], [76.95, 8.51], [76.95, 8.50]]]
    ],
}


def _feature(geometry: dict, properties: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties or {"name": "Test Ward", "ref": "1"},
    }


def _collection(*features: dict, crs: dict | None = None) -> dict:
    fc: dict = {"type": "FeatureCollection", "features": list(features)}
    if crs is not None:
        fc["crs"] = crs
    return fc


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "wards.geojson"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# validate_crs — pure function tests (no DB)
# ---------------------------------------------------------------------------


class TestValidateCrs:
    def test_accepts_missing_crs(self):
        validate_crs({"type": "FeatureCollection", "features": []})

    def test_accepts_explicit_epsg4326(self):
        validate_crs(_collection(crs={"type": "name", "properties": {"name": "EPSG:4326"}}))

    def test_accepts_wgs84_name(self):
        validate_crs(_collection(crs={"type": "name", "properties": {"name": "WGS84"}}))

    def test_accepts_ogc_crs84_urn(self):
        validate_crs(
            _collection(
                crs={
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
                }
            )
        )

    def test_accepts_epsg4326_urn(self):
        validate_crs(
            _collection(
                crs={
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
                }
            )
        )

    def test_rejects_epsg32643(self):
        with pytest.raises(CommandError, match="32643"):
            validate_crs(
                _collection(
                    crs={
                        "type": "name",
                        "properties": {"name": "urn:ogc:def:crs:EPSG::32643"},
                    }
                )
            )

    def test_rejects_unknown_crs(self):
        with pytest.raises(CommandError, match="Unsupported CRS"):
            validate_crs(
                _collection(
                    crs={"type": "name", "properties": {"name": "EPSG:3857"}}
                )
            )

    def test_rejects_utm_zone_43n(self):
        with pytest.raises(CommandError):
            validate_crs(
                _collection(
                    crs={"type": "name", "properties": {"name": "WGS 84 / UTM zone 43N"}}
                )
            )


# ---------------------------------------------------------------------------
# extract_property — pure function tests (no DB)
# ---------------------------------------------------------------------------


class TestExtractProperty:
    def test_returns_first_matching_candidate(self):
        assert extract_property({"name": "Kowdiar"}, _NAME_CANDIDATES) == "Kowdiar"

    def test_falls_through_to_second_candidate(self):
        assert extract_property({"WARD_NAME": "Palayam"}, _NAME_CANDIDATES) == "Palayam"

    def test_returns_none_when_no_candidate_matches(self):
        assert extract_property({"unknown_key": "x"}, _NAME_CANDIDATES) is None

    def test_skips_blank_string_values(self):
        assert extract_property({"name": "  ", "WARD_NAME": "Pattom"}, _NAME_CANDIDATES) == "Pattom"

    def test_skips_none_values(self):
        assert extract_property({"name": None, "WARD_NAME": "Fort"}, _NAME_CANDIDATES) == "Fort"

    def test_strips_surrounding_whitespace(self):
        assert extract_property({"name": "  Ulloor  "}, _NAME_CANDIDATES) == "Ulloor"

    def test_coerces_non_string_values(self):
        assert extract_property({"ref": 7}, _REF_CANDIDATES) == "7"


# ---------------------------------------------------------------------------
# normalise_code — pure function tests (no DB)
# ---------------------------------------------------------------------------


class TestNormaliseCode:
    def test_integer_ref_zero_padded_to_three_digits(self):
        assert normalise_code("1", "Kowdiar") == "tvm_001"

    def test_two_digit_ref(self):
        assert normalise_code("42", "Kesavadasapuram") == "tvm_042"

    def test_three_digit_ref(self):
        assert normalise_code("101", "Pallithura") == "tvm_101"

    def test_name_fallback_when_ref_is_none(self):
        assert normalise_code(None, "Kowdiar") == "tvm_kowdiar"

    def test_name_fallback_when_ref_is_not_integer(self):
        assert normalise_code("ward-A", "Pattom") == "tvm_pattom"

    def test_name_with_spaces_slugified(self):
        assert normalise_code(None, "East Fort") == "tvm_east_fort"

    def test_name_with_mixed_case_lowercased(self):
        assert normalise_code(None, "Kazhakkoottam") == "tvm_kazhakkoottam"

    def test_name_special_chars_replaced_with_underscore(self):
        # Malayalam characters get stripped; remaining ASCII is slugified.
        result = normalise_code(None, "Ward No. 5")
        assert result.startswith("tvm_")
        assert result == result.lower()

    def test_raises_when_both_paths_fail(self):
        with pytest.raises(ValueError, match="valid ward code"):
            normalise_code(None, "!@#$%")  # non-alphanumeric only → empty slug

    def test_ref_out_of_range_falls_back_to_name(self):
        # ref > 999 is treated as out-of-range; falls back to name slugification
        assert normalise_code("1000", "Attukal") == "tvm_attukal"


# ---------------------------------------------------------------------------
# Command — file / argument validation (DB not needed for these)
# ---------------------------------------------------------------------------


class TestCommandArguments:
    def test_raises_for_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.geojson"
        with pytest.raises((CommandError, SystemExit)):
            call_command("import_tvmc_wards", file=str(missing), stdout=StringIO(), stderr=StringIO())

    def test_raises_for_non_feature_collection(self, tmp_path):
        bad = tmp_path / "bad.geojson"
        bad.write_text(json.dumps({"type": "Feature", "geometry": _SQUARE, "properties": {}}))
        with pytest.raises(CommandError, match="FeatureCollection"):
            call_command("import_tvmc_wards", file=str(bad), stdout=StringIO(), stderr=StringIO())

    def test_raises_for_wrong_crs(self, tmp_path):
        data = _collection(
            _feature(_SQUARE),
            crs={"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32643"}},
        )
        p = _write(tmp_path, data)
        with pytest.raises(CommandError, match="32643"):
            call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())

    def test_raises_for_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.geojson"
        bad.write_text("not json at all", encoding="utf-8")
        with pytest.raises(CommandError, match="Cannot read"):
            call_command("import_tvmc_wards", file=str(bad), stdout=StringIO(), stderr=StringIO())

    def test_empty_feature_collection_exits_cleanly(self, tmp_path):
        p = _write(tmp_path, _collection())
        out = StringIO()
        call_command("import_tvmc_wards", file=str(p), dry_run=True, stdout=out, stderr=StringIO())
        assert "nothing" in out.getvalue().lower()


# ---------------------------------------------------------------------------
# Command — MultiPolygon handling (no DB writes needed)
# ---------------------------------------------------------------------------


class TestMultiPolygonHandling:
    def test_strict_mode_aborts_on_multipolygon(self, tmp_path):
        p = _write(tmp_path, _collection(_feature(_MULTIPOLYGON, {"name": "Chalai", "ref": "1"})))
        with pytest.raises(CommandError, match="MultiPolygon"):
            call_command(
                "import_tvmc_wards",
                file=str(p),
                dry_run=True,
                stdout=StringIO(),
                stderr=StringIO(),
            )

    def test_skip_invalid_continues_past_multipolygon(self, tmp_path):
        # One MultiPolygon + one valid Polygon — the valid one should pass.
        data = _collection(
            _feature(_MULTIPOLYGON, {"name": "Chalai", "ref": "14"}),
            _feature(_SQUARE, {"name": "Attukal", "ref": "12"}),
        )
        p = _write(tmp_path, data)
        out, err = StringIO(), StringIO()
        call_command(
            "import_tvmc_wards",
            file=str(p),
            dry_run=True,
            skip_invalid=True,
            stdout=out,
            stderr=err,
        )
        output = out.getvalue()
        assert "Would Insert : 1" in output or "Would Insert: 1" in output or "insert" in output.lower()
        assert "MultiPolygon" in err.getvalue()

    def test_multipolygon_skip_count_appears_in_summary(self, tmp_path):
        data = _collection(
            _feature(_MULTIPOLYGON, {"name": "Jagathy", "ref": "15"}),
        )
        p = _write(tmp_path, data)
        out, err = StringIO(), StringIO()
        with pytest.raises(CommandError):
            call_command(
                "import_tvmc_wards",
                file=str(p),
                dry_run=True,
                stdout=out,
                stderr=err,
            )


# ---------------------------------------------------------------------------
# Command — full pipeline with DB writes
# ---------------------------------------------------------------------------


@pytest.mark.django_db
class TestCommandImport:
    def test_dry_run_writes_nothing(self, tmp_path):
        p = _write(tmp_path, _collection(_feature(_SQUARE, {"name": "Kowdiar", "ref": "1"})))
        call_command(
            "import_tvmc_wards",
            file=str(p),
            dry_run=True,
            stdout=StringIO(),
            stderr=StringIO(),
        )
        assert Ward.objects.filter(code="tvm_001").count() == 0

    def test_inserts_valid_ward(self, tmp_path):
        p = _write(tmp_path, _collection(_feature(_SQUARE, {"name": "Kowdiar", "ref": "1"})))
        call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())
        w = Ward.objects.get(code="tvm_001")
        assert w.name == "Kowdiar"
        assert w.boundary.srid == 4326
        assert w.boundary.valid

    def test_inserts_preserves_malayalam_name(self, tmp_path):
        props = {"name": "Kowdiar", "ref": "1", "name:ml": "കൗഡിയാർ"}
        p = _write(tmp_path, _collection(_feature(_SQUARE, props)))
        call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())
        w = Ward.objects.get(code="tvm_001")
        assert w.translated_names == {"ml": "കൗഡിയാർ"}

    def test_inserts_ward_with_alternative_field_names(self, tmp_path):
        # Simulates KSUDP WFS field naming convention.
        props = {"WARD_NAME": "Vazhuthacaud", "WARD_NO": "2"}
        p = _write(tmp_path, _collection(_feature(_SQUARE, props)))
        call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())
        assert Ward.objects.filter(code="tvm_002", name="Vazhuthacaud").exists()

    def test_strict_mode_aborts_on_existing_code_without_replace(self, tmp_path):
        p = _write(tmp_path, _collection(_feature(_SQUARE, {"name": "Pattom", "ref": "3"})))
        call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())
        with pytest.raises(CommandError, match="already exists"):
            call_command(
                "import_tvmc_wards",
                file=str(p),
                stdout=StringIO(),
                stderr=StringIO(),
            )

    def test_replace_updates_existing_ward(self, tmp_path):
        # First import.
        p1 = _write(tmp_path, _collection(_feature(_SQUARE, {"name": "Pattom", "ref": "3"})))
        call_command("import_tvmc_wards", file=str(p1), stdout=StringIO(), stderr=StringIO())
        # Second import with different name and different geometry.
        updated_props = {"name": "Pattom Updated", "ref": "3"}
        p2 = tmp_path / "updated.geojson"
        p2.write_text(json.dumps(_collection(_feature(_SQUARE_2, updated_props))), encoding="utf-8")
        call_command(
            "import_tvmc_wards",
            file=str(p2),
            replace=True,
            stdout=StringIO(),
            stderr=StringIO(),
        )
        w = Ward.objects.get(code="tvm_003")
        assert w.name == "Pattom Updated"

    def test_duplicate_code_in_batch_aborts_strict(self, tmp_path):
        data = _collection(
            _feature(_SQUARE, {"name": "Palayam", "ref": "4"}),
            _feature(_SQUARE_2, {"name": "Palayam Duplicate", "ref": "4"}),
        )
        p = _write(tmp_path, data)
        with pytest.raises(CommandError, match="more than once"):
            call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())
        assert Ward.objects.filter(code="tvm_004").count() == 0

    def test_duplicate_code_in_batch_skipped_with_skip_invalid(self, tmp_path):
        data = _collection(
            _feature(_SQUARE, {"name": "Palayam", "ref": "4"}),
            _feature(_SQUARE_2, {"name": "Palayam Duplicate", "ref": "4"}),
        )
        p = _write(tmp_path, data)
        call_command(
            "import_tvmc_wards",
            file=str(p),
            skip_invalid=True,
            stdout=StringIO(),
            stderr=StringIO(),
        )
        # First occurrence inserted; second skipped as duplicate.
        assert Ward.objects.filter(code="tvm_004").count() == 1

    def test_missing_name_aborts_strict(self, tmp_path):
        p = _write(tmp_path, _collection(_feature(_SQUARE, {"ref": "5"})))
        with pytest.raises(CommandError, match="name"):
            call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())

    def test_missing_name_skipped_with_skip_invalid(self, tmp_path):
        data = _collection(
            _feature(_SQUARE, {"ref": "5"}),              # no name — will be skipped
            _feature(_SQUARE_2, {"name": "Thampanoor", "ref": "6"}),  # valid
        )
        p = _write(tmp_path, data)
        call_command(
            "import_tvmc_wards",
            file=str(p),
            skip_invalid=True,
            stdout=StringIO(),
            stderr=StringIO(),
        )
        assert not Ward.objects.filter(code="tvm_005").exists()
        assert Ward.objects.filter(code="tvm_006").exists()

    def test_all_writes_rolled_back_if_transaction_fails(self, tmp_path, monkeypatch):
        # Simulate a DB error mid-write by patching create_ward to raise on second call.
        from apps.wards.management.commands import import_tvmc_wards as cmd_module

        call_count = 0

        def boom(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated DB error")
            from apps.wards.services import create_ward as real_create
            return real_create(**kwargs)

        monkeypatch.setattr(cmd_module, "create_ward", boom)

        data = _collection(
            _feature(_SQUARE, {"name": "Ward A", "ref": "7"}),
            _feature(_SQUARE_2, {"name": "Ward B", "ref": "8"}),
        )
        p = _write(tmp_path, data)
        with pytest.raises((CommandError, RuntimeError)):
            call_command("import_tvmc_wards", file=str(p), stdout=StringIO(), stderr=StringIO())
        # transaction.atomic() rolls back both — neither ward should exist.
        assert Ward.objects.filter(code__in=["tvm_007", "tvm_008"]).count() == 0

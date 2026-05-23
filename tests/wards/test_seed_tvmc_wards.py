"""Tests for the seed_tvmc_wards management command.

Structure
---------
Unit tests (no DB)
  - registry integrity: count, uniqueness, code format, name presence,
    boundary wards, known 2025 new wards

DB tests (pytest.mark.django_db)
  - happy path: 101 rows inserted
  - all inserted wards pass full_clean()
  - all inserted wards are active
  - codes and names match the registry constant exactly
  - aborts with CommandError when wards exist and --replace is omitted
  - --replace deletes all old rows and inserts fresh 101
  - --dry-run writes nothing to the DB
  - --dry-run does not raise even when wards already exist
"""
from __future__ import annotations

import re
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.wards.management.commands.seed_tvmc_wards import TVMC_WARDS
from apps.wards.models import Ward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_TVM_NNN_PATTERN = re.compile(r"^tvm_\d{3}$")


def _seed(**kwargs) -> str:
    """Run seed_tvmc_wards and return captured stdout."""
    out = StringIO()
    call_command("seed_tvmc_wards", stdout=out, **kwargs)
    return out.getvalue()


# ===========================================================================
# Unit tests — pure registry checks, no database required
# ===========================================================================


def test_registry_has_exactly_101_entries():
    assert len(TVMC_WARDS) == 101


def test_all_codes_are_unique():
    codes = [code for code, _ in TVMC_WARDS]
    assert len(codes) == len(set(codes)), "Duplicate codes found in TVMC_WARDS"


def test_all_codes_satisfy_ward_code_validator():
    for code, name in TVMC_WARDS:
        assert _CODE_PATTERN.match(code), (
            f"Code '{code}' (ward '{name}') fails WARD_CODE_VALIDATOR regex"
        )


def test_all_codes_follow_tvm_nnn_format():
    for code, name in TVMC_WARDS:
        assert _TVM_NNN_PATTERN.match(code), (
            f"Code '{code}' (ward '{name}') is not in tvm_NNN format"
        )


def test_all_ward_names_are_non_empty_strings():
    for code, name in TVMC_WARDS:
        assert isinstance(name, str) and name.strip(), (
            f"Empty or non-string name for code '{code}'"
        )


def test_registry_starts_with_kazhakkoottam():
    assert TVMC_WARDS[0] == ("tvm_001", "Kazhakkoottam")


def test_registry_ends_with_pallithura():
    assert TVMC_WARDS[-1] == ("tvm_101", "Pallithura")


def test_ward_numbers_are_sequential():
    """Each tvm_NNN code must be exactly one more than its predecessor."""
    for i, (code, _) in enumerate(TVMC_WARDS, start=1):
        expected = f"tvm_{i:03d}"
        assert code == expected, (
            f"Position {i}: expected code '{expected}', found '{code}'"
        )


def test_known_2025_new_wards_are_present():
    """Six wards added in the 2025 Kerala SEC delimitation must be in registry."""
    lookup = {code: name for code, name in TVMC_WARDS}
    assert lookup.get("tvm_002") == "Sainika School"
    assert lookup.get("tvm_007") == "Chenkottukonam"
    assert lookup.get("tvm_036") == "Gowreeshapattom"
    assert lookup.get("tvm_065") == "Port"
    assert lookup.get("tvm_096") == "Alathara"
    assert lookup.get("tvm_097") == "Kuzhivila"


def test_known_legacy_wards_are_present():
    """Legacy ward names that existed before 2025 must survive in the registry."""
    lookup = {name for _, name in TVMC_WARDS}
    legacy = {
        "Kowdiar", "Vazhuthacaud", "Pattom", "Palayam",
        "Fort", "Thampanoor", "Pettah", "Sreekariyam",
        "Kazhakkoottam", "Ulloor", "Kesavadasapuram",
        "Attukal", "Manacaud", "Chalai", "Jagathy",
    }
    missing = legacy - lookup
    assert not missing, f"Legacy wards missing from registry: {missing}"


# ===========================================================================
# DB tests
# ===========================================================================


@pytest.mark.django_db
def test_seed_inserts_101_wards():
    _seed()
    assert Ward.objects.count() == 101


@pytest.mark.django_db
def test_seed_all_wards_pass_full_clean_after_insert():
    _seed()
    for ward in Ward.objects.all():
        ward.full_clean()  # must not raise


@pytest.mark.django_db
def test_seed_all_inserted_wards_are_active():
    _seed()
    assert not Ward.objects.filter(is_active=False).exists()


@pytest.mark.django_db
def test_seed_codes_match_registry():
    _seed()
    db_codes = set(Ward.objects.values_list("code", flat=True))
    expected_codes = {code for code, _ in TVMC_WARDS}
    assert db_codes == expected_codes


@pytest.mark.django_db
def test_seed_names_match_registry():
    _seed()
    db_map = {w.code: w.name for w in Ward.objects.all()}
    for code, name in TVMC_WARDS:
        assert db_map[code] == name, (
            f"{code}: expected '{name}', got '{db_map.get(code)}'"
        )


@pytest.mark.django_db
def test_seed_translated_names_defaults_to_empty_dict():
    _seed()
    assert not Ward.objects.exclude(translated_names={}).exists()


@pytest.mark.django_db
def test_seed_aborts_when_wards_exist_without_replace():
    _seed()
    assert Ward.objects.count() == 101

    with pytest.raises(CommandError, match="--replace"):
        _seed()

    # Row count must remain unchanged after aborted second run.
    assert Ward.objects.count() == 101


@pytest.mark.django_db
def test_seed_replace_deletes_old_rows_and_reinserts_101():
    _seed()
    first_ids = set(Ward.objects.values_list("id", flat=True))

    _seed(replace=True)

    second_ids = set(Ward.objects.values_list("id", flat=True))
    assert Ward.objects.count() == 101
    # All rows are brand-new — no old PK should survive.
    assert first_ids.isdisjoint(second_ids), (
        "Some old Ward PKs survived --replace; expected full delete-and-reinsert"
    )


@pytest.mark.django_db
def test_seed_replace_output_mentions_deleted_count():
    _seed()
    out = _seed(replace=True)
    assert "101" in out  # deleted count visible in output


@pytest.mark.django_db
def test_dry_run_writes_nothing_to_db():
    out = _seed(dry_run=True)
    assert Ward.objects.count() == 0
    assert "DRY RUN" in out


@pytest.mark.django_db
def test_dry_run_does_not_raise_when_wards_already_exist():
    _seed()
    assert Ward.objects.count() == 101

    # Must not raise CommandError even though wards exist and --replace is absent.
    out = _seed(dry_run=True)
    assert "DRY RUN" in out
    # DB unchanged.
    assert Ward.objects.count() == 101


@pytest.mark.django_db
def test_dry_run_reports_would_insert_101():
    out = _seed(dry_run=True)
    assert "101" in out

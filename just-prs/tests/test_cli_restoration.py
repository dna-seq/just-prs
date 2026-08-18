"""CLI ``--reference-restoration`` parser and result-cache key suffix."""

from pathlib import Path

import pytest
import typer

from just_prs.chip_coverage import Chip
from just_prs.cli import (
    _parse_restoration_scope,
    _restoration_cache_token,
    _result_cache_key,
)


def test_parse_restoration_scope_off_aliases() -> None:
    for value in ("off", "OFF", "false", "none"):
        assert _parse_restoration_scope(value) is False
        assert _restoration_cache_token(_parse_restoration_scope(value)) == "off"


def test_parse_restoration_scope_wgs_aliases() -> None:
    for value in ("wgs", "WGS", "true", "universe"):
        assert _parse_restoration_scope(value) is True
        assert _restoration_cache_token(_parse_restoration_scope(value)) == "wgs"


def test_parse_restoration_scope_chip() -> None:
    assert _parse_restoration_scope("gsa_v3") is Chip.GSA_V3
    assert _restoration_cache_token(Chip.GSA_V3) == "gsa_v3"


def test_parse_restoration_scope_invalid() -> None:
    with pytest.raises(typer.BadParameter, match="chip id"):
        _parse_restoration_scope("not-a-scope")


def test_result_cache_key_off_keeps_historical_key(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n")
    historical = _result_cache_key(vcf, "PGS000001", "GRCh38", "EUR")
    assert _result_cache_key(vcf, "PGS000001", "GRCh38", "EUR", "off") == historical
    assert "_restore=" not in historical


def test_result_cache_key_wgs_differs_from_off(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n")
    off_key = _result_cache_key(vcf, "PGS000001", "GRCh38", "EUR", "off")
    wgs_key = _result_cache_key(vcf, "PGS000001", "GRCh38", "EUR", "wgs")
    chip_key = _result_cache_key(vcf, "PGS000001", "GRCh38", "EUR", "gsa_v3")
    assert wgs_key != off_key
    assert chip_key != off_key
    assert chip_key != wgs_key
    assert wgs_key.endswith("_restore=wgs")
    assert chip_key.endswith("_restore=gsa_v3")

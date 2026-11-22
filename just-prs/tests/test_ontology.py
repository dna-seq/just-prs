from pathlib import Path

import polars as pl
import pytest

from just_prs.ontology import (
    enrich_with_requested_trait_aliases,
    enrich_with_trait_aliases,
    expand_trait_ids_from_alias_columns,
    normalize_trait_id,
)
from just_prs.prs_catalog import PRSCatalog


def test_normalize_trait_id_handles_curie_and_uri_forms() -> None:
    assert normalize_trait_id("MONDO:0005147") == "MONDO_0005147"
    assert normalize_trait_id("http://www.ebi.ac.uk/efo/EFO_0001359") == "EFO_0001359"


def test_enrich_with_trait_aliases_uses_cached_ols_aliases(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ols"
    cache_dir.mkdir()
    (cache_dir / "EFO_0001359.json").write_text(
        '{"efo_id": "EFO_0001359", "aliases": ["MONDO_0005147"]}'
    )
    df = pl.DataFrame({
        "efo_id": ["EFO_0001359"],
        "trait_label": ["type 1 diabetes"],
        "h2_liability": [0.13072],
    })

    enriched = enrich_with_trait_aliases(df, cache_dir=cache_dir, allow_network=False)

    assert set(enriched["efo_id"].to_list()) == {"EFO_0001359", "MONDO_0005147"}
    alias = enriched.filter(pl.col("efo_id") == "MONDO_0005147").row(0, named=True)
    assert alias["canonical_efo_id"] == "EFO_0001359"
    assert alias["mapping_source"] == "ols4_xref"


def test_enrich_with_requested_trait_aliases_resolves_mondo_via_icd10(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ols"
    cache_dir.mkdir()
    (cache_dir / "MONDO_0005147.json").write_text(
        '{"trait_id": "MONDO_0005147", "label": "type 1 diabetes mellitus", '
        '"aliases": ["Orphanet_243377"], "icd10_codes": ["E10"]}'
    )
    h2_df = pl.DataFrame({
        "efo_id": ["EFO_0001359"],
        "trait_label": ["E10 Insulin-dependent diabetes mellitus"],
        "h2_liability": [0.13072],
    })
    requested_df = pl.DataFrame({
        "trait_efo_id": ["MONDO_0005147"],
        "trait_efo": ["type 1 diabetes mellitus"],
    })
    mappings_df = pl.DataFrame({"ukb_code": ["E10"], "efo_id": ["EFO_0001359"]})

    enriched = enrich_with_requested_trait_aliases(
        h2_df,
        requested_traits_df=requested_df,
        efo_mappings_df=mappings_df,
        cache_dir=cache_dir,
        allow_network=False,
    )

    assert set(enriched["efo_id"].to_list()) == {"EFO_0001359", "MONDO_0005147"}
    alias = enriched.filter(pl.col("efo_id") == "MONDO_0005147").row(0, named=True)
    assert alias["canonical_efo_id"] == "EFO_0001359"
    assert alias["mapping_source"] == "ols4_icd10_to_efo"


def test_expand_trait_ids_from_alias_columns_recovers_canonical_efo() -> None:
    df = pl.DataFrame({
        "efo_id": ["EFO_0001359", "MONDO_0005147"],
        "canonical_efo_id": ["EFO_0001359", "EFO_0001359"],
        "mapped_from_id": [None, "EFO_0001359"],
        "mapping_source": ["direct", "ols4_xref"],
    })

    expanded = expand_trait_ids_from_alias_columns(["MONDO_0005147"], df)

    assert expanded == ["MONDO_0005147", "EFO_0001359"]


def _write_alias_risk_fixture(tmp_path: Path) -> Path:
    """Catalog cache where risk metadata needs a runtime MONDO -> EFO alias."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True)

    pl.DataFrame({
        "pgs_id": ["PGS_TEST"],
        "name": ["T1D test score"],
        "trait_reported": ["Type 1 diabetes"],
        "trait_efo": ["type 1 diabetes mellitus"],
        "trait_efo_id": ["MONDO_0005147"],
        "genome_build": ["GRCh38"],
        "n_variants": [10],
        "weight_type": ["NR"],
        "pgp_id": ["PGP_TEST"],
        "pmid": [None],
        "ftp_link": [None],
        "release_date": [None],
    }).write_parquet(metadata_dir / "scores.parquet")
    pl.DataFrame(schema={
        "pgs_id": pl.Utf8,
        "or_estimate": pl.Float64,
        "auroc_estimate": pl.Float64,
        "pgp_id": pl.Utf8,
    }).write_parquet(metadata_dir / "best_performance.parquet")
    pl.DataFrame(schema={"pgs_id": pl.Utf8}).write_parquet(metadata_dir / "performance.parquet")
    pl.DataFrame(schema={"pgp_id": pl.Utf8, "authors": pl.Utf8, "journal": pl.Utf8, "pmid": pl.Int64}).write_parquet(
        metadata_dir / "publications.parquet"
    )
    pl.DataFrame({
        "efo_id": ["EFO_0001359"],
        "trait_label": ["type I diabetes mellitus"],
        "prevalence": [0.005],
        "prevalence_lower": [None],
        "prevalence_upper": [None],
        "prevalence_type": ["lifetime"],
        "sex": [None],
        "ancestry": [None],
        "age_range": [None],
        "source": ["cdc"],
        "source_detail": ["CDC seed prevalence"],
        "xref_mondo": [None],
        "xref_icd10": [None],
        "confidence": ["high"],
        "canonical_efo_id": ["EFO_0001359"],
        "mapped_from_id": [None],
        "mapping_source": ["direct"],
    }).write_parquet(metadata_dir / "trait_prevalence.parquet")
    pl.DataFrame({
        "efo_id": ["EFO_0001359"],
        "trait_label": ["type I diabetes mellitus"],
        "h2_observed": [0.04],
        "h2_observed_se": [None],
        "h2_liability": [0.13072],
        "h2_liability_se": [None],
        "h2_z": [5.0],
        "ancestry": ["EUR"],
        "method": ["S-LDSC"],
        "source": ["pan_ukbb"],
        "source_detail": ["Pan-UKBB EUR S-LDSC"],
        "confidence": ["high"],
        "n_samples": [1000],
        "trait_type": ["categorical"],
        "canonical_efo_id": ["EFO_0001359"],
        "mapped_from_id": [None],
        "mapping_source": ["direct"],
    }).write_parquet(metadata_dir / "trait_heritability.parquet")
    raw_h2_dir = metadata_dir / "raw" / "heritability"
    raw_h2_dir.mkdir(parents=True)
    pl.DataFrame({
        "query": ["E10"],
        "efo_label": ["type I diabetes mellitus"],
        "efo_id": ["EFO_0001359"],
        "mapping_type": ["manual"],
        "ukb_code": ["E10"],
    }).write_parquet(raw_h2_dir / "efo_ukb_mappings.parquet")
    xref_dir = metadata_dir / "raw" / "ontology_xrefs"
    xref_dir.mkdir()
    (xref_dir / "MONDO_0005147.json").write_text(
        '{"trait_id": "MONDO_0005147", "label": "type 1 diabetes mellitus", '
        '"aliases": ["Orphanet_243377"], "icd10_codes": ["E10"]}'
    )

    return metadata_dir


def test_absolute_risk_bundle_uses_alias_mapped_heritability(tmp_path: Path) -> None:
    _write_alias_risk_fixture(tmp_path)

    catalog = PRSCatalog(cache_dir=tmp_path)
    bundle = catalog.absolute_risk_bundle("PGS_TEST", 1.0)

    assert bundle.heritability_status == "used"
    assert any(est.method == "h2_liability" for est in bundle.estimates)
    assert "EFO_0001359" in bundle.heritability_trait_ids
    assert bundle.best_estimate is not None
    assert bundle.best_estimate.method == "h2_liability"


def test_alias_enrichment_rewrites_shared_metadata_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alias rows are persisted via temp file + rename, never written in place.

    Several processes share one cache; an in-place ``write_parquet`` lets a concurrent
    reader hit a truncated file ("parquet: File out of specification").
    """
    import just_prs.prs_catalog as prs_catalog_module

    metadata_dir = _write_alias_risk_fixture(tmp_path)
    atomic_targets: list[Path] = []
    real_atomic = prs_catalog_module._atomic_write_parquet

    def spy_atomic(df: pl.DataFrame, dest: Path) -> None:
        atomic_targets.append(Path(dest))
        real_atomic(df, dest)

    shared = {
        (metadata_dir / "trait_heritability.parquet").resolve(),
        (metadata_dir / "trait_prevalence.parquet").resolve(),
    }
    real_write = pl.DataFrame.write_parquet

    def guarded_write(self: pl.DataFrame, file, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(file, (str, Path)) and Path(file).resolve() in shared:
            raise AssertionError(f"in-place write to shared cache file {file}")
        return real_write(self, file, *args, **kwargs)

    monkeypatch.setattr(prs_catalog_module, "_atomic_write_parquet", spy_atomic)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", guarded_write)

    bundle = PRSCatalog(cache_dir=tmp_path).absolute_risk_bundle("PGS_TEST", 1.0)

    assert bundle.heritability_status == "used"
    assert {p.resolve() for p in atomic_targets} & shared
    assert not list(metadata_dir.glob("*.tmp"))
    h2 = pl.read_parquet(metadata_dir / "trait_heritability.parquet")
    assert "MONDO_0005147" in h2["efo_id"].to_list()

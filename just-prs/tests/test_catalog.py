"""Integration tests for PGS Catalog REST API client."""

from pathlib import Path

from just_prs.catalog import PGSCatalogClient
from just_prs.prs_catalog import PRSCatalog
from just_prs.scoring import resolve_cache_dir
from just_prs.vcf import read_genotypes


def test_compute_prs_genotypes_lf_reuse_and_attach_performance(
    vcf_path: Path, scoring_cache_dir: Path
) -> None:
    """F23: PRSCatalog.compute_prs accepts genotypes_lf and still attaches performance.

    The reused-genotype-frame path must produce the same score as re-reading the VCF,
    and attach_performance must populate PRSResult.performance in one call.
    """
    catalog = PRSCatalog(cache_dir=resolve_cache_dir())

    from_vcf = catalog.compute_prs(vcf_path=vcf_path, pgs_id="PGS000001")

    geno_lf = read_genotypes(vcf_path)
    from_lf = catalog.compute_prs(
        vcf_path=vcf_path,
        pgs_id="PGS000001",
        genotypes_lf=geno_lf,
        attach_performance=True,
    )

    assert from_lf.score == from_vcf.score
    assert from_lf.variants_matched == from_vcf.variants_matched
    assert from_lf.performance is not None  # attach_performance honored on the reuse path


def test_get_score_pgs000001() -> None:
    """Fetch PGS000001 and verify its metadata."""
    with PGSCatalogClient() as client:
        score = client.get_score("PGS000001")

    assert score.id == "PGS000001"
    assert score.trait_reported == "Breast cancer"
    assert score.variants_number == 77
    assert score.publication is not None
    assert score.publication.id == "PGP000001"


def test_get_score_download_url() -> None:
    """Verify harmonized download URL resolution."""
    with PGSCatalogClient() as client:
        url = client.get_score_download_url("PGS000001", genome_build="GRCh38")

    assert "PGS000001" in url
    assert "GRCh38" in url
    assert url.startswith("https://")


def test_get_trait_coronary_artery_disease() -> None:
    """Fetch coronary artery disease trait and verify metadata."""
    with PGSCatalogClient() as client:
        trait = client.get_trait("MONDO_0005010")

    assert trait.id == "MONDO_0005010"
    assert trait.label == "coronary artery disorder"
    assert "coronary artery disease" in trait.trait_synonyms
    assert len(trait.associated_pgs_ids) > 0
    assert any(pid.startswith("PGS") for pid in trait.associated_pgs_ids)


def test_search_traits_diabetes() -> None:
    """Search for diabetes traits and verify results."""
    with PGSCatalogClient() as client:
        results = client.search_traits("diabetes", limit=5)

    assert len(results) > 0
    labels = [t.label for t in results if t.label is not None]
    assert any("diabetes" in label.lower() for label in labels)

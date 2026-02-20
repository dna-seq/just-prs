"""Integration tests for PGS Catalog REST API client."""

from just_prs.catalog import PGSCatalogClient


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
        trait = client.get_trait("EFO_0001645")

    assert trait.id == "EFO_0001645"
    assert trait.label == "coronary artery disease"
    assert len(trait.associated_pgs_ids) > 0
    assert any(pid.startswith("PGS") for pid in trait.associated_pgs_ids)


def test_search_traits_diabetes() -> None:
    """Search for diabetes traits and verify results."""
    with PGSCatalogClient() as client:
        results = client.search_traits("diabetes", limit=5)

    assert len(results) > 0
    labels = [t.label for t in results if t.label is not None]
    assert any("diabetes" in label.lower() for label in labels)

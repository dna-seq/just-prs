from pathlib import Path

from just_prs.heritability import download_gwas_atlas_heritability


def test_download_gwas_atlas_heritability_via_release_endpoint(tmp_path: Path) -> None:
    """GWAS Atlas serves release files via POST, not direct static URLs."""
    df = download_gwas_atlas_heritability(tmp_path / "gwas_atlas.parquet", overwrite=True)

    assert df.height > 1000
    assert {"trait_label", "h2_observed", "ancestry", "source", "source_detail", "confidence"} <= set(df.columns)
    assert set(df["source"].unique().to_list()) == {"gwas_atlas"}
    assert "high" not in set(df["confidence"].unique().to_list())
    assert df["source_detail"].str.contains("archival snapshot \\(2019\\)").all()

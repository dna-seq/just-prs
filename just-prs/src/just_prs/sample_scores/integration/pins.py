"""Pinned source revisions and SHA256s for the public sample-score integration.

These hashes are the offline gate. Staging verifies downloaded bytes against
this table and then closes the network boundary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SAMPLE_SCORES_REPO = "just-dna-seq/prs-sample-scores"
CATALOG_REPO = "just-dna-seq/pgs-catalog"
PERCENTILES_REPO = "just-dna-seq/prs-percentiles"

SAMPLE_SCORES_REVISION = "1248dc9223f131c953af8d90a08e2472685d8fec"
EVIDENCE_PARENT_REVISION = "d575527f1767fe199dc90ea8b50eb286bce965e5"
CATALOG_REVISION = "1c32cc15987e6bba5a3cf5929fcff9926742a7eb"
PERCENTILES_REVISION = "440e705d1b02bec48cfb22ebc374e16cf25806a5"

SUPERPOPULATIONS: tuple[str, ...] = ("AFR", "AMR", "EAS", "EUR", "SAS")
FINE_COHORT_DISCLAIMER = (
    "CEU and IBS are nearest 1000 Genomes reference cohorts, not ethnicity, "
    "nationality, or separate percentile populations."
)

RUNTIME_MANIFEST_BYTES_SHA256 = "a95f8aa1e08af02d3427c41cadcb618b857285f7c0074c2622e2aeffccb01994"
EVIDENCE_MANIFEST_BYTES_SHA256 = "9d8c65853af8277920f4c05d658ad37b7001ba0ffb54cc4685196128780e4853"


class PinnedFile(BaseModel):
    """One immutable source file. Paths are repo-relative."""

    repo_id: str
    revision: str
    repo_path: str
    sha256: str
    expected_rows: int | None = None
    local_name: str


PINNED_FILES: tuple[PinnedFile, ...] = (
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/samples.parquet",
        sha256="039b5fd71ccb8fd2b0724285cdfd8f43d0e7fcc6532f4a1ace0c234e62f34cbc",
        expected_rows=7,
        local_name="samples.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/runtime_results.parquet",
        sha256="2bf25723c8872d6577427a539d75ce165eeea4d740827d5e230e3de1089d36d6",
        expected_rows=74718,
        local_name="runtime_results.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/runtime_manifest.json",
        sha256=RUNTIME_MANIFEST_BYTES_SHA256,
        local_name="runtime_manifest.json",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/sample_ancestry.parquet",
        sha256="432e17bf0bbe116aeb5458c03ae5244ec5c3b27a5b28e1b5e2f5a8ca9f046872",
        expected_rows=7,
        local_name="sample_ancestry.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/evidence_manifest.json",
        sha256=EVIDENCE_MANIFEST_BYTES_SHA256,
        local_name="evidence_manifest.json",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/traits.parquet",
        sha256="",
        expected_rows=844,
        local_name="traits.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/score_trait_links.parquet",
        sha256="",
        expected_rows=6789,
        local_name="score_trait_links.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/papers.parquet",
        sha256="",
        expected_rows=789,
        local_name="papers.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/score_paper_links.parquet",
        sha256="",
        expected_rows=11255,
        local_name="score_paper_links.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/guidelines.parquet",
        sha256="",
        expected_rows=13,
        local_name="guidelines.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/guideline_trait_links.parquet",
        sha256="",
        expected_rows=20,
        local_name="guideline_trait_links.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/actionability.parquet",
        sha256="",
        expected_rows=852,
        local_name="actionability.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/trait_contexts.parquet",
        sha256="",
        expected_rows=646,
        local_name="trait_contexts.parquet",
    ),
    PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/record_search_terms.parquet",
        sha256="",
        expected_rows=3623,
        local_name="record_search_terms.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/scores.parquet",
        sha256="60947990d0739d1d28bce066afe4f16c22753f21bee6780cc996b2c6c091e658",
        expected_rows=5337,
        local_name="scores.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/best_performance.parquet",
        sha256="101499084566b5aa679f806464b31a26560ebe6836ee70c6266fd967098d7fb0",
        expected_rows=5319,
        local_name="best_performance.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/performance.parquet",
        sha256="891d6fb37da6eadc81b1fc33508f64206c84b9a3f460909e71c4150143b9f2ad",
        expected_rows=21196,
        local_name="performance.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/pgs_quality_scores.parquet",
        sha256="3843dd8d73367c699e7ab10326811a64a328d8922149eaac2056d4887097297d",
        expected_rows=604,
        local_name="pgs_quality_scores.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/publications.parquet",
        sha256="f69f60feba39ed35cd2a8a1756c3e322b88ae1f42877f7d05db11d572849fe40",
        expected_rows=790,
        local_name="publications.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/trait_prevalence.parquet",
        sha256="a0d11c2eec44066a659b8c1e9145fc8996b9bba3fa86cfcbf6ac11001da00523",
        expected_rows=3087,
        local_name="trait_prevalence.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/trait_heritability.parquet",
        sha256="1cdd6b9766a16b3a4546059d2ee601d2ed459dc52c374e87c62838209d44aa81",
        expected_rows=2754,
        local_name="trait_heritability.parquet",
    ),
    PinnedFile(
        repo_id=CATALOG_REPO,
        revision=CATALOG_REVISION,
        repo_path="data/metadata/catalog_scoring_flags.parquet",
        sha256="1de49ed1c47be604eb31112c91417c11bad07bde5f4aa7a8a83e8a0aec0ff93f",
        expected_rows=847,
        local_name="catalog_scoring_flags.parquet",
    ),
    PinnedFile(
        repo_id=PERCENTILES_REPO,
        revision=PERCENTILES_REVISION,
        repo_path="data/1000g_distributions.parquet",
        sha256="d23b1f8c4bc1df22e7749f8c473bd2066616455a8acc78c8a3d5513409993b86",
        expected_rows=26660,
        local_name="1000g_distributions.parquet",
    ),
    PinnedFile(
        repo_id=PERCENTILES_REPO,
        revision=PERCENTILES_REVISION,
        repo_path="data/1000g_quality.parquet",
        sha256="01a55eed49553839813d64b90c5766c879344f802907dba1801da97ca9aee5d5",
        expected_rows=5332,
        local_name="1000g_quality.parquet",
    ),
    PinnedFile(
        repo_id=PERCENTILES_REPO,
        revision=PERCENTILES_REVISION,
        repo_path="data/1000g_distribution_quality_issues.parquet",
        sha256="3dca33b829381498639f035b1047eeb2faf9b2b90d93a72bb9fccaffef4bf323",
        expected_rows=4331,
        local_name="1000g_distribution_quality_issues.parquet",
    ),
    PinnedFile(
        repo_id=PERCENTILES_REPO,
        revision=PERCENTILES_REVISION,
        repo_path="data/1000g_distribution_audit_summary.json",
        sha256="05cb38f671eba445ea142e6b8a8f2b124b36cbdc72f2b75c759e1c8e1ae77233",
        local_name="1000g_distribution_audit_summary.json",
    ),
)


class StagingIndexRow(BaseModel):
    """One staged source file recorded after hash verification."""

    repo_id: str
    revision: str
    repo_path: str
    sha256: str
    bytes: int
    rows: int | None = None
    retrieved_at: str
    local_name: str
    omitted: bool = False
    omit_reason: str | None = None


class PinOverrides(BaseModel):
    """Test-only pin table. Production uses ``PINNED_FILES``."""

    files: list[PinnedFile] = Field(default_factory=list)


def required_pins(*, include_evidence_hashes: bool = True) -> tuple[PinnedFile, ...]:
    """Return production pins. Evidence parquet hashes are filled from the manifest."""
    return PINNED_FILES

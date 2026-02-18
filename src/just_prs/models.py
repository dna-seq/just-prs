"""Pydantic v2 models for PGS Catalog data and PRS computation results."""

from pydantic import BaseModel, Field


class PublicationInfo(BaseModel):
    """Publication metadata from PGS Catalog."""

    id: str = Field(description="PGS Catalog Publication ID (PGP)")
    title: str | None = Field(default=None, description="Publication title")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    PMID: int | None = Field(default=None, description="PubMed ID")
    journal: str | None = Field(default=None, description="Journal name")
    firstauthor: str | None = Field(default=None, description="First author")
    date_publication: str | None = Field(default=None, description="Publication date")


class HarmonizedFileInfo(BaseModel):
    """URLs for harmonized scoring files by genome build."""

    positions: str | None = Field(default=None, description="URL for position-harmonized file")


class ScoreInfo(BaseModel):
    """Polygenic Score metadata from PGS Catalog REST API."""

    id: str = Field(description="PGS Catalog Score ID (e.g. PGS000001)")
    name: str | None = Field(default=None, description="Score name")
    trait_reported: str | None = Field(default=None, description="Reported trait")
    trait_additional: str | None = Field(default=None, description="Additional trait info")
    variants_number: int | None = Field(default=None, description="Number of variants in score")
    variants_genomebuild: str | None = Field(default=None, description="Original genome build")
    weight_type: str | None = Field(default=None, description="Weight type (beta, OR, HR, etc.)")
    ftp_scoring_file: str | None = Field(default=None, description="URL for original scoring file")
    ftp_harmonized_scoring_files: dict[str, HarmonizedFileInfo] | None = Field(
        default=None, description="Harmonized scoring file URLs by genome build"
    )
    publication: PublicationInfo | None = Field(default=None, description="Source publication")
    matches_publication: bool | None = Field(default=None)
    license: str | None = Field(default=None, description="License text")

    def get_download_url(self, genome_build: str = "GRCh38") -> str | None:
        """Get the harmonized scoring file download URL for a given genome build."""
        if self.ftp_harmonized_scoring_files is None:
            return self.ftp_scoring_file
        build_info = self.ftp_harmonized_scoring_files.get(genome_build)
        if build_info is not None and build_info.positions is not None:
            return build_info.positions
        return self.ftp_scoring_file


class TraitInfo(BaseModel):
    """Trait metadata from PGS Catalog REST API."""

    id: str = Field(description="EFO trait ID (e.g. EFO_0001645)")
    label: str | None = Field(default=None, description="Trait label")
    description: str | None = Field(default=None, description="Trait description")
    url: str | None = Field(default=None, description="EFO URL")
    trait_categories: list[str] = Field(default_factory=list, description="Trait categories")
    trait_synonyms: list[str] = Field(default_factory=list, description="Trait synonyms")
    associated_pgs_ids: list[str] = Field(
        default_factory=list, description="Associated PGS IDs"
    )
    child_associated_pgs_ids: list[str] = Field(
        default_factory=list, description="Child trait associated PGS IDs"
    )


class EffectSizeInfo(BaseModel):
    """A single effect size or classification accuracy metric."""

    name_short: str = Field(description="Short name (e.g. OR, HR, AUROC, C-index)")
    name_long: str | None = Field(default=None, description="Full metric name")
    estimate: float = Field(description="Point estimate")
    ci_lower: float | None = Field(default=None, description="Lower bound of confidence interval")
    ci_upper: float | None = Field(default=None, description="Upper bound of confidence interval")
    se: float | None = Field(default=None, description="Standard error")


class PerformanceInfo(BaseModel):
    """Performance metrics for a PGS score from PGS Catalog evaluation studies."""

    ppm_id: str = Field(description="Performance Metric ID (PPM)")
    effect_sizes: list[EffectSizeInfo] = Field(default_factory=list)
    class_acc: list[EffectSizeInfo] = Field(default_factory=list)
    sample_number: int | None = Field(default=None, description="Evaluation sample size")
    ancestry_broad: str | None = Field(default=None, description="Broad ancestry of evaluation cohort")
    phenotyping_reported: str | None = Field(default=None, description="Phenotype evaluated")
    covariates: str | None = Field(default=None, description="Covariates used in evaluation")


class PRSResult(BaseModel):
    """Result of a polygenic risk score computation."""

    pgs_id: str = Field(description="PGS Catalog Score ID")
    score: float = Field(description="Computed polygenic risk score")
    variants_matched: int = Field(description="Number of scoring variants matched in VCF")
    variants_total: int = Field(description="Total number of variants in scoring file")
    match_rate: float = Field(description="Fraction of scoring variants matched (0-1)")
    trait_reported: str | None = Field(default=None, description="Reported trait for the score")
    performance: PerformanceInfo | None = Field(
        default=None, description="Best available performance metric from PGS Catalog"
    )

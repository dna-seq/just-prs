"""Blocking validators for staged sources and integration outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from just_prs.canary_audit import excluded_pgs_ids
from just_prs.sample_scores.completeness import validate_runtime_results
from just_prs.sample_scores.evidence.checks import validate_evidence_tables
from just_prs.sample_scores.evidence.contexts import drug_response_pgs_ids
from just_prs.sample_scores.evidence.models import (
    ActionabilityRecord,
    GuidelineRecord,
    GuidelineTraitLink,
    PaperRecord,
    RecordSearchTerm,
    ScorePaperLink,
    ScoreTraitLink,
    TraitContextRecord,
    TraitRecord,
)
from just_prs.sample_scores.integration.models import (
    MODEL_ANALYSIS_PRIMARY_KEY,
    TRAIT_SUMMARY_PRIMARY_KEY,
)
from just_prs.sample_scores.integration.pins import SUPERPOPULATIONS
from just_prs.sample_scores.integration.staging import work_cache_dir
from just_prs.sample_scores.models import EXPECTED_PUBLIC_SAMPLE_ANCESTRY, PRIVATE_INGEST_ALIASES
from just_prs.scoring import parquet_cache_is_readable
from just_prs.trait_summary import summarize_trait_rows


class IntegrationCheckError(ValueError):
    """One or more blocking integration invariants failed."""


class IntegrationCheckReport(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    n_analysis_rows: int = 0
    n_eligible: int = 0
    n_summaries: int = 0
    pgs_without_distribution: list[str] = Field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise IntegrationCheckError("; ".join(self.issues) or "integration checks failed")


def _load_model_rows(output_dir: Path, name: str, model: type[object]) -> list[object]:
    path = output_dir / f"{name}.parquet"
    if not parquet_cache_is_readable(path):
        return []
    return [model.model_validate(row) for row in pl.read_parquet(path).iter_rows(named=True)]


def validate_staged_sources(cache_dir: Path) -> list[str]:
    """Re-check published runtime/evidence artifacts without rescoring."""
    issues: list[str] = []
    work = work_cache_dir(cache_dir)
    scores = pl.read_parquet(work / "metadata" / "scores.parquet")
    catalog_ids = sorted(scores["pgs_id"].unique().to_list())
    report = validate_runtime_results(
        work,
        expected_pgs_ids=catalog_ids,
        require_ancestry=True,
    )
    if not report.passed:
        issues.extend(report.issues)

    sample_dir = work / "sample_scores"
    traits = _load_model_rows(sample_dir, "traits", TraitRecord)
    links = _load_model_rows(sample_dir, "score_trait_links", ScoreTraitLink)
    papers = _load_model_rows(sample_dir, "papers", PaperRecord)
    paper_links = _load_model_rows(sample_dir, "score_paper_links", ScorePaperLink)
    guidelines = _load_model_rows(sample_dir, "guidelines", GuidelineRecord)
    guideline_links = _load_model_rows(sample_dir, "guideline_trait_links", GuidelineTraitLink)
    actionability = _load_model_rows(sample_dir, "actionability", ActionabilityRecord)
    contexts = _load_model_rows(sample_dir, "trait_contexts", TraitContextRecord)
    terms = _load_model_rows(sample_dir, "record_search_terms", RecordSearchTerm)
    pgs_by_trait: dict[str, set[str]] = {}
    for link in links:
        payload = link.model_dump()
        pgs_by_trait.setdefault(str(payload["trait_id"]), set()).add(str(payload["pgs_id"]))
    try:
        validate_evidence_tables(
            traits=traits,  # type: ignore[arg-type]
            score_trait_links=links,
            papers=papers,  # type: ignore[arg-type]
            score_paper_links=paper_links,
            guidelines=guidelines,  # type: ignore[arg-type]
            guideline_trait_links=guideline_links,
            actionability=actionability,
            trait_contexts=contexts,  # type: ignore[arg-type]
            record_search_terms=terms,
            drug_response_pgs_ids=drug_response_pgs_ids(scores.iter_rows(named=True)),
            score_trait_pgs_by_trait=pgs_by_trait,
        )
    except Exception as exc:
        issues.append(str(exc))

    samples = pl.read_parquet(sample_dir / "samples.parquet")
    leaked = [
        alias
        for aliases in samples["aliases"].to_list()
        for alias in (aliases or [])
        if str(alias).casefold() in PRIVATE_INGEST_ALIASES
    ]
    if leaked:
        issues.append(f"private ingest aliases published: {leaked}")
    if "oksana" in {str(sid).casefold() for sid in samples["sample_id"].to_list()}:
        issues.append("private sample_id oksana present in samples.parquet")
    ancestry = pl.read_parquet(sample_dir / "sample_ancestry.parquet")
    for row in ancestry.iter_rows(named=True):
        expected = EXPECTED_PUBLIC_SAMPLE_ANCESTRY.get(str(row["sample_id"]))
        if expected is None:
            continue
        want_super, want_fine = expected
        if row["superpopulation"] != want_super or row["fine_population"] != want_fine:
            issues.append(
                f"{row['sample_id']} ancestry {row['superpopulation']}/{row['fine_population']} "
                f"!= {want_super}/{want_fine}"
            )
    return issues


def validate_model_analysis(analysis: pl.DataFrame, *, runtime: pl.DataFrame) -> list[str]:
    issues: list[str] = []
    if analysis.height != runtime.height:
        issues.append(
            f"model_analysis rows {analysis.height} != runtime rows {runtime.height}"
        )
    dupes = analysis.group_by(list(MODEL_ANALYSIS_PRIMARY_KEY)).len().filter(pl.col("len") > 1)
    if dupes.height:
        issues.append(f"{dupes.height} duplicate model_analysis keys")
    if analysis.filter(pl.col("selected_superpopulation").is_in(["CEU", "IBS"])).height:
        issues.append("selected_superpopulation used a fine 1000G cohort code")
    ineligible = analysis.filter(~pl.col("analysis_eligible"))
    for row in ineligible.iter_rows(named=True):
        metrics = row.get("population_metrics") or []
        for metric in metrics:
            if metric.get("available") or metric.get("percentile") is not None or metric.get("z_score") is not None:
                issues.append("ineligible row carries lookup-like percentile values")
                break
        if issues and issues[-1].startswith("ineligible"):
            break
    eligible = analysis.filter(pl.col("analysis_eligible"))
    for row in eligible.head(min(200, eligible.height)).iter_rows(named=True):
        metrics = row.get("population_metrics") or []
        if len(metrics) != len(SUPERPOPULATIONS):
            issues.append("population_metrics is not exactly five superpopulations")
            break
        codes = [item.get("superpopulation") for item in metrics]
        if codes != list(SUPERPOPULATIONS):
            issues.append(f"population_metrics order/codes {codes}")
            break
        for metric in metrics:
            if not metric.get("available"):
                if metric.get("percentile") is not None or metric.get("z_score") is not None:
                    issues.append("unavailable metric has fabricated z/percentile")
                    break
                if not metric.get("exclusion_reason"):
                    issues.append("unavailable metric missing exclusion_reason")
                    break
                continue
            pct = metric.get("percentile")
            std = metric.get("std")
            mean = metric.get("mean")
            z = metric.get("z_score")
            if pct is None or z is None or std is None or mean is None:
                issues.append("available metric missing stats")
                break
            if not (0.0 <= float(pct) <= 100.0) or not math.isfinite(float(z)) or float(std) <= 0:
                issues.append("available metric fails percentile invariants")
                break
    runtime_ids = set(runtime["pgs_id"].to_list())
    extra = set()
    for row in analysis.iter_rows(named=True):
        extra.update(item.get("pgs_id") for item in [] if False)
    invented = [pgs for pgs in extra if pgs not in runtime_ids]
    if invented:
        issues.append(f"invented analysis PGS IDs: {invented[:5]}")
    return issues


def validate_trait_summaries(
    summaries: pl.DataFrame,
    analysis: pl.DataFrame,
    *,
    max_recompute: int | None = None,
) -> list[str]:
    issues: list[str] = []
    if summaries.height == 0:
        issues.append("trait_summaries is empty")
        return issues
    dupes = summaries.group_by(list(TRAIT_SUMMARY_PRIMARY_KEY)).len().filter(pl.col("len") > 1)
    if dupes.height:
        issues.append(f"{dupes.height} duplicate trait_summary keys")
    if summaries.filter(pl.col("percentile_population").is_in(["CEU", "IBS"])).height:
        issues.append("trait summaries used a fine cohort as percentile_population")
    profiles = set(summaries["score_profile_id"].unique().to_list())
    if any("," in str(item) or " " in str(item) for item in profiles):
        issues.append("summary mixed multiple profiles in score_profile_id")
    if summaries.filter(pl.col("n_eligible") < 0).height:
        issues.append("negative eligible counts")

    from just_prs.sample_scores.integration.summaries import _selected_metrics, _trait_model_row

    if max_recompute == 0:
        return issues
    expected = {
        (row["sample_id"], row["trait_id"], row["score_profile_id"], row["percentile_population"]): row
        for row in summaries.iter_rows(named=True)
    }
    slim = _selected_metrics(analysis).explode("traits", empty_as_null=True).unnest("traits")
    slim = slim.filter(pl.col("trait_id").is_not_null() & pl.col("analysis_eligible"))
    checked = 0
    for key, block in slim.group_by(
        "sample_id", "trait_id", "score_profile_id", "selected_superpopulation"
    ):
        sample_id, trait_id, profile, pop = (str(part) for part in key)
        row = expected.get((sample_id, trait_id, profile, pop))
        if row is None:
            issues.append(f"missing summary for {sample_id} {trait_id} {profile}")
            break
        models = [_trait_model_row(item, item, pop) for item in block.to_dicts()]
        stats = summarize_trait_rows(
            models,
            model_scope="usable",
            selected_ancestry=pop,
            percentile_source="selected",
            sample_ancestries=[pop],
        )
        if stats.n_usable != row.get("n_usable"):
            issues.append(f"n_usable mismatch {sample_id} {trait_id}: {stats.n_usable} != {row.get('n_usable')}")
            break
        left, right = stats.median_pct, row.get("median_pct")
        if left is None and right is None:
            pass
        elif left is None or right is None or abs(float(left) - float(right)) > 1e-9:
            issues.append(f"median mismatch {sample_id} {trait_id} {profile}: {left} != {right}")
            break
        checked += 1
        if max_recompute is not None and checked >= max_recompute:
            break
    return issues


def validate_integration_outputs(
    *,
    cache_dir: Path,
    analysis: pl.DataFrame,
    summaries: pl.DataFrame,
    runtime: pl.DataFrame,
    recompute_summaries: bool = True,
) -> IntegrationCheckReport:
    issues = validate_staged_sources(cache_dir)
    issues.extend(validate_model_analysis(analysis, runtime=runtime))
    issues.extend(
        validate_trait_summaries(
            summaries,
            analysis,
            max_recompute=None if recompute_summaries else 0,
        )
    )
    flags_path = work_cache_dir(cache_dir) / "metadata" / "catalog_scoring_flags.parquet"
    excluded = set(excluded_pgs_ids(pl.read_parquet(flags_path))) if parquet_cache_is_readable(flags_path) else set()
    leaked_eligible = analysis.filter(
        pl.col("analysis_eligible") & pl.col("pgs_id").is_in(sorted(excluded))
    )
    if leaked_eligible.height:
        issues.append(f"{leaked_eligible.height} quarantined PGS marked analysis_eligible")
    if summaries.filter(pl.col("trait_id").is_in(list(excluded))).height:
        pass
    if "oksana" in " ".join(json.dumps(analysis.head(1).to_dicts())).lower():
        issues.append("oksana leaked into analysis preview")
    return IntegrationCheckReport(
        passed=not issues,
        issues=issues,
        n_analysis_rows=analysis.height,
        n_eligible=int(analysis.filter(pl.col("analysis_eligible")).height),
        n_summaries=summaries.height,
    )

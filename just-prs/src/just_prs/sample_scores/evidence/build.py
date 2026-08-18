"""Build and optionally publish evidence tables without scoring genomes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from eliot import start_action
from pydantic import BaseModel

from just_prs.sample_scores.evidence.actionability import derive_actionability
from just_prs.sample_scores.evidence.checks import validate_evidence_tables
from just_prs.sample_scores.evidence.contexts import (
    assign_trait_contexts,
    build_record_search_terms,
    drug_response_pgs_ids,
)
from just_prs.sample_scores.evidence.docs import render_agents, render_readme
from just_prs.sample_scores.evidence.guidelines import build_guideline_tables
from just_prs.sample_scores.evidence.papers import build_paper_tables
from just_prs.sample_scores.evidence.traits import build_trait_tables
from just_prs.sample_scores.models import DEFAULT_SAMPLE_SCORES_REPO
from just_prs.sample_scores.store import MANIFEST_FILENAME, sample_scores_dir
from just_prs.scoring import parquet_cache_is_readable, resolve_cache_dir

EVIDENCE_TABLE_FILES: tuple[str, ...] = (
    "traits.parquet",
    "score_trait_links.parquet",
    "papers.parquet",
    "score_paper_links.parquet",
    "guidelines.parquet",
    "guideline_trait_links.parquet",
    "actionability.parquet",
    "trait_contexts.parquet",
    "record_search_terms.parquet",
)

EVIDENCE_DOC_FILES: tuple[str, ...] = ("README.md", "AGENTS.md")


@dataclass
class EvidenceBuildResult:
    """Counts written by an evidence build. No sample scores."""

    n_traits: int = 0
    n_score_trait_links: int = 0
    n_papers: int = 0
    n_score_paper_links: int = 0
    n_guidelines: int = 0
    n_guideline_trait_links: int = 0
    n_actionability: int = 0
    n_trait_contexts: int = 0
    n_record_search_terms: int = 0
    output_dir: Path | None = None
    pushed: bool = False
    tables: dict[str, int] = field(default_factory=dict)


def _annotation_to_polars(annotation: object) -> pl.DataType:
    args = getattr(annotation, "__args__", ())
    if args:
        non_none = [item for item in args if item is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
    mapping: dict[object, pl.DataType] = {
        str: pl.Utf8,
        bool: pl.Boolean,
        int: pl.Int64,
        float: pl.Float64,
    }
    return mapping.get(annotation, pl.Utf8)


def _rows_to_frame(rows: list[BaseModel]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    model = type(rows[0])
    schema = {
        name: _annotation_to_polars(field.annotation)
        for name, field in model.model_fields.items()
    }
    return pl.DataFrame(
        [row.model_dump() for row in rows],
        schema=schema,
        infer_schema_length=None,
    )


def _write_table(name: str, rows: list[BaseModel], output_dir: Path) -> Path:
    path = output_dir / f"{name}.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    _rows_to_frame(rows).write_parquet(path)
    return path


def _load_catalog_frames(cache_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
    from just_prs.prs_catalog import PRSCatalog

    catalog = PRSCatalog(cache_dir=cache_dir)
    scores = catalog.scores(include_harmonized=True, include_excluded=True).collect()
    publications = catalog.publications()
    publications_df = publications.collect() if publications is not None else None
    performance_path = catalog.metadata_dir / "performance.parquet"
    performance_df = (
        pl.read_parquet(performance_path)
        if parquet_cache_is_readable(performance_path)
        else None
    )
    return scores, publications_df, performance_df


def write_evidence_manifest(
    output_dir: Path,
    result: EvidenceBuildResult,
    *,
    repo_id: str,
    published_at: str,
) -> Path:
    from just_prs import __version__

    existing: dict[str, object] = {}
    path = output_dir / MANIFEST_FILENAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        **existing,
        "schema_version": existing.get("schema_version", 1),
        "repo_id": repo_id,
        "evidence_published_at": published_at,
        "just_prs_version": __version__,
        "evidence_tables": list(EVIDENCE_TABLE_FILES),
        "n_traits": result.n_traits,
        "n_papers": result.n_papers,
        "n_guidelines": result.n_guidelines,
        "note": (
            "Evidence tables are catalog-level. Runtime sample scores "
            "(samples.parquet / runtime_results.parquet) are a later upload. "
            "Do not invent scores."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_sample_score_evidence(
    cache_dir: Path | None = None,
    *,
    allow_network: bool = True,
    push: bool = False,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
    token: str | None = None,
) -> EvidenceBuildResult:
    """Read cleaned catalog cache, write evidence tables, optionally push to HF.

    Does not score genomes and does not wait for ``runtime_results``.
    """
    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    output_dir = sample_scores_dir(root)
    evidence_cache = output_dir / "evidence_cache"
    published_at = datetime.now(timezone.utc).isoformat()
    with start_action(action_type="sample_scores:evidence:build", allow_network=allow_network):
        scores_df, publications_df, performance_df = _load_catalog_frames(root)
        traits, score_trait_links = build_trait_tables(
            scores_df,
            ols_cache_dir=evidence_cache / "ols",
            allow_network=allow_network,
            retrieved_at=published_at,
        )
        papers, score_paper_links = build_paper_tables(
            scores_df,
            publications_df,
            performance_df,
            epmc_cache_dir=evidence_cache / "europepmc",
            allow_network=allow_network,
        )
        known_trait_ids = {trait.trait_id for trait in traits}
        guidelines, guideline_links = build_guideline_tables(
            known_trait_ids,
            cache_dir=evidence_cache / "guidelines",
            allow_network=allow_network,
            retrieved_at=published_at,
        )
        actionability = derive_actionability(traits, guidelines, guideline_links)
        drug_pgs = drug_response_pgs_ids(scores_df.iter_rows(named=True))
        contexts = assign_trait_contexts(
            traits,
            score_trait_links,
            drug_response_pgs_ids=drug_pgs,
        )
        search_terms = build_record_search_terms(
            traits, contexts, guidelines, guideline_links
        )
        pgs_by_trait: dict[str, set[str]] = {}
        for link in score_trait_links:
            pgs_by_trait.setdefault(link.trait_id, set()).add(link.pgs_id)
        validate_evidence_tables(
            traits=traits,
            score_trait_links=score_trait_links,
            papers=papers,
            score_paper_links=score_paper_links,
            guidelines=guidelines,
            guideline_trait_links=guideline_links,
            actionability=actionability,
            trait_contexts=contexts,
            record_search_terms=search_terms,
            drug_response_pgs_ids=drug_pgs,
            score_trait_pgs_by_trait=pgs_by_trait,
        )
        tables = {
            "traits": traits,
            "score_trait_links": score_trait_links,
            "papers": papers,
            "score_paper_links": score_paper_links,
            "guidelines": guidelines,
            "guideline_trait_links": guideline_links,
            "actionability": actionability,
            "trait_contexts": contexts,
            "record_search_terms": search_terms,
        }
        for name, rows in tables.items():
            _write_table(name, rows, output_dir)
        result = EvidenceBuildResult(
            n_traits=len(traits),
            n_score_trait_links=len(score_trait_links),
            n_papers=len(papers),
            n_score_paper_links=len(score_paper_links),
            n_guidelines=len(guidelines),
            n_guideline_trait_links=len(guideline_links),
            n_actionability=len(actionability),
            n_trait_contexts=len(contexts),
            n_record_search_terms=len(search_terms),
            output_dir=output_dir,
            tables={name: len(rows) for name, rows in tables.items()},
        )
        write_evidence_manifest(output_dir, result, repo_id=repo_id, published_at=published_at)
        (output_dir / "README.md").write_text(
            render_readme(
                published_at=published_at,
                n_traits=result.n_traits,
                n_papers=result.n_papers,
                n_guidelines=result.n_guidelines,
            ),
            encoding="utf-8",
        )
        (output_dir / "AGENTS.md").write_text(render_agents(), encoding="utf-8")
        if push:
            publish_sample_score_evidence(output_dir, repo_id=repo_id, token=token)
            result.pushed = True
        return result


def publish_sample_score_evidence(
    local_dir: Path,
    *,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
    token: str | None = None,
) -> None:
    """Upload evidence parquets, manifest, and root docs. Never uploads runtime scores."""
    from just_prs.hf import push_sample_score_evidence

    push_sample_score_evidence(local_dir, repo_id=repo_id, token=token)

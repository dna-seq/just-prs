"""Score curation: paper queue, trait worklists, lookup context, checks, and progress lines.

Backs the ``curate-papers`` and ``curate-trait`` skills of the just-prs Claude Code plugin:

    prs curation --help

Curated files live under ``curation/`` (source of truth, small text, reviewed in git).
Fetched paper text lives under ``<cache>/annotation_lookups/<PGP>/`` and trait triage output
under ``<cache>/annotation_lookups/_traits/<slug>/`` (never in the repo).
Rules and file formats: ``docs/curation-rules.md``.
"""

from __future__ import annotations

import csv
import html
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from just_prs.prs_catalog import PRSCatalog
from just_prs.scoring import ensure_scoring_file, resolve_cache_dir

DEFAULT_ROOT = Path("curation")  # relative to the working directory (the repo root)
MAX_ATTEMPTS = 3
STALE_HOURS = 6.0
CORR_THRESHOLD = 0.2
CORR_MIN_SAMPLES = 100
TOP_VARIANTS_MAX_SCORES = 20

QUEUE_COLUMNS = [
    "pgp_id",
    "phase",
    "priority",
    "n_scores",
    "target_pgs_ids",
    "status",
    "attempts",
    "n_annotated",
    "n_verified",
    "last_error",
    "claimed_at",
    "updated_at",
]

_MIXED_POLARITY_RE = re.compile(
    r"\b(never|ever|absence|without|non-|age at|onset|survival|mortality|longevity|response)\b",
    flags=re.IGNORECASE,
)
_SOURCE_SUFFIXES = {".txt", ".md", ".xml", ".html", ".htm", ".json", ".tsv", ".csv"}
_AGENT_OUTPUT_PREFIX = "lookup"  # lookup.json / lookup_<slug>.json are agent output, never evidence

TRIAGE_SIGN_R = 0.1
TRIAGE_MAX_ROWS = 200

app = typer.Typer(help="Curate score direction and trait clusters (curation/ in the current directory).", no_args_is_help=True)
RootOpt = Annotated[Path, typer.Option("--root", help="Curation directory")]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class MeasurementKind(StrEnum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    ONEHOT_CATEGORY = "onehot_category"
    TIME_TO_EVENT = "time_to_event"
    COMPOSITE = "composite"
    RESPONSE = "response"


class Direction(StrEnum):
    INCREASES = "increases"
    DECREASES = "decreases"


class Valence(StrEnum):
    DESIRABLE = "desirable"
    UNDESIRABLE = "undesirable"
    NEUTRAL = "neutral"
    CONTEXT_DEPENDENT = "context_dependent"


class AnnotationStatus(StrEnum):
    AGENT_PROPOSED = "agent_proposed"
    HUMAN_VERIFIED = "human_verified"
    REJECTED = "rejected"


class QueueStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CHECKS_PASSED = "checks_passed"
    CHECKS_FAILED = "checks_failed"
    BLOCKED = "blocked"
    HUMAN_VERIFIED = "human_verified"


class EvidenceSource(StrEnum):
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    SUPPLEMENT = "supplement"
    CATALOG = "catalog"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LeadVariantStatus(StrEnum):
    AGREE = "agree"
    DISAGREE = "disagree"
    MIXED = "mixed"
    NOT_RUN = "not_run"


class Level(StrEnum):
    ERROR = "ERROR"
    WARN = "WARN"
    NOT_RUN = "NOT_RUN"
    OK = "OK"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Phenotype(_Strict):
    label: str
    sign: int

    @field_validator("sign")
    @classmethod
    def _unit_sign(cls, value: int) -> int:
        if value not in (1, -1):
            raise ValueError("sign must be 1 or -1 (direction relative to the concept axis)")
        return value


class Concept(_Strict):
    label: str
    axis: str
    valence: Valence
    valence_basis: str
    status: AnnotationStatus = AnnotationStatus.AGENT_PROPOSED
    phenotypes: dict[str, Phenotype]


class Domain(_Strict):
    label: str
    concepts: dict[str, Concept]


class ClusterFile(_Strict):
    version: int = 0
    domains: dict[str, Domain] = Field(default_factory=dict)

    def concept(self, path: str | None) -> Concept | None:
        if not path or "." not in path:
            return None
        domain_key, concept_key = path.split(".", 1)
        domain = self.domains.get(domain_key)
        return domain.concepts.get(concept_key) if domain else None


class Provenance(_Strict):
    source: EvidenceSource
    locator: str
    quote: str = Field(min_length=10)
    confidence: Confidence


class ScoreEffect(_Strict):
    direction: Direction
    target: str
    coding: str | None = None


class EvaluationOutcome(_Strict):
    phenotype: str | None = None
    expected_metric_sign: int | None = None
    basis: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> EvaluationOutcome:
        if (self.phenotype is None) == (self.expected_metric_sign is None):
            raise ValueError("set exactly one of phenotype / expected_metric_sign")
        if self.expected_metric_sign is not None:
            if self.expected_metric_sign not in (1, -1):
                raise ValueError("expected_metric_sign must be 1 or -1")
            if not self.basis:
                raise ValueError("expected_metric_sign needs a basis")
        return self


class LeadVariantCheck(_Strict):
    status: LeadVariantStatus = LeadVariantStatus.NOT_RUN
    note: str | None = None


class ScoreAnnotation(_Strict):
    cluster: str | None = None
    phenotype: str | None = None
    measurement_kind: MeasurementKind | None = None
    score_effect: ScoreEffect | None = None
    polarity: int | None = None
    evaluation_outcome: EvaluationOutcome | None = None
    strata: dict[str, str] = Field(default_factory=dict)
    provenance: list[Provenance] = Field(default_factory=list)
    lead_variant_check: LeadVariantCheck = Field(default_factory=LeadVariantCheck)
    unannotated_reason: str | None = None
    status: AnnotationStatus = AnnotationStatus.AGENT_PROPOSED
    notes: str | None = None

    @model_validator(mode="after")
    def _complete(self) -> ScoreAnnotation:
        if self.score_effect is None:
            if not self.unannotated_reason:
                raise ValueError("score_effect is null: set unannotated_reason (never guess a direction)")
            return self
        missing = [
            name
            for name in ("cluster", "phenotype", "measurement_kind", "polarity")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"annotated score is missing: {', '.join(missing)}")
        if self.polarity not in (1, -1):
            raise ValueError("polarity must be 1 or -1")
        if not self.provenance:
            raise ValueError("annotated score needs at least one provenance entry with a verbatim quote")
        return self


class Paper(_Strict):
    pmid: str | int | None = None
    doi: str | None = None
    title: str | None = None


class EvidenceRead(_Strict):
    abstract: bool = False
    full_text: str | None = None
    supplement: list[str] = Field(default_factory=list)


class Template(_Strict):
    rule: str
    provenance: list[Provenance] = Field(min_length=1)


class PublicationFile(_Strict):
    pgp_id: str
    paper: Paper = Field(default_factory=Paper)
    evidence_read: EvidenceRead = Field(default_factory=EvidenceRead)
    template: Template | None = None
    scores: dict[str, ScoreAnnotation]
    exploration_findings: str | None = None


class Finding(BaseModel):
    level: Level
    check: str
    pgs_id: str
    message: str


class TraitDecision(StrEnum):
    PENDING = "pending"
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


class TraitStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    RESOLVED = "resolved"


class TraitScore(_Strict):
    decision: TraitDecision = TraitDecision.PENDING
    note: str | None = None


class TraitMatch(_Strict):
    terms: list[str] = Field(default_factory=list)
    efo_ids: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class TraitWorklist(_Strict):
    """One user question ("sort out longevity"). Grain: ``slug``; scores keyed by PGS ID."""

    slug: str
    query: str
    match: TraitMatch = Field(default_factory=TraitMatch)
    anchor: str | None = None
    status: TraitStatus = TraitStatus.IN_PROGRESS
    concepts: list[str] = Field(default_factory=list)
    scores: dict[str, TraitScore] = Field(default_factory=dict)
    summary: str | None = None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _queue_path(root: Path) -> Path:
    return root / "queue.csv"


def _progress_path(root: Path) -> Path:
    return root / "progress.txt"


def _publication_path(root: Path, pgp_id: str) -> Path:
    return root / "publications" / f"{pgp_id}.yaml"


def _lookup_dir(pgp_id: str) -> Path:
    return resolve_cache_dir() / "annotation_lookups" / pgp_id


def _read_queue(root: Path) -> dict[str, dict[str, str]]:
    path = _queue_path(root)
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["pgp_id"]: row for row in csv.DictReader(handle)}


def _write_queue(root: Path, rows: dict[str, dict[str, str]]) -> None:
    ordered = sorted(
        rows.values(),
        key=lambda r: (r["phase"] != "explore", int(r["priority"]), r["pgp_id"]),
    )
    path = _queue_path(root)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    tmp.replace(path)


def _require_row(rows: dict[str, dict[str, str]], pgp_id: str) -> dict[str, str]:
    if pgp_id not in rows:
        typer.echo(f"{pgp_id} is not in the queue; run `init` first.", err=True)
        raise typer.Exit(2)
    return rows[pgp_id]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _load_clusters(root: Path) -> ClusterFile:
    path = root / "clusters.yaml"
    return ClusterFile.model_validate(_load_yaml(path)) if path.exists() else ClusterFile()


def _load_exploration(root: Path) -> dict[str, list[str]]:
    """Map PGS ID -> exploration types it was picked for."""
    path = root / "exploration" / "set.yaml"
    if not path.exists():
        return {}
    by_id: dict[str, list[str]] = {}
    for type_name, ids in (_load_yaml(path).get("types") or {}).items():
        for pgs_id in ids:
            by_id.setdefault(pgs_id, []).append(type_name)
    return by_id


def _load_publication(root: Path, pgp_id: str) -> PublicationFile | None:
    path = _publication_path(root, pgp_id)
    return PublicationFile.model_validate(_load_yaml(path)) if path.exists() else None


def _all_publications(root: Path) -> dict[str, PublicationFile]:
    loaded: dict[str, PublicationFile] = {}
    for path in sorted((root / "publications").glob("*.yaml")):
        try:
            loaded[path.stem] = PublicationFile.model_validate(_load_yaml(path))
        except ValidationError:
            continue  # reported by `check` on that publication
    return loaded


def _annotation_counts(pub: PublicationFile | None) -> tuple[int, int]:
    if pub is None:
        return 0, 0
    annotated = sum(1 for a in pub.scores.values() if a.score_effect is not None)
    verified = sum(1 for a in pub.scores.values() if a.status == AnnotationStatus.HUMAN_VERIFIED)
    return annotated, verified


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"}))
    text = re.sub(r"\s+", " ", text)
    # Stripped citation tags leave "( 15 )"; close brackets so verbatim quotes still match.
    text = re.sub(r"([(\[])\s+", r"\1", re.sub(r"\s+([)\]])", r"\1", text))
    return text.strip().lower()


def _source_corpus(pgp_id: str) -> str | None:
    """Concatenated raw source text for quote verification (agent output excluded)."""
    lookup_dir = _lookup_dir(pgp_id)
    if not lookup_dir.exists():
        return None
    parts = [
        _normalize_text(path.read_text(errors="ignore"))
        for path in sorted(lookup_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES and not (path.name.startswith(_AGENT_OUTPUT_PREFIX) and path.suffix == ".json")
    ]
    return " ".join(parts) if parts else None


def _check_quotes(
    pgs_id: str, provenance: list[Provenance], corpus: str | None, catalog_text: str
) -> list[Finding]:
    findings: list[Finding] = []
    for entry in provenance:
        if entry.source == EvidenceSource.CATALOG:
            found = _normalize_text(entry.quote) in catalog_text
            findings.append(
                Finding(
                    level=Level.OK if found else Level.ERROR,
                    check="quote",
                    pgs_id=pgs_id,
                    message=f"{'found' if found else 'NOT found verbatim'} in catalog fields ({entry.locator})",
                )
            )
            continue
        if corpus is None:
            findings.append(
                Finding(level=Level.NOT_RUN, check="quote", pgs_id=pgs_id, message="no cached source text to verify against")
            )
            continue
        found = _normalize_text(entry.quote) in corpus
        findings.append(
            Finding(
                level=Level.OK if found else Level.ERROR,
                check="quote",
                pgs_id=pgs_id,
                message=f"{'found' if found else 'NOT found verbatim'} ({entry.source}, {entry.locator})",
            )
        )
    return findings


def _metric_sign(row: dict[str, object]) -> tuple[int | None, str]:
    """Sign of the best evaluation metric: +1 / -1, or None when uninformative."""
    for name, null_value in (("hr", 1.0), ("or", 1.0), ("beta", 0.0)):
        estimate = row.get(f"{name}_estimate")
        if estimate is None:
            continue
        lower, upper = row.get(f"{name}_ci_lower"), row.get(f"{name}_ci_upper")
        label = f"{name.upper()} {estimate:g}" + (f" [{lower:g}, {upper:g}]" if lower is not None and upper is not None else "")
        if lower is not None and upper is not None and lower <= null_value <= upper:
            return None, f"{label}: CI includes the null"
        if estimate == null_value:
            return None, f"{label}: estimate equals the null"
        return (1 if estimate > null_value else -1), label
    return None, "no HR/OR/beta (AUROC and C-index carry no direction)"


def _check_metric(pgs_id: str, ann: ScoreAnnotation, concept: Concept, best: dict[str, object] | None) -> Finding:
    if best is None:
        return Finding(level=Level.NOT_RUN, check="metric_sign", pgs_id=pgs_id, message="no catalog evaluation")
    observed, label = _metric_sign(best)
    evaluated = best.get("trait_reported") or "?"
    if observed is None:
        return Finding(level=Level.NOT_RUN, check="metric_sign", pgs_id=pgs_id, message=f"{label} vs '{evaluated}'")
    outcome = ann.evaluation_outcome
    if outcome is not None and outcome.expected_metric_sign is not None:
        expected = outcome.expected_metric_sign
    else:
        outcome_key = outcome.phenotype if outcome is not None else ann.phenotype
        outcome_phenotype = concept.phenotypes.get(outcome_key or "")
        if outcome_phenotype is None:
            return Finding(
                level=Level.ERROR,
                check="metric_sign",
                pgs_id=pgs_id,
                message=f"evaluation_outcome phenotype '{outcome_key}' is not in the concept",
            )
        expected = (ann.polarity or 0) * outcome_phenotype.sign
    agrees = expected == observed
    return Finding(
        level=Level.OK if agrees else Level.WARN,
        check="metric_sign",
        pgs_id=pgs_id,
        message=(
            f"{label} vs '{evaluated}' {'agrees' if agrees else 'DISAGREES'} (expected sign {expected:+d})"
            + ("" if agrees else "; set evaluation_outcome if the metric targets another outcome, else re-read")
        ),
    )


def _reference_scores(catalog: PRSCatalog, pgs_id: str, cache: dict[str, pl.DataFrame | None]) -> pl.DataFrame | None:
    if pgs_id not in cache:
        try:
            cache[pgs_id] = catalog.reference_individual_scores(pgs_id, superpopulation="EUR").collect()
        except FileNotFoundError:
            cache[pgs_id] = None
    return cache[pgs_id]


def _check_correlations(
    catalog: PRSCatalog,
    pgp_id: str,
    pub: PublicationFile,
    publications: dict[str, PublicationFile],
    threshold: float,
) -> list[Finding]:
    """Within one concept, aligned 1000G EUR correlation should not be clearly negative."""
    peers: dict[str, list[tuple[str, int]]] = {}
    for other in {**publications, pgp_id: pub}.values():
        for pgs_id, ann in other.scores.items():
            if ann.score_effect is not None and ann.status != AnnotationStatus.REJECTED and ann.cluster:
                peers.setdefault(ann.cluster, []).append((pgs_id, ann.polarity or 0))
    cache: dict[str, pl.DataFrame | None] = {}
    findings: list[Finding] = []
    for pgs_id, ann in pub.scores.items():
        if ann.score_effect is None or not ann.cluster:
            continue
        others = [(o, p) for o, p in peers.get(ann.cluster, []) if o != pgs_id]
        if not others:
            findings.append(Finding(level=Level.NOT_RUN, check="corr_1000g", pgs_id=pgs_id, message="no annotated peer in this concept yet"))
            continue
        own = _reference_scores(catalog, pgs_id, cache)
        if own is None:
            findings.append(Finding(level=Level.NOT_RUN, check="corr_1000g", pgs_id=pgs_id, message="no cached 1000G per-individual scores"))
            continue
        for other_id, other_polarity in others:
            other_df = _reference_scores(catalog, other_id, cache)
            if other_df is None:
                findings.append(
                    Finding(level=Level.NOT_RUN, check="corr_1000g", pgs_id=pgs_id, message=f"vs {other_id}: no cached 1000G scores")
                )
                continue
            joined = own.join(other_df.select("iid", pl.col("prs_score").alias("peer")), on="iid", how="inner")
            if joined.height < CORR_MIN_SAMPLES:
                findings.append(
                    Finding(level=Level.NOT_RUN, check="corr_1000g", pgs_id=pgs_id, message=f"vs {other_id}: only {joined.height} shared samples")
                )
                continue
            r = joined.select(pl.corr("prs_score", "peer")).item()
            if r is None:
                findings.append(
                    Finding(level=Level.NOT_RUN, check="corr_1000g", pgs_id=pgs_id, message=f"vs {other_id}: zero variance")
                )
                continue
            aligned = r * (ann.polarity or 0) * other_polarity
            findings.append(
                Finding(
                    level=Level.WARN if aligned < -threshold else Level.OK,
                    check="corr_1000g",
                    pgs_id=pgs_id,
                    message=f"vs {other_id}: r={r:+.2f}, aligned={aligned:+.2f} (EUR n={joined.height})"
                    + (" — opposite after alignment; polarity or phenotype may be wrong" if aligned < -threshold else ""),
                )
            )
    return findings


def run_checks(root: Path, pgp_id: str, threshold: float) -> list[Finding]:
    try:
        pub = _load_publication(root, pgp_id)
    except ValidationError as exc:
        return [Finding(level=Level.ERROR, check="schema", pgs_id="-", message=str(exc).replace("\n", " | "))]
    if pub is None:
        return [Finding(level=Level.ERROR, check="schema", pgs_id="-", message=f"{_publication_path(root, pgp_id)} does not exist")]
    try:
        clusters = _load_clusters(root)
    except ValidationError as exc:
        return [Finding(level=Level.ERROR, check="clusters", pgs_id="-", message=str(exc).replace("\n", " | "))]

    catalog = PRSCatalog()
    paper_scores = catalog.scores(include_excluded=True).filter(pl.col("pgp_id") == pgp_id).collect()
    paper_ids = set(paper_scores["pgs_id"])
    best_rows = {
        row["pgs_id"]: row
        for row in catalog.best_performance().filter(pl.col("pgs_id").is_in(list(pub.scores))).collect().to_dicts()
    }
    catalog_text: dict[str, str] = {}
    text_cols = ("name", "trait_reported", "trait_efo", "trait_efo_id")
    for row in paper_scores.select(["pgs_id", *text_cols]).to_dicts():
        catalog_text[row["pgs_id"]] = " | ".join(str(row[c] or "") for c in text_cols)
    for row in catalog.performance().filter(pl.col("pgs_id").is_in(list(pub.scores))).select(
        "pgs_id", "trait_reported", "covariates"
    ).collect().to_dicts():
        catalog_text[row["pgs_id"]] = catalog_text.get(row["pgs_id"], "") + f" | {row['trait_reported'] or ''} | {row['covariates'] or ''}"
    catalog_text = {k: _normalize_text(v) for k, v in catalog_text.items()}
    all_catalog_text = " ".join(catalog_text.values())
    corpus = _source_corpus(pgp_id)
    findings: list[Finding] = []
    if pub.pgp_id != pgp_id:
        findings.append(Finding(level=Level.ERROR, check="schema", pgs_id="-", message=f"pgp_id field is {pub.pgp_id}"))
    if pub.template is not None:
        findings.extend(_check_quotes("template", pub.template.provenance, corpus, all_catalog_text))

    for pgs_id, ann in pub.scores.items():
        if pgs_id not in paper_ids:
            findings.append(Finding(level=Level.ERROR, check="membership", pgs_id=pgs_id, message=f"not a score of {pgp_id}"))
        if ann.score_effect is None:
            findings.append(Finding(level=Level.NOT_RUN, check="direction", pgs_id=pgs_id, message=f"unannotated: {ann.unannotated_reason}"))
            continue
        concept = clusters.concept(ann.cluster)
        if concept is None:
            findings.append(Finding(level=Level.ERROR, check="cluster", pgs_id=pgs_id, message=f"unknown cluster '{ann.cluster}'"))
            continue
        phenotype = concept.phenotypes.get(ann.phenotype or "")
        if phenotype is None:
            findings.append(
                Finding(level=Level.ERROR, check="cluster", pgs_id=pgs_id, message=f"phenotype '{ann.phenotype}' not in {ann.cluster}")
            )
            continue
        direction_sign = 1 if ann.score_effect.direction == Direction.INCREASES else -1
        expected_polarity = phenotype.sign * direction_sign
        findings.append(
            Finding(
                level=Level.OK if expected_polarity == ann.polarity else Level.ERROR,
                check="polarity",
                pgs_id=pgs_id,
                message=f"{ann.score_effect.direction} {ann.phenotype} (sign {phenotype.sign:+d}) → polarity {expected_polarity:+d}, declared {ann.polarity:+d}",
            )
        )
        findings.extend(_check_quotes(pgs_id, ann.provenance, corpus, catalog_text.get(pgs_id, "")))
        findings.append(_check_metric(pgs_id, ann, concept, best_rows.get(pgs_id)))
        lead = ann.lead_variant_check
        lead_level = {
            LeadVariantStatus.AGREE: Level.OK,
            LeadVariantStatus.DISAGREE: Level.WARN,
            LeadVariantStatus.MIXED: Level.WARN,
            LeadVariantStatus.NOT_RUN: Level.NOT_RUN,
        }[lead.status]
        findings.append(Finding(level=lead_level, check="lead_variant", pgs_id=pgs_id, message=f"{lead.status}: {lead.note or ''}".strip()))

    findings.extend(_check_correlations(catalog, pgp_id, pub, _all_publications(root), threshold))
    return findings


def _summarize(findings: list[Finding]) -> dict[str, int]:
    return {level.value: sum(1 for f in findings if f.level == level) for level in Level}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(root: RootOpt = DEFAULT_ROOT) -> None:
    """Build or refresh queue.csv from the catalog. Keeps existing statuses and attempts."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "publications").mkdir(exist_ok=True)
    exploration = _load_exploration(root)
    scores = PRSCatalog().scores(include_excluded=True).select("pgs_id", "pgp_id", "trait_reported").collect()
    existing = _read_queue(root)
    rows: dict[str, dict[str, str]] = {}
    for group in scores.partition_by("pgp_id"):
        pgp_id = group["pgp_id"][0]
        ids = group["pgs_id"].to_list()
        targets = sorted(i for i in ids if i in exploration)
        if targets:
            phase, priority = "explore", 0
        elif len(ids) > 100:
            phase, priority = "main", 4
        elif any(_MIXED_POLARITY_RE.search(t or "") for t in group["trait_reported"].to_list()):
            phase, priority = "main", 1
        elif len(ids) <= 5:
            phase, priority = "main", 2
        else:
            phase, priority = "main", 3
        previous = existing.get(pgp_id, {})
        rows[pgp_id] = {
            "pgp_id": pgp_id,
            "phase": phase,
            "priority": str(priority),
            "n_scores": str(len(ids)),
            "target_pgs_ids": ";".join(targets),
            "status": previous.get("status", QueueStatus.PENDING.value),
            "attempts": previous.get("attempts", "0"),
            "n_annotated": previous.get("n_annotated", "0"),
            "n_verified": previous.get("n_verified", "0"),
            "last_error": previous.get("last_error", ""),
            "claimed_at": previous.get("claimed_at", ""),
            "updated_at": previous.get("updated_at", ""),
        }
    _write_queue(root, rows)
    missing = sorted(set(exploration) - set(scores["pgs_id"]))
    n_explore = sum(1 for r in rows.values() if r["phase"] == "explore")
    typer.echo(f"queue: {len(rows)} publications, {scores.height} scores; explore phase: {n_explore} publications")
    if missing:
        typer.echo(f"exploration IDs not in the catalog: {', '.join(missing)}", err=True)


@app.command("next")
def next_items(
    n: Annotated[int, typer.Option("--n")] = 10,
    phase: Annotated[str | None, typer.Option("--phase", help="explore | main")] = None,
    stale_hours: Annotated[float, typer.Option("--stale-hours")] = STALE_HOURS,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Print the next publications to work on (pending, retryable, or interrupted) as JSON."""
    now = datetime.now(UTC)
    picked: list[dict[str, str]] = []
    for row in _read_queue(root).values():
        if phase and row["phase"] != phase:
            continue
        reason = ""
        if row["status"] == QueueStatus.PENDING:
            reason = "pending"
        elif row["status"] == QueueStatus.CHECKS_FAILED and int(row["attempts"]) < MAX_ATTEMPTS:
            reason = f"retry ({row['last_error']})"
        elif row["status"] == QueueStatus.IN_PROGRESS and row["claimed_at"]:
            age = (now - datetime.fromisoformat(row["claimed_at"])).total_seconds() / 3600
            if age >= stale_hours:
                reason = f"interrupted {age:.1f} h ago"
        if reason:
            picked.append({**row, "reason": reason})
    picked.sort(key=lambda r: (r["phase"] != "explore", int(r["priority"]), r["pgp_id"]))
    typer.echo(json.dumps(picked[:n], indent=1))


@app.command()
def claim(pgp_id: str, force: bool = False, root: RootOpt = DEFAULT_ROOT) -> None:
    """Mark a publication in_progress and count the attempt."""
    rows = _read_queue(root)
    row = _require_row(rows, pgp_id)
    if row["status"] == QueueStatus.HUMAN_VERIFIED and not force:
        typer.echo(f"{pgp_id} is human_verified; pass --force to reopen it.", err=True)
        raise typer.Exit(2)
    now = _now()
    row.update(status=QueueStatus.IN_PROGRESS.value, attempts=str(int(row["attempts"]) + 1), claimed_at=now, updated_at=now)
    _write_queue(root, rows)
    _lookup_dir(pgp_id).mkdir(parents=True, exist_ok=True)
    typer.echo(json.dumps(row, indent=1))


@app.command()
def context(
    pgp_id: str,
    pgs_ids: Annotated[str | None, typer.Option("--pgs-ids", help="Comma-separated subset")] = None,
    top_variants: Annotated[int, typer.Option("--top-variants", help="Top-|weight| variants per score")] = 0,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Print everything the orchestrator and lookup agent need for one publication, as JSON."""
    catalog = PRSCatalog()
    row = _read_queue(root).get(pgp_id, {})
    if pgs_ids:
        wanted = [p.strip() for p in pgs_ids.split(",") if p.strip()]
    elif row.get("target_pgs_ids"):
        wanted = row["target_pgs_ids"].split(";")
    else:
        wanted = []
    scores_lf = catalog.scores(include_excluded=True).filter(pl.col("pgp_id") == pgp_id)
    if wanted:
        scores_lf = scores_lf.filter(pl.col("pgs_id").is_in(wanted))
    scores = scores_lf.select(
        "pgs_id", "name", "trait_reported", "trait_efo", "trait_efo_id", "n_variants", "weight_type", "genome_build"
    ).collect()
    ids = scores["pgs_id"].to_list()
    metric_cols = [
        "pgs_id", "trait_reported", "covariates", "ancestry_broad", "n_individuals", "n_cases", "n_controls",
        "hr_estimate", "hr_ci_lower", "hr_ci_upper", "or_estimate", "or_ci_lower", "or_ci_upper",
        "beta_estimate", "beta_ci_lower", "beta_ci_upper", "auroc_estimate", "cindex_estimate",
    ]
    performance = catalog.performance().filter(pl.col("pgs_id").is_in(ids)).select(metric_cols).collect()
    best = catalog.best_performance().filter(pl.col("pgs_id").is_in(ids)).select(metric_cols).collect()
    publications = catalog.publications(pgp_id)
    paper = publications.collect().to_dicts() if publications is not None else []

    variants: dict[str, object] = {}
    if top_variants:
        if len(ids) > TOP_VARIANTS_MAX_SCORES:
            typer.echo(f"--top-variants needs --pgs-ids with <= {TOP_VARIANTS_MAX_SCORES} scores", err=True)
            raise typer.Exit(2)
        for pgs_id in ids:
            path = ensure_scoring_file(pgs_id, catalog.cache_dir / "scores", "GRCh38")
            lf = pl.scan_parquet(path)
            if "effect_weight" not in lf.collect_schema().names():
                variants[pgs_id] = "per-dosage weights (no single effect_weight); check lead variants by hand"
                continue
            variants[pgs_id] = (
                lf.sort(pl.col("effect_weight").abs(), descending=True)
                .head(top_variants)
                .select([c for c in ("rsID", "hm_rsID", "hm_chr", "hm_pos", "effect_allele", "other_allele", "effect_weight") if c in lf.collect_schema().names()])
                .collect()
                .to_dicts()
            )

    clusters = _load_clusters(root)
    cluster_index = [
        {
            "cluster": f"{domain_key}.{concept_key}",
            "axis": concept.axis,
            "valence": concept.valence,
            "phenotypes": {key: ph.sign for key, ph in concept.phenotypes.items()},
        }
        for domain_key, domain in clusters.domains.items()
        for concept_key, concept in domain.concepts.items()
    ]
    existing = _publication_path(root, pgp_id)
    payload = {
        "pgp_id": pgp_id,
        "queue": row,
        "paper": paper,
        "scores": scores.to_dicts(),
        "best_performance": best.to_dicts(),
        "all_evaluations": performance.to_dicts(),
        "top_variants": variants,
        "clusters": cluster_index,
        "lookup_dir": str(_lookup_dir(pgp_id)),
        "annotation_file": str(existing),
        "existing_annotation": _load_yaml(existing) if existing.exists() else None,
    }
    typer.echo(json.dumps(payload, indent=1, default=str))


@app.command()
def check(
    pgp_id: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    corr_threshold: Annotated[float, typer.Option("--corr-threshold")] = CORR_THRESHOLD,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Run deterministic checks on one publication. Exit 1 on any ERROR."""
    findings = run_checks(root, pgp_id, corr_threshold)
    if as_json:
        typer.echo(json.dumps([f.model_dump() for f in findings], indent=1))
    else:
        for f in findings:
            typer.echo(f"{f.level.value:<8} {f.check:<13} {f.pgs_id:<10} {f.message}")
        counts = _summarize(findings)
        typer.echo(f"-- {pgp_id}: {counts['ERROR']} error / {counts['WARN']} warn / {counts['NOT_RUN']} not_run / {counts['OK']} ok")
    if any(f.level == Level.ERROR for f in findings):
        raise typer.Exit(1)


def _check_coverage(root: Path, pgp_id: str, row: dict[str, str]) -> list[Finding]:
    """Every target score needs an entry: annotated, or null with an unannotated_reason."""
    try:
        pub = _load_publication(root, pgp_id)
    except ValidationError:
        return []  # the schema error is already reported by run_checks
    if row["target_pgs_ids"]:
        targets = set(row["target_pgs_ids"].split(";"))
    else:
        targets = set(
            PRSCatalog().scores(include_excluded=True).filter(pl.col("pgp_id") == pgp_id).select("pgs_id").collect().to_series()
        )
    missing = sorted(targets - set(pub.scores if pub else {}))
    if not missing:
        return []
    shown = ", ".join(missing[:10]) + (f" … (+{len(missing) - 10})" if len(missing) > 10 else "")
    return [
        Finding(
            level=Level.ERROR,
            check="coverage",
            pgs_id="-",
            message=f"{len(missing)} target score(s) have no entry: {shown}; annotate or set score_effect: null with a reason",
        )
    ]


def _progress_line(rows: dict[str, dict[str, str]], row: dict[str, str], counts: dict[str, int], lookups: int, minutes: float) -> str:
    done_states = {QueueStatus.CHECKS_PASSED.value, QueueStatus.HUMAN_VERIFIED.value}
    done = sum(1 for r in rows.values() if r["status"] in done_states)
    total = len(rows)
    explore = [r for r in rows.values() if r["phase"] == "explore"]
    explore_done = sum(1 for r in explore if r["status"] in done_states)
    n_scores = sum(int(r["n_scores"]) for r in rows.values())
    annotated = sum(int(r["n_annotated"]) for r in rows.values())
    verified = sum(int(r["n_verified"]) for r in rows.values())
    return (
        f"{_now()} Annotation: {done}/{total} PGP ({100 * done / max(total, 1):.1f}%), "
        f"explore {explore_done}/{len(explore)}, scores {annotated}/{n_scores} ({100 * annotated / max(n_scores, 1):.1f}%) annotated, "
        f"{verified} verified | {row['pgp_id']} -> {row['status']}, {row['n_annotated']} scores, "
        f"{counts['ERROR']} error / {counts['WARN']} warn / {counts['NOT_RUN']} not_run, "
        f"lookups {lookups}, {minutes:.1f} min, attempt {row['attempts']}"
    )


@app.command()
def finish(
    pgp_id: str,
    lookups: Annotated[int, typer.Option("--lookups", help="Lookup agents dispatched")] = 0,
    blocked: Annotated[str | None, typer.Option("--blocked", help="Reason; marks the publication blocked")] = None,
    corr_threshold: Annotated[float, typer.Option("--corr-threshold")] = CORR_THRESHOLD,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Run checks, set the queue status, and append one progress line."""
    rows = _read_queue(root)
    row = _require_row(rows, pgp_id)
    findings: list[Finding] = [] if blocked else run_checks(root, pgp_id, corr_threshold)
    if not blocked:
        findings.extend(_check_coverage(root, pgp_id, row))
    errors = [f for f in findings if f.level == Level.ERROR]
    if blocked:
        status, last_error = QueueStatus.BLOCKED, blocked
    elif errors:
        exhausted = int(row["attempts"]) >= MAX_ATTEMPTS
        status = QueueStatus.BLOCKED if exhausted else QueueStatus.CHECKS_FAILED
        last_error = f"{errors[0].check} {errors[0].pgs_id}: {errors[0].message}"[:300]
    else:
        status, last_error = QueueStatus.CHECKS_PASSED, ""
    try:
        annotated, verified = _annotation_counts(_load_publication(root, pgp_id))
    except ValidationError:
        annotated, verified = 0, 0
    if status == QueueStatus.CHECKS_PASSED and annotated and verified == annotated:
        status = QueueStatus.HUMAN_VERIFIED
    claimed = row["claimed_at"]
    minutes = (datetime.now(UTC) - datetime.fromisoformat(claimed)).total_seconds() / 60 if claimed else 0.0
    row.update(
        status=status.value,
        last_error=last_error,
        n_annotated=str(annotated),
        n_verified=str(verified),
        updated_at=_now(),
    )
    _write_queue(root, rows)
    line = _progress_line(rows, row, _summarize(findings), lookups, minutes)
    with _progress_path(root).open("a") as handle:
        handle.write(line + "\n")
    typer.echo(line)
    for f in findings:
        if f.level in (Level.ERROR, Level.WARN):
            typer.echo(f"  {f.level.value:<6} {f.check:<13} {f.pgs_id:<10} {f.message}")


@app.command()
def log(
    pgp_id: str,
    message: Annotated[str, typer.Option("--message", help="e.g. 'block 3/39 (PGS001100..PGS001119)'")],
    lookups: Annotated[int, typer.Option("--lookups")] = 0,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Append a progress line for work inside a publication (biobank blocks) without changing its status."""
    rows = _read_queue(root)
    row = _require_row(rows, pgp_id)
    try:
        annotated, verified = _annotation_counts(_load_publication(root, pgp_id))
    except ValidationError:
        annotated, verified = int(row["n_annotated"]), int(row["n_verified"])
    row.update(n_annotated=str(annotated), n_verified=str(verified), updated_at=_now())
    _write_queue(root, rows)
    claimed = row["claimed_at"]
    minutes = (datetime.now(UTC) - datetime.fromisoformat(claimed)).total_seconds() / 60 if claimed else 0.0
    line = _progress_line(rows, row, {level.value: 0 for level in Level}, lookups, minutes) + f" | {message}"
    with _progress_path(root).open("a") as handle:
        handle.write(line + "\n")
    typer.echo(line)


@app.command()
def status(root: RootOpt = DEFAULT_ROOT) -> None:
    """Refresh counts from the YAML files and print progress, backlog, and exploration coverage."""
    rows = _read_queue(root)
    if not rows:
        typer.echo("no queue yet; run `init`.")
        raise typer.Exit(1)
    publications = _all_publications(root)
    for pgp_id, row in rows.items():
        annotated, verified = _annotation_counts(publications.get(pgp_id))
        row.update(n_annotated=str(annotated), n_verified=str(verified))
        if row["status"] == QueueStatus.CHECKS_PASSED and annotated and verified == annotated:
            row["status"] = QueueStatus.HUMAN_VERIFIED.value
    _write_queue(root, rows)

    by_status: dict[str, int] = {}
    for row in rows.values():
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    n_scores = sum(int(r["n_scores"]) for r in rows.values())
    annotated = sum(int(r["n_annotated"]) for r in rows.values())
    verified = sum(int(r["n_verified"]) for r in rows.values())
    typer.echo("publications by status: " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
    typer.echo(f"scores annotated: {annotated}/{n_scores} ({100 * annotated / max(n_scores, 1):.1f}%), verified {verified}, review backlog {annotated - verified}")

    exploration = _load_exploration(root)
    if exploration:
        annotations = {
            pgs_id: ann for pub in publications.values() for pgs_id, ann in pub.scores.items()
        }
        per_type: dict[str, list[str]] = {}
        for pgs_id, types in exploration.items():
            for type_name in types:
                per_type.setdefault(type_name, []).append(pgs_id)
        typer.echo("exploration coverage (annotated / verified / total):")
        for type_name, ids in per_type.items():
            n_ann = sum(1 for i in ids if i in annotations and annotations[i].score_effect is not None)
            n_ver = sum(1 for i in ids if i in annotations and annotations[i].status == AnnotationStatus.HUMAN_VERIFIED)
            typer.echo(f"  {type_name:<22} {n_ann:>2} / {n_ver:>2} / {len(ids):>2}")

    blocked_rows = [r for r in rows.values() if r["status"] in (QueueStatus.BLOCKED, QueueStatus.CHECKS_FAILED)]
    for row in blocked_rows[:20]:
        typer.echo(f"  {row['status']:<14} {row['pgp_id']} (attempt {row['attempts']}): {row['last_error']}")
    progress = _progress_path(root)
    if progress.exists():
        tail = progress.read_text().splitlines()[-5:]
        typer.echo("recent progress:")
        for line in tail:
            typer.echo(f"  {line}")


# ---------------------------------------------------------------------------
# Build / publish
# ---------------------------------------------------------------------------

DEFAULT_BUILD_DIR = Path("data/output/curation")
_CONFIDENCE_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}


class CurationBuildError(ValueError):
    """The curated files are inconsistent; nothing may be published."""


def _git_commit(root: Path) -> str | None:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _higher_percentile_means(ann: ScoreAnnotation) -> str | None:
    if ann.score_effect is None:
        return None
    verb = "more" if ann.score_effect.direction == Direction.INCREASES else "less"
    return f"higher score = {verb} {ann.score_effect.target}"


def build_tables(root: Path, verified_only: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Compile ``curation/`` into ``score_annotations`` and ``trait_clusters`` tables.

    Every publication file must parse and every annotated entry must point at an existing
    concept + phenotype with a consistent polarity; otherwise ``CurationBuildError``.
    ``rejected`` entries are never exported; ``verified_only`` also drops ``agent_proposed``.
    """
    clusters = _load_clusters(root)
    problems: list[str] = []
    rows: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    usage: dict[tuple[str, str], int] = {}
    for path in sorted((root / "publications").glob("*.yaml")):
        try:
            pub = PublicationFile.model_validate(_load_yaml(path))
        except ValidationError as exc:
            problems.append(f"{path.name}: {str(exc).splitlines()[0]}")
            continue
        for pgs_id, ann in pub.scores.items():
            if pgs_id in seen:
                problems.append(f"{pgs_id} appears in {seen[pgs_id]} and {path.stem}")
                continue
            seen[pgs_id] = path.stem
            if ann.status == AnnotationStatus.REJECTED:
                continue
            if verified_only and ann.status != AnnotationStatus.HUMAN_VERIFIED:
                continue
            concept = clusters.concept(ann.cluster) if ann.score_effect is not None else None
            phenotype = concept.phenotypes.get(ann.phenotype or "") if concept else None
            if ann.score_effect is not None:
                if concept is None or phenotype is None:
                    problems.append(f"{path.stem}/{pgs_id}: unknown cluster/phenotype {ann.cluster}.{ann.phenotype}")
                    continue
                direction_sign = 1 if ann.score_effect.direction == Direction.INCREASES else -1
                if phenotype.sign * direction_sign != ann.polarity:
                    problems.append(f"{path.stem}/{pgs_id}: polarity {ann.polarity:+d} ≠ phenotype sign × direction")
                    continue
                usage[(ann.cluster or "", ann.phenotype or "")] = usage.get((ann.cluster or "", ann.phenotype or ""), 0) + 1
            outcome = ann.evaluation_outcome
            rows.append(
                {
                    "pgs_id": pgs_id,
                    "pgp_id": path.stem,
                    "annotated": ann.score_effect is not None,
                    "annotation_status": ann.status.value,
                    "cluster": ann.cluster,
                    "domain": ann.cluster.split(".", 1)[0] if ann.cluster else None,
                    "concept_label": concept.label if concept else None,
                    "axis": concept.axis if concept else None,
                    "valence": concept.valence.value if concept else None,
                    "phenotype": ann.phenotype,
                    "phenotype_label": phenotype.label if phenotype else None,
                    "phenotype_sign": phenotype.sign if phenotype else None,
                    "measurement_kind": ann.measurement_kind.value if ann.measurement_kind else None,
                    "direction": ann.score_effect.direction.value if ann.score_effect else None,
                    "target": ann.score_effect.target if ann.score_effect else None,
                    "coding": ann.score_effect.coding if ann.score_effect else None,
                    "polarity": ann.polarity,
                    "higher_percentile_means": _higher_percentile_means(ann),
                    "evaluation_outcome_phenotype": outcome.phenotype if outcome else None,
                    "expected_metric_sign": outcome.expected_metric_sign if outcome else None,
                    "strata_json": json.dumps(ann.strata, sort_keys=True),
                    "confidence": max((p.confidence for p in ann.provenance), key=_CONFIDENCE_RANK.get).value
                    if ann.provenance
                    else None,
                    "provenance_json": json.dumps([p.model_dump(mode="json") for p in ann.provenance], ensure_ascii=False),
                    "lead_variant_status": ann.lead_variant_check.status.value,
                    "lead_variant_note": ann.lead_variant_check.note,
                    "unannotated_reason": ann.unannotated_reason,
                    "notes": ann.notes,
                    "paper_pmid": str(pub.paper.pmid) if pub.paper.pmid is not None else None,
                    "paper_doi": pub.paper.doi,
                }
            )
    if problems:
        raise CurationBuildError("curation files are inconsistent:\n  " + "\n  ".join(problems))

    cluster_rows = [
        {
            "cluster": f"{domain_key}.{concept_key}",
            "domain": domain_key,
            "domain_label": domain.label,
            "concept": concept_key,
            "concept_label": concept.label,
            "axis": concept.axis,
            "valence": concept.valence.value,
            "valence_basis": concept.valence_basis,
            "concept_status": concept.status.value,
            "phenotype": phenotype_key,
            "phenotype_label": phenotype.label,
            "phenotype_sign": phenotype.sign,
            "n_scores": usage.get((f"{domain_key}.{concept_key}", phenotype_key), 0),
        }
        for domain_key, domain in clusters.domains.items()
        for concept_key, concept in domain.concepts.items()
        for phenotype_key, phenotype in concept.phenotypes.items()
    ]
    annotation_schema = {
        "pgs_id": pl.Utf8, "pgp_id": pl.Utf8, "annotated": pl.Boolean, "annotation_status": pl.Utf8,
        "cluster": pl.Utf8, "domain": pl.Utf8, "concept_label": pl.Utf8, "axis": pl.Utf8, "valence": pl.Utf8,
        "phenotype": pl.Utf8, "phenotype_label": pl.Utf8, "phenotype_sign": pl.Int8,
        "measurement_kind": pl.Utf8, "direction": pl.Utf8, "target": pl.Utf8, "coding": pl.Utf8,
        "polarity": pl.Int8, "higher_percentile_means": pl.Utf8, "evaluation_outcome_phenotype": pl.Utf8,
        "expected_metric_sign": pl.Int8, "strata_json": pl.Utf8, "confidence": pl.Utf8,
        "provenance_json": pl.Utf8, "lead_variant_status": pl.Utf8, "lead_variant_note": pl.Utf8,
        "unannotated_reason": pl.Utf8, "notes": pl.Utf8, "paper_pmid": pl.Utf8, "paper_doi": pl.Utf8,
    }
    cluster_schema = {
        "cluster": pl.Utf8, "domain": pl.Utf8, "domain_label": pl.Utf8, "concept": pl.Utf8,
        "concept_label": pl.Utf8, "axis": pl.Utf8, "valence": pl.Utf8, "valence_basis": pl.Utf8,
        "concept_status": pl.Utf8, "phenotype": pl.Utf8, "phenotype_label": pl.Utf8,
        "phenotype_sign": pl.Int8, "n_scores": pl.Int32,
    }
    annotations = pl.DataFrame(rows, schema=annotation_schema).sort("pgs_id")
    cluster_df = pl.DataFrame(cluster_rows, schema=cluster_schema).sort("cluster", "phenotype")
    manifest: dict[str, object] = {
        "built_at": _now(),
        "source_commit": _git_commit(root),
        "verified_only": verified_only,
        "n_scores": annotations.height,
        "n_annotated": int(annotations["annotated"].sum()),
        "n_verified": int((annotations["annotation_status"] == AnnotationStatus.HUMAN_VERIFIED.value).sum()),
        "n_publications": annotations["pgp_id"].n_unique(),
        "n_concepts": cluster_df["cluster"].n_unique(),
        "n_phenotypes": cluster_df.height,
    }
    return annotations, cluster_df, manifest


def write_tables(root: Path, out_dir: Path, verified_only: bool = False) -> dict[str, object]:
    """Build and write the three curation files; the manifest records their SHA256."""
    import hashlib

    annotations, cluster_df, manifest = build_tables(root, verified_only)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"score_annotations.parquet": annotations, "trait_clusters.parquet": cluster_df}
    hashes: dict[str, str] = {}
    for name, frame in files.items():
        tmp = out_dir / f"{name}.tmp"
        frame.write_parquet(tmp, compression="zstd")
        tmp.replace(out_dir / name)
        hashes[name] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
    manifest["sha256"] = hashes
    (out_dir / "curation_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


@app.command()
def build(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = DEFAULT_BUILD_DIR,
    verified_only: Annotated[bool, typer.Option("--verified-only", help="Export human_verified entries only")] = False,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Compile curation/ into score_annotations.parquet, trait_clusters.parquet and a manifest."""
    try:
        manifest = write_tables(root, out, verified_only)
    except CurationBuildError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"built {out}: {manifest['n_scores']} scores ({manifest['n_annotated']} annotated, "
        f"{manifest['n_verified']} verified) from {manifest['n_publications']} papers; "
        f"{manifest['n_concepts']} concepts / {manifest['n_phenotypes']} phenotypes"
    )


_CLUSTER_DEFINITION_COLS = ("axis", "valence", "phenotype_sign")


def merge_with_published(out_dir: Path, published_dir: Path) -> dict[str, object]:
    """Merge a local build in ``out_dir`` with previously published tables in ``published_dir``.

    Published rows absent locally are kept (a partial clone never shrinks the release). On
    overlap the local row wins, except a published ``human_verified`` row is never replaced by
    an unreviewed one. A cluster phenotype defined differently on both sides is an error.
    Rewrites the files in ``out_dir`` and returns the updated manifest.
    """
    import hashlib

    local_ann = pl.read_parquet(out_dir / "score_annotations.parquet")
    local_clu = pl.read_parquet(out_dir / "trait_clusters.parquet")
    manifest = json.loads((out_dir / "curation_manifest.json").read_text())
    remote_ann_path = published_dir / "score_annotations.parquet"
    remote_clu_path = published_dir / "trait_clusters.parquet"
    kept_remote = protected = 0
    if remote_ann_path.exists():
        remote_ann = pl.read_parquet(remote_ann_path).select(local_ann.columns).cast(local_ann.schema)
        verified = remote_ann.filter(pl.col("annotation_status") == AnnotationStatus.HUMAN_VERIFIED.value)
        protect_ids = set(verified["pgs_id"]) & set(
            local_ann.filter(pl.col("annotation_status") != AnnotationStatus.HUMAN_VERIFIED.value)["pgs_id"]
        )
        protected = len(protect_ids)
        local_ids = set(local_ann["pgs_id"]) - protect_ids
        remote_keep = remote_ann.filter(~pl.col("pgs_id").is_in(list(local_ids)))
        kept_remote = remote_keep.height - protected
        local_ann = pl.concat([local_ann.filter(pl.col("pgs_id").is_in(list(local_ids))), remote_keep]).sort("pgs_id")
    if remote_clu_path.exists():
        remote_clu = pl.read_parquet(remote_clu_path).select(local_clu.columns).cast(local_clu.schema)
        both = local_clu.join(remote_clu, on=["cluster", "phenotype"], suffix="_remote")
        clashes = [
            f"{row['cluster']}.{row['phenotype']}: {col} {row[col]!r} here vs {row[col + '_remote']!r} published"
            for row in both.to_dicts()
            for col in _CLUSTER_DEFINITION_COLS
            if row[col] != row[col + "_remote"]
        ]
        if clashes:
            raise CurationBuildError("cluster definitions differ from the published release:\n  " + "\n  ".join(clashes))
        local_clu = pl.concat(
            [local_clu, remote_clu.join(local_clu, on=["cluster", "phenotype"], how="anti")]
        )
    usage = local_ann.filter(pl.col("annotated")).group_by("cluster", "phenotype").len()
    local_clu = (
        local_clu.drop("n_scores")
        .join(usage, on=["cluster", "phenotype"], how="left")
        .with_columns(pl.col("len").fill_null(0).cast(pl.Int32).alias("n_scores"))
        .drop("len")
        .select(local_clu.columns)
        .sort("cluster", "phenotype")
    )
    hashes: dict[str, str] = {}
    for name, frame in (("score_annotations.parquet", local_ann), ("trait_clusters.parquet", local_clu)):
        tmp = out_dir / f"{name}.tmp"
        frame.write_parquet(tmp, compression="zstd")
        tmp.replace(out_dir / name)
        hashes[name] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
    manifest.update(
        n_scores=local_ann.height,
        n_annotated=int(local_ann["annotated"].sum()),
        n_verified=int((local_ann["annotation_status"] == AnnotationStatus.HUMAN_VERIFIED.value).sum()),
        n_publications=local_ann["pgp_id"].n_unique(),
        n_concepts=local_clu["cluster"].n_unique(),
        n_phenotypes=local_clu.height,
        merged_from_published=remote_ann_path.exists(),
        n_kept_from_published=kept_remote,
        n_protected_verified=protected,
        sha256=hashes,
    )
    (out_dir / "curation_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


@app.command()
def push(
    out: Annotated[Path, typer.Option("--out", help="Build directory")] = DEFAULT_BUILD_DIR,
    verified_only: Annotated[bool, typer.Option("--verified-only")] = False,
    replace: Annotated[bool, typer.Option("--replace", help="Overwrite the release instead of merging")] = False,
    repo: Annotated[str | None, typer.Option("--repo", help="HF dataset (default just-dna-seq/pgs-catalog)")] = None,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Rebuild, merge with the published release, and upload all three files in one HF commit."""
    import tempfile

    from just_prs.hf import (
        DEFAULT_HF_CATALOG_REPO,
        pull_curation_tables,
        push_curation_tables,
    )

    repo_id = repo or DEFAULT_HF_CATALOG_REPO
    try:
        manifest = write_tables(root, out, verified_only)
        if not replace:
            with tempfile.TemporaryDirectory() as tmp:
                pull_curation_tables(Path(tmp), repo_id=repo_id)
                manifest = merge_with_published(out, Path(tmp))
    except CurationBuildError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if not replace:
        typer.echo(
            f"merged with published: kept {manifest['n_kept_from_published']} rows not in this clone, "
            f"protected {manifest['n_protected_verified']} verified rows"
        )
    if manifest["n_scores"] == 0:
        typer.echo("nothing curated yet; refusing to publish empty tables.", err=True)
        raise typer.Exit(1)
    uploaded = push_curation_tables(
        out,
        repo_id=repo_id,
        commit_message=(
            f"Curation: {manifest['n_annotated']} annotated / {manifest['n_verified']} verified scores, "
            f"{manifest['n_concepts']} concepts (source {manifest['source_commit'] or 'uncommitted'})"
        ),
    )
    typer.echo(f"pushed {', '.join(uploaded)} to {repo_id}/data/metadata/")


# ---------------------------------------------------------------------------
# Prioritization (no LLM): which trait groups most need review
# ---------------------------------------------------------------------------

_FLIP_WORDS_RE = (
    r"(?i)\b(never|ever|non|without|absence|low|high|protect|surviv\w*|longevity|mortality|death|age at|"
    r"onset|response|remission|resistance|tolerance|ease|factor|index|frequency|reverse)\b"
)
#: Traits people commonly look up; boosts review priority. Substrings of the ontology label.
_POPULAR_TRAITS = (
    "body height", "body mass index", "coronary", "type 2 diabetes", "breast", "prostate", "colorectal",
    "alzheimer", "low density lipoprotein", "high density lipoprotein", "total cholesterol", "triglyceride",
    "blood pressure", "smoking", "alcohol", "chronotype", "sleep", "insomnia", "hair color", "suntan",
    "balding", "asthma", "depress", "schizophrenia", "bipolar", "glomerular", "kidney", "stroke",
    "atrial fibrillation", "lung carcinoma", "bone", "heart rate", "coffee", "lipoprotein a", "body weight",
    "educational attainment", "myopia", "eye colo", "skin pigmentation", "migraine", "osteoarthritis",
)


def trait_priorities(root: Path) -> pl.DataFrame:
    """One row per ontology trait group with review signals and a priority score.

    Signals: score count, distinct reported phenotypes, flip wording, 1000G EUR negative
    correlations inside the group (the strongest hint of opposite or sign-flipped scores),
    popularity, and whether a trait worklist already exists.
    """
    catalog = PRSCatalog()
    scores = catalog.scores(include_excluded=True).select("pgs_id", "pgp_id", "trait_reported", "trait_efo").collect()
    done = {
        pgs_id
        for path in (root / "traits").glob("*.yaml")
        for pgs_id, s in (_load_yaml(path).get("scores") or {}).items()
        if (s or {}).get("decision") == TraitDecision.IN_SCOPE.value
    }
    cache: dict[str, pl.DataFrame | None] = {}
    rows: list[dict[str, object]] = []
    for group in scores.partition_by("trait_efo"):
        ids = group["pgs_id"].to_list()
        frames = [
            df.select("iid", pl.col("prs_score").alias(i))
            for i in ids[:80]
            if (df := _reference_scores(catalog, i, cache)) is not None
        ]
        min_r, n_neg, n_pairs = None, 0, 0
        if len(frames) >= 2:
            wide = frames[0]
            for frame in frames[1:]:
                wide = wide.join(frame, on="iid", how="inner")
            matrix = wide.drop("iid").to_numpy()
            matrix = matrix[:, matrix.std(axis=0) > 0]
            if matrix.shape[1] >= 2:
                values = np.corrcoef(matrix, rowvar=False)[np.triu_indices(matrix.shape[1], 1)]
                min_r, n_neg, n_pairs = float(np.nanmin(values)), int((values < -CORR_THRESHOLD).sum()), len(values)
        label = str(group["trait_efo"][0] or "")
        popular = any(p in label.lower() for p in _POPULAR_TRAITS)
        n_reported = group["trait_reported"].n_unique()
        n_flip = int(group["trait_reported"].fill_null("").str.contains(_FLIP_WORDS_RE).sum())
        curated = len(set(ids) & done)
        priority = (
            (4.0 if n_neg else 0.0)
            + min(n_neg / max(n_pairs, 1) * 10, 3.0)
            + float(np.log2(len(ids) + 1))
            + (1.0 if n_reported >= 3 else 0.0)
            + (1.0 if n_flip else 0.0)
            + (2.0 if popular else 0.0)
        )
        rows.append(
            {
                "trait_efo": label,
                "priority": round(priority, 2),
                "n_scores": len(ids),
                "n_papers": group["pgp_id"].n_unique(),
                "n_reported": n_reported,
                "n_flip_words": n_flip,
                "n_neg_pairs": n_neg,
                "n_pairs": n_pairs,
                "min_r": None if min_r is None else round(min_r, 3),
                "popular": popular,
                "n_curated": curated,
                "examples": " | ".join(group["trait_reported"].drop_nulls().unique().sort().head(6).to_list()),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(["n_curated", "priority"], descending=[False, True])


@app.command()
def prioritize(
    out: Annotated[Path, typer.Option("--out")] = DEFAULT_BUILD_DIR / "trait_priorities.csv",
    top: Annotated[int, typer.Option("--top", help="Rows to print")] = 40,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Rank ontology trait groups by how much they need review (no LLM; ~30 s)."""
    df = trait_priorities(root)
    df.write_csv(out)
    typer.echo(f"{df.height} trait groups → {out}; {df.filter(pl.col('n_neg_pairs') > 0).height} with negative pairs")
    typer.echo("prio  n    neg/pairs   min_r  cur  trait")
    for r in df.head(top).to_dicts():
        min_r = f"{r['min_r']:+.2f}" if r["min_r"] is not None else "  -  "
        typer.echo(
            f"{r['priority']:>5.1f} {r['n_scores']:>4} {r['n_neg_pairs']:>5}/{r['n_pairs']:<5} {min_r} {r['n_curated']:>4}  {r['trait_efo']}"
        )


# ---------------------------------------------------------------------------
# Map-reduce: stage a private copy per trait agent, merge it back
# ---------------------------------------------------------------------------


def _staging_dir(root: Path, slug: str) -> Path:
    return root / "staging" / slug


@app.command()
def stage(slug: str, root: RootOpt = DEFAULT_ROOT) -> None:
    """Create curation/staging/<slug>/: a private copy for one trait agent (plus a base snapshot)."""
    import shutil

    target = _staging_dir(root, slug)
    if target.exists():
        typer.echo(f"{target} already exists (an unmerged run?); merge it or delete it first.", err=True)
        raise typer.Exit(2)
    for dest in (target, target / ".base"):
        (dest / "publications").mkdir(parents=True)
        (dest / "traits").mkdir()
        if (root / "clusters.yaml").exists():
            shutil.copy2(root / "clusters.yaml", dest / "clusters.yaml")
        for path in (root / "publications").glob("*.yaml"):
            shutil.copy2(path, dest / "publications" / path.name)
        if _trait_path(root, slug).exists():
            shutil.copy2(_trait_path(root, slug), dest / "traits" / f"{slug}.yaml")
    typer.echo(str(target))


def _merge_entries(
    base: dict[str, object], mine: dict[str, object], theirs: dict[str, object], label: str
) -> tuple[dict[str, object], list[str], int]:
    """Three-way merge of keyed entries. Returns (merged main, conflicts, n_taken)."""
    merged, conflicts, taken = dict(mine), [], 0
    for key, value in theirs.items():
        before, current = base.get(key), mine.get(key)
        if value == current or value == before:
            continue  # unchanged by the trait agent, or already equal
        if isinstance(current, dict) and current.get("status") == AnnotationStatus.HUMAN_VERIFIED.value:
            conflicts.append(f"{label}/{key}: main is human_verified; trait change not applied")
            continue
        if current is not None and current != before:
            conflicts.append(f"{label}/{key}: changed in main and by the trait agent")
            continue
        merged[key] = value
        taken += 1
    return merged, conflicts, taken


@app.command()
def merge(
    slug: str,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    keep: Annotated[bool, typer.Option("--keep", help="Keep the staging dir after merging")] = False,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Merge curation/staging/<slug>/ back into curation/ (three-way, per score and per phenotype).

    Takes what the trait agent changed relative to its base snapshot. Anything also changed in
    main since staging, or already human_verified in main, is reported as a conflict and left alone.
    """
    import shutil

    stage_dir = _staging_dir(root, slug)
    base_dir = stage_dir / ".base"
    if not stage_dir.exists():
        typer.echo(f"no staging dir {stage_dir}", err=True)
        raise typer.Exit(2)
    conflicts: list[str] = []
    n_scores = n_phenotypes = 0
    writes: dict[Path, str] = {}

    # clusters: merge phenotype-by-phenotype; concept fields must agree
    main_clusters = _load_yaml(root / "clusters.yaml") if (root / "clusters.yaml").exists() else {"version": 0, "domains": {}}
    base_clusters = _load_yaml(base_dir / "clusters.yaml") if (base_dir / "clusters.yaml").exists() else {"domains": {}}
    stage_clusters = _load_yaml(stage_dir / "clusters.yaml") if (stage_dir / "clusters.yaml").exists() else {"domains": {}}
    domains = main_clusters.setdefault("domains", {}) or {}
    main_clusters["domains"] = domains
    for d_key, d_val in (stage_clusters.get("domains") or {}).items():
        main_domain = domains.setdefault(d_key, {"label": d_val["label"], "concepts": {}})
        base_domain = (base_clusters.get("domains") or {}).get(d_key, {"concepts": {}})
        for c_key, c_val in (d_val.get("concepts") or {}).items():
            main_concept = main_domain["concepts"].get(c_key)
            base_concept = (base_domain.get("concepts") or {}).get(c_key)
            if main_concept is None:
                main_domain["concepts"][c_key] = c_val
                n_phenotypes += len(c_val.get("phenotypes") or {})
                continue
            fields = {k: v for k, v in c_val.items() if k != "phenotypes"}
            base_fields = {k: v for k, v in (base_concept or {}).items() if k != "phenotypes"}
            main_fields = {k: v for k, v in main_concept.items() if k != "phenotypes"}
            if fields != base_fields and fields != main_fields:
                if main_fields != base_fields:
                    conflicts.append(f"clusters/{d_key}.{c_key}: concept fields changed in main and by the trait agent")
                else:
                    main_concept.update(fields)
            merged, found, taken = _merge_entries(
                (base_concept or {}).get("phenotypes") or {},
                main_concept.get("phenotypes") or {},
                c_val.get("phenotypes") or {},
                f"clusters/{d_key}.{c_key}",
            )
            main_concept["phenotypes"] = merged
            conflicts += found
            n_phenotypes += taken
    writes[root / "clusters.yaml"] = yaml.safe_dump(main_clusters, sort_keys=False, width=100, allow_unicode=True)

    # publications: per score entry; paper-level fields filled when main lacks them
    for path in sorted((stage_dir / "publications").glob("*.yaml")):
        theirs = _load_yaml(path)
        base = _load_yaml(base_dir / "publications" / path.name) if (base_dir / "publications" / path.name).exists() else {}
        main_path = root / "publications" / path.name
        mine = _load_yaml(main_path) if main_path.exists() else {"pgp_id": path.stem, "scores": {}}
        if theirs == base:
            continue
        merged_scores, found, taken = _merge_entries(
            base.get("scores") or {}, mine.get("scores") or {}, theirs.get("scores") or {}, path.stem
        )
        conflicts += found
        n_scores += taken
        for field in ("paper", "evidence_read", "template", "exploration_findings"):
            if theirs.get(field) and not mine.get(field):
                mine[field] = theirs[field]
        mine["scores"] = merged_scores
        writes[main_path] = yaml.safe_dump(mine, sort_keys=False, width=110, allow_unicode=True)

    trait_file = stage_dir / "traits" / f"{slug}.yaml"
    if trait_file.exists():
        writes[_trait_path(root, slug)] = trait_file.read_text()
    progress = stage_dir / "progress.txt"

    typer.echo(f"merge {slug}: {n_scores} score entries, {n_phenotypes} phenotypes, {len(conflicts)} conflicts")
    for conflict in conflicts:
        typer.echo(f"  CONFLICT {conflict}")
    if dry_run:
        return
    for target, text in writes.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    if progress.exists():
        with _progress_path(root).open("a") as handle:
            handle.write(progress.read_text())
    try:
        build_tables(root)
    except CurationBuildError as exc:
        typer.echo(f"merged, but the result is inconsistent (staging kept):\n{exc}", err=True)
        raise typer.Exit(1) from exc
    if not keep and not conflicts:
        shutil.rmtree(stage_dir)
        typer.echo(f"removed {stage_dir}")
    elif conflicts:
        typer.echo(f"kept {stage_dir} because of conflicts; resolve them, then delete it")


# ---------------------------------------------------------------------------
# Trait worklists
# ---------------------------------------------------------------------------


def _trait_path(root: Path, slug: str) -> Path:
    return root / "traits" / f"{slug}.yaml"


def _trait_cache_dir(slug: str) -> Path:
    return resolve_cache_dir() / "annotation_lookups" / "_traits" / slug


def _load_trait(root: Path, slug: str) -> TraitWorklist:
    path = _trait_path(root, slug)
    if not path.exists():
        typer.echo(f"no worklist {path}; run `trait-scan {slug} --query ... --term ...` first.", err=True)
        raise typer.Exit(2)
    return TraitWorklist.model_validate(_load_yaml(path))


def _save_trait(root: Path, worklist: TraitWorklist) -> None:
    path = _trait_path(root, worklist.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(worklist.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100))
    tmp.replace(path)


def _annotation_index(root: Path) -> dict[str, tuple[str, ScoreAnnotation]]:
    """PGS ID -> (PGP ID, annotation) across all publication files."""
    return {
        pgs_id: (pgp_id, ann)
        for pgp_id, pub in _all_publications(root).items()
        for pgs_id, ann in pub.scores.items()
    }


def _correlation_matrix(catalog: PRSCatalog, pgs_ids: list[str]) -> tuple[list[str], list[list[float]]]:
    """Pearson correlation of 1000G EUR per-individual scores; IDs without cached scores are dropped."""
    cache: dict[str, pl.DataFrame | None] = {}
    frames = [
        df.select("iid", pl.lit(pgs_id).alias("pgs_id"), "prs_score")
        for pgs_id in pgs_ids
        if (df := _reference_scores(catalog, pgs_id, cache)) is not None
    ]
    if not frames:
        return [], []
    wide = pl.concat(frames).pivot(on="pgs_id", index="iid", values="prs_score").drop_nulls()
    usable = [c for c in wide.columns if c != "iid" and wide[c].std() not in (None, 0.0)]
    if len(usable) < 2 or wide.height < CORR_MIN_SAMPLES:
        return usable, []
    matrix = wide.select(usable).to_numpy()
    return usable, np.corrcoef(matrix, rowvar=False).tolist()


@app.command("trait-scan")
def trait_scan(
    slug: str,
    query: Annotated[str | None, typer.Option("--query", help="The user's question, e.g. 'sort out longevity'")] = None,
    term: Annotated[list[str] | None, typer.Option("--term", help="Whole-word match on name / reported trait / ontology label")] = None,
    efo: Annotated[list[str] | None, typer.Option("--efo", help="Ontology ID contained in trait_efo_id")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help="Regex on reported trait to drop")] = None,
    anchor: Annotated[str | None, typer.Option("--anchor", help="PGS ID the correlation signs are read against")] = None,
    rows: Annotated[int, typer.Option("--rows")] = TRIAGE_MAX_ROWS,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Find every score for a trait, write/merge its worklist, and print a triage table.

    Terms, ontology IDs and exclusions accumulate across runs, so widen the search by rerunning
    with extra ``--term`` / ``--efo``. Existing per-score decisions are kept.
    """
    path = _trait_path(root, slug)
    if path.exists():
        worklist = TraitWorklist.model_validate(_load_yaml(path))
    elif query:
        worklist = TraitWorklist(slug=slug, query=query)
    else:
        typer.echo("new worklist: pass --query with the user's question.", err=True)
        raise typer.Exit(2)
    match = worklist.match
    match.terms = sorted(set(match.terms) | set(term or []))
    match.efo_ids = sorted(set(match.efo_ids) | set(efo or []))
    match.exclude = sorted(set(match.exclude) | set(exclude or []))
    if anchor:
        worklist.anchor = anchor
    if not match.terms and not match.efo_ids:
        typer.echo("give at least one --term or --efo.", err=True)
        raise typer.Exit(2)

    catalog = PRSCatalog()
    text = pl.concat_str(
        [pl.col("name").fill_null(""), pl.col("trait_reported").fill_null(""), pl.col("trait_efo").fill_null("")],
        separator=" | ",
    )
    conditions = [text.str.contains(rf"(?i)\b{re.escape(t)}\b") for t in match.terms]
    conditions += [pl.col("trait_efo_id").fill_null("").str.contains(re.escape(e)) for e in match.efo_ids]
    hit = conditions[0]
    for condition in conditions[1:]:
        hit = hit | condition
    lf = catalog.scores(include_excluded=True).filter(hit)
    for pattern in match.exclude:
        lf = lf.filter(~pl.col("trait_reported").fill_null("").str.contains(f"(?i){pattern}"))
    matched = lf.select("pgs_id", "pgp_id", "trait_reported", "trait_efo", "trait_efo_id", "n_variants").collect().sort("pgs_id")
    ids = matched["pgs_id"].to_list()
    for pgs_id in ids:
        worklist.scores.setdefault(pgs_id, TraitScore())
    dropped = sorted(set(worklist.scores) - set(ids))

    best = {
        r["pgs_id"]: r for r in catalog.best_performance().filter(pl.col("pgs_id").is_in(ids)).collect().to_dicts()
    }
    candidates = [
        i for i in ids if worklist.scores[i].decision != TraitDecision.OUT_OF_SCOPE
    ]
    corr_ids, corr = _correlation_matrix(catalog, candidates)
    position = {pgs_id: k for k, pgs_id in enumerate(corr_ids)}
    if worklist.anchor not in position and corr_ids:
        worklist.anchor = max(corr_ids, key=lambda i: (best.get(i, {}).get("n_individuals") or 0, i))
    anchor_pos = position.get(worklist.anchor or "")
    _save_trait(root, worklist)

    cache_dir = _trait_cache_dir(slug)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if corr:
        pl.DataFrame(
            [
                {"pgs_a": a, "pgs_b": b, "r": corr[i][j]}
                for i, a in enumerate(corr_ids)
                for j, b in enumerate(corr_ids)
                if i < j
            ]
        ).write_parquet(cache_dir / "corr.parquet")

    annotations = _annotation_index(root)
    triage: list[dict[str, object]] = []
    for row in matched.to_dicts():
        pgs_id = row["pgs_id"]
        sign, metric = _metric_sign(best[pgs_id]) if pgs_id in best else (None, "no evaluation")
        r_anchor = corr[position[pgs_id]][anchor_pos] if corr and anchor_pos is not None and pgs_id in position else None
        min_r, partner = None, None
        if corr and pgs_id in position:
            others = [(corr[position[pgs_id]][k], other) for other, k in position.items() if other != pgs_id]
            if others:
                min_r, partner = min(others)
        group = "?" if r_anchor is None else ("+" if r_anchor >= TRIAGE_SIGN_R else "-" if r_anchor <= -TRIAGE_SIGN_R else "~")
        pgp_ann = annotations.get(pgs_id)
        ann = pgp_ann[1] if pgp_ann else None
        triage.append(
            {
                "pgs_id": pgs_id,
                "pgp_id": row["pgp_id"],
                "trait_reported": row["trait_reported"],
                "trait_efo": row["trait_efo"],
                "decision": worklist.scores[pgs_id].decision.value,
                "metric": metric,
                "metric_sign": sign,
                "evaluated_as": best.get(pgs_id, {}).get("trait_reported"),
                "r_anchor": None if r_anchor is None else round(r_anchor, 3),
                "group_vs_anchor": group,
                "min_r": None if min_r is None else round(min_r, 3),
                "min_r_partner": partner,
                "annotation": (
                    f"{ann.cluster}.{ann.phenotype} {ann.polarity:+d} {ann.status}"
                    if ann is not None and ann.score_effect is not None
                    else ("unannotated" if ann is not None else "")
                ),
            }
        )
    triage_df = pl.DataFrame(triage, infer_schema_length=None)
    triage_df.write_csv(cache_dir / "triage.csv")

    typer.echo(
        f"trait {slug}: {len(ids)} scores from {matched['pgp_id'].n_unique()} papers; "
        f"anchor {worklist.anchor or '-'}; 1000G EUR correlations for {len(corr_ids)} scores"
    )
    by_reported = matched.group_by("trait_reported").len().sort("len", descending=True)
    typer.echo(f"{by_reported.height} distinct reported phenotypes; top: " + "; ".join(
        f"{r['trait_reported']} ({r['len']})" for r in by_reported.head(12).to_dicts()
    ))
    typer.echo("grp  r_anc  min_r(partner)          pgs_id     pgp        decision      metric → evaluated as | reported trait | annotation")
    for row in triage[:rows]:
        min_part = f"{row['min_r']:+.2f}({row['min_r_partner']})" if row["min_r"] is not None else "-"
        r_anc = f"{row['r_anchor']:+.2f}" if row["r_anchor"] is not None else "  -  "
        typer.echo(
            f"{row['group_vs_anchor']:<4} {r_anc:<6} {min_part:<23} {row['pgs_id']:<10} {row['pgp_id']:<10} "
            f"{row['decision']:<13} {row['metric']} → {row['evaluated_as'] or '-'} | {row['trait_reported']} | {row['annotation']}"
        )
    if len(triage) > rows:
        typer.echo(f"... {len(triage) - rows} more rows in {cache_dir / 'triage.csv'}")
    if dropped:
        typer.echo(f"in the worklist but no longer matched (kept, review them): {', '.join(dropped)}")
    typer.echo(f"worklist: {path}  triage: {cache_dir / 'triage.csv'}")


def _trait_counts(worklist: TraitWorklist, annotations: dict[str, tuple[str, ScoreAnnotation]]) -> dict[str, int]:
    in_scope = [i for i, s in worklist.scores.items() if s.decision == TraitDecision.IN_SCOPE]
    annotated = [i for i in in_scope if i in annotations and annotations[i][1].score_effect is not None]
    return {
        "total": len(worklist.scores),
        "in_scope": len(in_scope),
        "pending": sum(1 for s in worklist.scores.values() if s.decision == TraitDecision.PENDING),
        "out_of_scope": sum(1 for s in worklist.scores.values() if s.decision == TraitDecision.OUT_OF_SCOPE),
        "annotated": len(annotated),
        "unannotated": sum(1 for i in in_scope if i in annotations and annotations[i][1].score_effect is None),
        "verified": sum(1 for i in annotated if annotations[i][1].status == AnnotationStatus.HUMAN_VERIFIED),
    }


@app.command("trait-status")
def trait_status(
    slug: str,
    corr_threshold: Annotated[float, typer.Option("--corr-threshold")] = CORR_THRESHOLD,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Print the sorted view of a trait: concept → phenotype → polarity, plus conflicts and gaps."""
    worklist = _load_trait(root, slug)
    annotations = _annotation_index(root)
    clusters = _load_clusters(root)
    counts = _trait_counts(worklist, annotations)
    typer.echo(
        f"trait {slug} ({worklist.status}): {counts['annotated']}/{counts['in_scope']} in-scope annotated, "
        f"{counts['unannotated']} unannotated with reason, {counts['pending']} undecided, "
        f"{counts['out_of_scope']} out of scope, {counts['verified']} verified"
    )
    grouped: dict[str, dict[str, list[str]]] = {}
    polarity_of: dict[str, int] = {}
    cluster_of: dict[str, str] = {}
    for pgs_id, entry in worklist.scores.items():
        if entry.decision != TraitDecision.IN_SCOPE:
            continue
        pgp_ann = annotations.get(pgs_id)
        if pgp_ann is None or pgp_ann[1].score_effect is None:
            continue
        ann = pgp_ann[1]
        polarity_of[pgs_id] = ann.polarity or 0
        cluster_of[pgs_id] = ann.cluster or ""
        label = f"{'+' if ann.polarity == 1 else '−'} {ann.phenotype}"
        marker = "" if ann.status == AnnotationStatus.HUMAN_VERIFIED else "*"
        grouped.setdefault(ann.cluster or "?", {}).setdefault(label, []).append(f"{pgs_id}{marker}")
    for cluster_key, phenotypes in sorted(grouped.items()):
        concept = clusters.concept(cluster_key)
        header = f"{cluster_key}  (axis: {concept.axis}; valence: {concept.valence})" if concept else cluster_key
        typer.echo(header)
        for label, members in sorted(phenotypes.items()):
            typer.echo(f"  {label}: {', '.join(members)}")
    typer.echo("  (* = agent_proposed, not yet reviewed)")

    stray = sorted({c for c in cluster_of.values() if c not in worklist.concepts})
    if stray:
        typer.echo(f"concepts used but not listed in the worklist: {', '.join(stray)}")
    corr_path = _trait_cache_dir(slug) / "corr.parquet"
    if corr_path.exists():
        conflicts = [
            (row["pgs_a"], row["pgs_b"], row["r"], row["r"] * polarity_of[row["pgs_a"]] * polarity_of[row["pgs_b"]])
            for row in pl.read_parquet(corr_path).to_dicts()
            if row["pgs_a"] in polarity_of
            and row["pgs_b"] in polarity_of
            and cluster_of[row["pgs_a"]] == cluster_of[row["pgs_b"]]
        ]
        flagged = [c for c in conflicts if c[3] < -corr_threshold]
        typer.echo(f"aligned-correlation conflicts (< -{corr_threshold}): {len(flagged)} of {len(conflicts)} same-concept pairs")
        for a, b, r, aligned in sorted(flagged, key=lambda c: c[3])[:20]:
            typer.echo(f"  {a} vs {b}: r={r:+.2f}, aligned={aligned:+.2f}")
    pending = [i for i, s in worklist.scores.items() if s.decision == TraitDecision.PENDING]
    missing = [
        i for i, s in worklist.scores.items()
        if s.decision == TraitDecision.IN_SCOPE and i not in annotations
    ]
    if pending:
        typer.echo(f"undecided ({len(pending)}): {', '.join(pending[:40])}{' …' if len(pending) > 40 else ''}")
    if missing:
        typer.echo(f"in scope, not yet read ({len(missing)}): {', '.join(missing[:40])}{' …' if len(missing) > 40 else ''}")
    out = [(i, s.note) for i, s in worklist.scores.items() if s.decision == TraitDecision.OUT_OF_SCOPE]
    if out:
        typer.echo("out of scope: " + "; ".join(f"{i} ({note or 'no note'})" for i, note in out[:30]))


@app.command("trait-log")
def trait_log(
    slug: str,
    message: Annotated[str, typer.Option("--message")],
    lookups: Annotated[int, typer.Option("--lookups")] = 0,
    root: RootOpt = DEFAULT_ROOT,
) -> None:
    """Append a progress line for a trait worklist."""
    worklist = _load_trait(root, slug)
    counts = _trait_counts(worklist, _annotation_index(root))
    pct = 100 * counts["annotated"] / max(counts["in_scope"], 1)
    line = (
        f"{_now()} Trait {slug}: {counts['annotated']}/{counts['in_scope']} in-scope annotated ({pct:.1f}%), "
        f"{counts['unannotated']} unannotated, {counts['pending']} undecided, {counts['out_of_scope']} out of scope, "
        f"{counts['verified']} verified, lookups {lookups} | {message}"
    )
    with _progress_path(root).open("a") as handle:
        handle.write(line + "\n")
    typer.echo(line)


if __name__ == "__main__":
    app()

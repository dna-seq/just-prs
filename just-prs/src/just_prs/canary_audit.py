"""Canary-genome collapse flags for catalog / percentile quarantine.

A genome-wide score can look healthy on the 1000G pgen (match ~100%) and still
produce a 0th-percentile on every variant-only WGS VCF, because the user sum
only includes ALT-present sites while the published reference is the full sum.
``PGS003724`` (IQ) is the type specimen: user totals 32–145 vs EUR mean 767.

Callers pass canary VCFs via ``--vcf`` (path, alias, or ``Label=path``). The
catalog is scored on those genomes (not the 1000G panel) and persisted to
``results/canary_scores.parquet``. A PGS ID is marked unreliable when a
majority of those samples land at percentile 0 or close to it (or the
symmetric 100th-percentile collapse) **and** the result fails a scale check
(median ``|z|`` ≥ 6 or match < 50%). ERROR rows are merged into the percentile
audit sidecar so ``PRSCatalog.reference_distributions()`` and ``scores()`` drop
the ID without rewriting the 1000G distributions parquet.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel

from just_prs.reference import DISTRIBUTION_ISSUE_SCHEMA

CANARY_COLLAPSE_ISSUE = "canary_collapsed_percentile"
CANARY_FLAGS_FILENAME = "catalog_scoring_flags.parquet"
CANARY_SCORES_FILENAME = "canary_scores.parquet"

CANARY_RESULT_SCHEMA: dict[str, pl.DataType] = {
    "pgs_id": pl.Utf8,
    "sample_id": pl.Utf8,
    "percentile": pl.Float64,
    "z_score": pl.Float64,
    "match_rate": pl.Float64,
    "score": pl.Float64,
}


class CanarySample(BaseModel):
    """One canary genome: a CLI label plus a resolved VCF path."""

    label: str
    vcf_path: Path
    genome_build: str | None = None


@dataclass
class CanaryScoreProgress:
    """Incremental scoring counters for Dagster metadata."""

    n_total: int = 0
    n_ok: int = 0
    n_cached: int = 0
    n_failed: int = 0
    failed_ids: list[str] = field(default_factory=list)

DEFAULT_PERCENTILE_FLOOR = 1.0
DEFAULT_PERCENTILE_CEIL = 99.0
DEFAULT_MIN_ABS_Z = 6.0
DEFAULT_MAX_MATCH_RATE = 0.50

CANARY_FLAGS_SCHEMA: dict[str, pl.DataType] = {
    "pgs_id": pl.Utf8,
    "flag": pl.Utf8,
    "severity": pl.Utf8,
    "exclude_from_catalog": pl.Boolean,
    "n_canary_samples": pl.Int64,
    "n_extreme": pl.Int64,
    "min_percentile": pl.Float64,
    "max_percentile": pl.Float64,
    "median_abs_z": pl.Float64,
    "median_match_rate": pl.Float64,
    "reason": pl.Utf8,
}


def majority_count(n_samples: int) -> int:
    """Smallest integer strictly greater than half of ``n_samples``."""
    if n_samples < 2:
        return 2
    return n_samples // 2 + 1


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _row_from_result(payload: dict[str, Any], *, sample_id: str) -> dict[str, Any] | None:
    pgs_id = payload.get("pgs_id")
    percentile = payload.get("percentile")
    if not pgs_id or percentile is None:
        return None
    z_score = payload.get("z_score")
    match_rate = payload.get("match_rate")
    return {
        "pgs_id": str(pgs_id).upper(),
        "sample_id": sample_id,
        "percentile": float(percentile),
        "z_score": float(z_score) if z_score is not None else None,
        "match_rate": float(match_rate) if match_rate is not None else None,
        "score": float(payload["score"]) if payload.get("score") is not None else None,
    }


def _rows_from_mapping(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        sample_id = str(payload.get("sample_id") or payload.get("label") or key)
        row = _row_from_result(payload, sample_id=sample_id)
        if row is not None:
            rows.append(row)
    return rows


def _rows_from_list(data: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, payload in enumerate(data):
        if not isinstance(payload, dict):
            continue
        sample_id = str(payload.get("sample_id") or payload.get("label") or f"row_{idx}")
        row = _row_from_result(payload, sample_id=sample_id)
        if row is not None:
            rows.append(row)
    return rows


def load_canary_result_rows(
    cache_dir: Path,
    extra_json: Path | list[Path] | None = None,
) -> pl.DataFrame:
    """Load scored canary rows from ``canary_scores.parquet`` and optional extra JSON."""
    rows: list[dict[str, Any]] = []
    persisted = canary_scores_path(cache_dir)
    if persisted.exists():
        persisted_df = _read_canary_scores(persisted)
        if persisted_df.height:
            rows.extend(persisted_df.to_dicts())

    extra_paths: list[Path] = []
    if extra_json is not None:
        extra_paths.extend(extra_json if isinstance(extra_json, list) else [extra_json])
    env_extra = os.environ.get("PRS_CANARY_RESULTS", "").strip()
    if env_extra:
        extra_paths.extend(Path(part.strip()) for part in env_extra.split(",") if part.strip())
    for path in extra_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.extend(_rows_from_mapping(payload))
        elif isinstance(payload, list):
            rows.extend(_rows_from_list(payload))

    if not rows:
        return pl.DataFrame(schema=CANARY_RESULT_SCHEMA)
    return pl.DataFrame(rows).unique(subset=["pgs_id", "sample_id"], keep="last")


def flag_canary_collapses(
    rows: pl.DataFrame,
    min_samples: int | None = None,
    n_samples: int | None = None,
    percentile_floor: float | None = None,
    percentile_ceil: float | None = None,
    min_abs_z: float | None = None,
    max_match_rate: float | None = None,
) -> pl.DataFrame:
    """Flag PGS IDs that land at 0 (or ~100) on a majority of canary samples.

    ``min_samples`` is the majority threshold (default: more than half of the
    canary set). A collapse is only an ERROR when those extreme rows also fail
    a scale check: median ``|z|`` ≥ 6 **or** median match rate < 50%. Two
    people at the 2nd percentile with 90% match is a real tail, not this.
    """
    if rows.height == 0:
        return pl.DataFrame(schema=CANARY_FLAGS_SCHEMA)

    sample_set_size = n_samples if n_samples is not None else rows["sample_id"].n_unique()
    min_samples = min_samples if min_samples is not None else _env_int(
        "PRS_CANARY_MIN_SAMPLES", majority_count(int(sample_set_size))
    )
    percentile_floor = percentile_floor if percentile_floor is not None else _env_float(
        "PRS_CANARY_PERCENTILE_FLOOR", DEFAULT_PERCENTILE_FLOOR
    )
    percentile_ceil = percentile_ceil if percentile_ceil is not None else _env_float(
        "PRS_CANARY_PERCENTILE_CEIL", DEFAULT_PERCENTILE_CEIL
    )
    min_abs_z = min_abs_z if min_abs_z is not None else _env_float(
        "PRS_CANARY_MIN_ABS_Z", DEFAULT_MIN_ABS_Z
    )
    max_match_rate = max_match_rate if max_match_rate is not None else _env_float(
        "PRS_CANARY_MAX_MATCH_RATE", DEFAULT_MAX_MATCH_RATE
    )

    annotated = rows.with_columns(
        (
            (pl.col("percentile") <= percentile_floor)
            | (pl.col("percentile") >= percentile_ceil)
        ).alias("_extreme")
    )
    totals = annotated.group_by("pgs_id").agg(
        pl.col("sample_id").n_unique().alias("n_canary_samples"),
    )
    extremes = (
        annotated.filter(pl.col("_extreme"))
        .group_by("pgs_id")
        .agg(
            pl.len().alias("n_extreme"),
            pl.col("percentile").min().alias("min_percentile"),
            pl.col("percentile").max().alias("max_percentile"),
            pl.col("z_score").abs().median().alias("median_abs_z"),
            pl.col("match_rate").median().alias("median_match_rate"),
        )
    )
    if extremes.height == 0:
        return pl.DataFrame(schema=CANARY_FLAGS_SCHEMA)

    grouped = totals.join(extremes, on="pgs_id", how="inner")
    scale_break = (
        pl.col("median_abs_z").fill_null(0.0) >= min_abs_z
    ) | (
        pl.col("median_match_rate").fill_null(1.0) < max_match_rate
    )
    flagged = grouped.filter((pl.col("n_extreme") >= min_samples) & scale_break)
    if flagged.height == 0:
        return pl.DataFrame(schema=CANARY_FLAGS_SCHEMA)

    return flagged.with_columns(
        pl.lit(CANARY_COLLAPSE_ISSUE).alias("flag"),
        pl.lit("ERROR").alias("severity"),
        pl.lit(True).alias("exclude_from_catalog"),
        pl.format(
            "Near-zero (or ~100th) percentile on {}/{} canary genomes "
            "(median |z|={}, median match={}); marked unreliable.",
            pl.col("n_extreme"),
            pl.col("n_canary_samples"),
            pl.col("median_abs_z").round(2),
            (pl.col("median_match_rate") * 100.0).round(1),
        ).alias("reason"),
    ).select(list(CANARY_FLAGS_SCHEMA))


def canary_collapse_issues(
    distributions_df: pl.DataFrame,
    flags_df: pl.DataFrame,
) -> pl.DataFrame:
    """Expand catalog flags into per-superpopulation ERROR audit rows."""
    if flags_df.height == 0 or distributions_df.height == 0:
        return pl.DataFrame(schema=DISTRIBUTION_ISSUE_SCHEMA)
    excluded = flags_df.filter(pl.col("exclude_from_catalog")).select("pgs_id").unique()
    if excluded.height == 0:
        return pl.DataFrame(schema=DISTRIBUTION_ISSUE_SCHEMA)
    numeric_cols = ("mean", "std", "n", "median", "p5", "p25", "p75", "p95")
    matched = distributions_df.join(excluded, on="pgs_id", how="inner")
    if matched.height == 0:
        return pl.DataFrame(schema=DISTRIBUTION_ISSUE_SCHEMA)
    exprs: list[pl.Expr] = [
        pl.col("pgs_id").cast(pl.Utf8),
        pl.col("superpopulation").cast(pl.Utf8),
        pl.lit("ERROR").alias("severity"),
        pl.lit(CANARY_COLLAPSE_ISSUE).alias("issue"),
        pl.lit("exclude_from_catalog_and_percentiles").alias("recommended_action"),
    ]
    for col in numeric_cols:
        if col in matched.columns:
            dtype = pl.Int64 if col == "n" else pl.Float64
            exprs.append(pl.col(col).cast(dtype))
        else:
            exprs.append(pl.lit(None).alias(col))
    return matched.select(exprs)


def merge_audit_issues(base: pl.DataFrame, extra: pl.DataFrame) -> pl.DataFrame:
    """Concatenate audit issue tables, dropping duplicate (pgs_id, pop, issue)."""
    frames = [df for df in (base, extra) if df.height > 0]
    if not frames:
        return pl.DataFrame(schema=DISTRIBUTION_ISSUE_SCHEMA)
    return pl.concat(frames, how="diagonal_relaxed").unique(
        subset=["pgs_id", "superpopulation", "issue"],
        keep="last",
    )


def excluded_pgs_ids(flags_df: pl.DataFrame) -> list[str]:
    """PGS IDs marked ``exclude_from_catalog``."""
    if flags_df.height == 0 or "exclude_from_catalog" not in flags_df.columns:
        return []
    return sorted(
        flags_df.filter(pl.col("exclude_from_catalog"))["pgs_id"].unique().to_list()
    )


def catalog_flags_path(cache_dir: Path) -> Path:
    return cache_dir / "metadata" / CANARY_FLAGS_FILENAME


def canary_scores_path(cache_dir: Path) -> Path:
    return cache_dir / "results" / CANARY_SCORES_FILENAME


def write_catalog_flags(flags_df: pl.DataFrame, cache_dir: Path) -> Path:
    """Persist the flags parquet under ``<cache>/metadata/``."""
    path = catalog_flags_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags_df.write_parquet(path)
    return path


def parse_canary_vcf_spec(spec: str) -> tuple[str, str]:
    """Parse ``label=path_or_alias`` or a bare path/alias into ``(label, target)``."""
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty canary VCF spec.")
    if "=" in raw:
        label, _, target = raw.partition("=")
        label, target = label.strip(), target.strip()
        if not label or not target:
            raise ValueError(f"Invalid canary VCF spec {spec!r}; expected label=path_or_alias.")
        return label, target
    stem = Path(raw).stem
    for suffix in (".vcf", ".hard-filtered", ".g"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or "sample", raw


def resolve_canary_vcf(target: str, cache_dir: Path) -> Path:
    """Resolve a VCF path or alias to an existing file.

    Built-in aliases (``anton``, ``livia``) auto-download from Zenodo on first use.
    """
    path = Path(target).expanduser()
    if path.exists():
        return path.resolve()

    from just_prs.cli import BUILTIN_ZENODO_URLS, _download_builtin_vcf, _load_aliases

    aliases = _load_aliases(cache_dir)
    key = target.lower().strip()
    if key in aliases:
        resolved = Path(aliases[key]).expanduser()
        if not resolved.exists() and key in BUILTIN_ZENODO_URLS:
            _download_builtin_vcf(key, resolved)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Alias {target!r} points to {resolved} but the file does not exist. "
                f"Set it with: prs alias set {key} /path/to/file.vcf.gz"
            )
        return resolved.resolve()

    known = ", ".join(sorted(aliases)) or "(none)"
    raise FileNotFoundError(
        f"{target!r} is neither an existing file nor a known alias. "
        f"Known aliases: {known}. Add one with: prs alias set {key} /path/to/file.vcf.gz"
    )


def parse_canary_vcf_specs(
    specs: list[str],
    cache_dir: Path,
    genome_build: str | None = None,
) -> list[CanarySample]:
    """Resolve repeated ``--vcf`` specs into labeled canary samples."""
    samples: list[CanarySample] = []
    seen: set[str] = set()
    for spec in specs:
        label, target = parse_canary_vcf_spec(spec)
        key = label.lower()
        if key in seen:
            raise ValueError(f"Duplicate canary sample label {label!r}.")
        seen.add(key)
        samples.append(
            CanarySample(
                label=label,
                vcf_path=resolve_canary_vcf(target, cache_dir),
                genome_build=genome_build,
            )
        )
    return samples


def parse_canary_samples_env(raw: str, cache_dir: Path) -> list[CanarySample]:
    """Parse ``PRS_CANARY_VCFS`` JSON or comma-separated ``label=path`` specs."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        samples: list[CanarySample] = []
        for item in payload:
            samples.append(
                CanarySample(
                    label=str(item["label"]),
                    vcf_path=Path(item["path"]).expanduser(),
                    genome_build=item.get("genome_build"),
                )
            )
        return samples
    return parse_canary_vcf_specs(
        [part.strip() for part in text.split(",") if part.strip()],
        cache_dir,
    )


def encode_canary_samples_env(samples: list[CanarySample]) -> str:
    """Serialize samples for ``PRS_CANARY_VCFS`` (survives ``os.execvp`` into Dagster)."""
    return json.dumps([
        {
            "label": sample.label,
            "path": str(sample.vcf_path),
            "genome_build": sample.genome_build,
        }
        for sample in samples
    ])


def _read_canary_scores(path: Path) -> pl.DataFrame:
    try:
        df = pl.read_parquet(path)
    except (pl.exceptions.ComputeError, OSError):
        path.unlink(missing_ok=True)
        return pl.DataFrame(schema=CANARY_RESULT_SCHEMA)
    missing = [col for col in CANARY_RESULT_SCHEMA if col not in df.columns]
    for col in missing:
        df = df.with_columns(pl.lit(None).cast(CANARY_RESULT_SCHEMA[col]).alias(col))
    return df.select(list(CANARY_RESULT_SCHEMA))


def load_canary_scores(cache_dir: Path) -> pl.DataFrame:
    """Load persisted per-sample canary scores (empty frame if missing)."""
    path = canary_scores_path(cache_dir)
    if not path.exists():
        return pl.DataFrame(schema=CANARY_RESULT_SCHEMA)
    return _read_canary_scores(path)


def upsert_canary_score_rows(cache_dir: Path, rows: pl.DataFrame) -> Path:
    """Append/replace rows keyed by ``(pgs_id, sample_id)``."""
    path = canary_scores_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = rows.select(list(CANARY_RESULT_SCHEMA)) if rows.height else pl.DataFrame(schema=CANARY_RESULT_SCHEMA)
    existing = load_canary_scores(cache_dir)
    combined = pl.concat([existing, incoming], how="diagonal_relaxed")
    if combined.height:
        combined = combined.unique(subset=["pgs_id", "sample_id"], keep="last")
    tmp = path.with_suffix(".parquet.tmp")
    combined.write_parquet(tmp)
    tmp.replace(path)
    return path


def _normalized_canary_parquet(cache_dir: Path, sample: CanarySample) -> Path:
    from just_prs.normalize import normalize_vcf

    out_dir = cache_dir / "normalized" / "canary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample.label}.parquet"
    source = sample.vcf_path
    if out_path.exists() and out_path.stat().st_mtime >= source.stat().st_mtime:
        return out_path
    return normalize_vcf(source, out_path)


def _catalog_pgs_ids(catalog: Any, genome_build: str, requested: list[str] | None, limit: int | None) -> list[str]:
    lf = catalog.scores(
        genome_build=genome_build,
        include_harmonized=True,
        include_excluded=True,
    )
    ids = sorted(lf.select("pgs_id").unique().collect()["pgs_id"].to_list())
    if requested:
        wanted = {item.strip().upper() for item in requested if item.strip()}
        ids = [pgs_id for pgs_id in ids if pgs_id in wanted]
    if limit is not None:
        ids = ids[:limit]
    return ids


def score_canary_catalog(
    samples: list[CanarySample],
    cache_dir: Path,
    *,
    pgs_ids: list[str] | None = None,
    limit: int | None = None,
    ancestry: str = "EUR",
    panel: str = "1000g",
    genome_build: str | None = None,
    skip_existing: bool = True,
    progress_every: int = 10,
    log: Callable[[str], None] | None = None,
) -> tuple[pl.DataFrame, CanaryScoreProgress]:
    """Score catalog PGS IDs on the given canary VCFs and persist the rows.

    Does **not** recompute 1000G reference scores. Already-scored
    ``(pgs_id, sample_id)`` pairs in ``canary_scores.parquet`` are skipped
    unless ``skip_existing`` is False. Returns the rows for these samples only.
    """
    from just_prs.prs import compute_prs_duckdb
    from just_prs.prs_catalog import PRSCatalog
    from just_prs.vcf import detect_genome_build

    if not samples:
        raise ValueError("score_canary_catalog requires at least one --vcf sample.")

    catalog = PRSCatalog(cache_dir=cache_dir)
    emit = log or (lambda _msg: None)
    progress = CanaryScoreProgress()
    sample_labels = [sample.label for sample in samples]

    for sample in samples:
        build = sample.genome_build or genome_build or detect_genome_build(sample.vcf_path) or "GRCh38"
        ids = _catalog_pgs_ids(catalog, build, pgs_ids, limit)
        existing = load_canary_scores(cache_dir)
        done: set[str] = set()
        if skip_existing and existing.height:
            done = set(
                existing.filter(pl.col("sample_id") == sample.label)["pgs_id"].to_list()
            )
        todo = [pgs_id for pgs_id in ids if pgs_id not in done]
        progress.n_total += len(ids)
        progress.n_cached += len(ids) - len(todo)
        emit(
            f"Canary sample {sample.label}: {len(todo)} to score, "
            f"{len(ids) - len(todo)} cached, build={build}, vcf={sample.vcf_path}"
        )
        if not todo:
            continue

        parquet = _normalized_canary_parquet(cache_dir, sample)
        for index, pgs_id in enumerate(todo, start=1):
            try:
                result = compute_prs_duckdb(
                    vcf_path=sample.vcf_path,
                    scoring_file=pgs_id,
                    genome_build=build,
                    cache_dir=cache_dir / "scores",
                    pgs_id=pgs_id,
                    genotypes_parquet=str(parquet),
                )
                pctl = catalog.percentile_full(
                    result.score,
                    pgs_id,
                    ancestry=ancestry,
                    panel=panel,
                    weight_mass_coverage=result.weight_mass_coverage,
                    user_match_rate=result.match_rate,
                )
                upsert_canary_score_rows(
                    cache_dir,
                    pl.DataFrame([{
                        "pgs_id": pgs_id,
                        "sample_id": sample.label,
                        "percentile": pctl.percentile,
                        "z_score": pctl.z_score,
                        "match_rate": result.match_rate,
                        "score": result.score,
                    }], schema=CANARY_RESULT_SCHEMA),
                )
                progress.n_ok += 1
            except Exception as exc:
                progress.n_failed += 1
                progress.failed_ids.append(f"{sample.label}:{pgs_id}")
                emit(f"Canary score failed {sample.label} {pgs_id}: {exc}")
            if index % progress_every == 0 or index == len(todo):
                emit(
                    f"Canary sample {sample.label}: {index}/{len(todo)} scored "
                    f"({progress.n_ok} ok, {progress.n_failed} failed)"
                )

    scored = load_canary_scores(cache_dir)
    if scored.height:
        scored = scored.filter(pl.col("sample_id").is_in(sample_labels))
    return scored, progress

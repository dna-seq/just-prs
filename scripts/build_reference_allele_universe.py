#!/usr/bin/env python3
"""Build (and optionally publish) the reference-allele universe parquet.

This reproduces ``data/reference/reference_allele_universe.parquet`` on
``just-dna-seq/pgs-catalog`` — the small ``(genome_build, chrom, pos, ref,
ref_source)`` table that lets the runtime fill a scoring variant's missing
``reference_allele`` so absent loci in a variant-only WGS VCF score as
homozygous-reference (the F15 coverage lever). It is the canonical, deterministic
way to regenerate the published artifact.

What it does (all resumable; cached steps are skipped):
  ebi fingerprints -> reference_panel (~7 GB, both builds extracted)
                   -> scoring_files (~5,385 .txt.gz) -> scoring_files_parquet
                   -> reference_fasta (Ensembl GRCh38 primary assembly, ~3 GB)
                   -> reference_allele_universe  (panel .pvar tier + FASTA SNV tier)

The 3 GB FASTA and the 75M-row panel .pvar are PRECOMPUTE-ONLY inputs; the runtime
only ever reads the ~75 MB universe parquet this produces.

Memory/disk notes (learned the hard way):
  * Point the cache at a large volume via PRS_CACHE_DIR (the default ~/.cache may
    be on a small partition; the panel + FASTA + scoring caches total ~40 GB).
    This script loads .env, so PRS_CACHE_DIR there is honored.
  * The position-union over the whole catalog is ~2e9 rows -> ~3.5e7 distinct
    positions. The asset bounds memory by streaming per-file parts and aggregating
    per chromosome, so peak RSS stays ~8-9 GB. DuckDB is capped by
    PRS_DUCKDB_MEMORY_LIMIT (default below).

Usage:
    # Build locally only (no HF upload) — validate, inspect, then push separately
    uv run python scripts/build_reference_allele_universe.py

    # Build and publish to HuggingFace (requires HF_TOKEN in env or .env)
    uv run python scripts/build_reference_allele_universe.py --push

    # Tune the DuckDB memory cap (default 20GB; the per-chrom union stays well under)
    PRS_CACHE_DIR=/data/cache uv run python scripts/build_reference_allele_universe.py --duckdb-memory-limit 16GB

    # Build + publish the GRCh37 universe (writes reference_allele_universe_GRCh37.parquet).
    # Long pole: this first downloads ~5,385 hmPOS_GRCh37 scoring files + converts to parquet.
    PRS_CACHE_DIR=/data/just-dna-lite/just-prs PRS_DUCKDB_MEMORY_LIMIT=20GB \
      uv run python scripts/build_reference_allele_universe.py --genome-build GRCh37 --push
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing just_prs so PRS_CACHE_DIR (e.g. a large /data volume)
# is honored by resolve_cache_dir() / CacheDirResource. A bare `uv run python`
# does not auto-load dotenv.
load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true", help="Also upload the parquet to HuggingFace (needs HF_TOKEN).")
    parser.add_argument(
        "--genome-build",
        default="GRCh38",
        choices=["GRCh38", "GRCh37"],
        help="Genome build to build the universe for (GRCh38 is unsuffixed, GRCh37 is _GRCh37-suffixed).",
    )
    parser.add_argument("--duckdb-memory-limit", default="20GB", help="DuckDB memory cap for the union step (default 20GB).")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=float(os.environ.get("PRS_PIPELINE_MIN_COVERAGE", "0.90")),
        help="Refuse --push if scoring-parquet coverage of the catalog is below this (default 0.90).",
    )
    args = parser.parse_args()

    os.environ.setdefault("PRS_DUCKDB_MEMORY_LIMIT", args.duckdb_memory_limit)
    # Drive the build-parameterized universe lineage (scoring download, FASTA,
    # panel .pvar tier, output filename) via the env var the assets read.
    os.environ["PRS_PIPELINE_GENOME_BUILD"] = args.genome_build

    import dagster as dg
    import polars as pl

    from just_prs.hf import reference_allele_universe_filename
    from just_prs.scoring import resolve_cache_dir
    from prs_pipeline.assets import (
        ebi_reference_panel_fingerprint,
        ebi_scoring_files_fingerprint,
        reference_allele_universe,
        reference_fasta,
        reference_panel,
        scoring_files,
        scoring_files_parquet,
    )
    from prs_pipeline.resources import CacheDirResource, HuggingFaceResource

    cache = resolve_cache_dir()
    print(f"cache_dir = {cache}")
    print(f"genome_build = {args.genome_build}")
    print(f"PRS_DUCKDB_MEMORY_LIMIT = {os.environ['PRS_DUCKDB_MEMORY_LIMIT']}")

    result = dg.materialize(
        [
            ebi_reference_panel_fingerprint,
            ebi_scoring_files_fingerprint,
            reference_panel,
            scoring_files,
            scoring_files_parquet,
            reference_fasta,
            reference_allele_universe,
        ],
        resources={
            "cache_dir_resource": CacheDirResource(),
            "hf_resource": HuggingFaceResource(),
        },
    )
    if not result.success:
        print("MATERIALIZE FAILED", file=sys.stderr)
        return 1

    out = cache / "percentiles" / reference_allele_universe_filename(args.genome_build)
    df = pl.read_parquet(out)
    by_src = dict(zip(*df.group_by("ref_source").len().to_dict(as_series=False).values()))
    n_dupes = df.height - df.select("chrom", "pos").n_unique()
    bad_fasta = df.filter((pl.col("ref_source") == "fasta") & ~pl.col("ref").is_in(["A", "C", "G", "T"])).height
    resolved_null = df.filter((pl.col("ref_source") != "unresolved") & pl.col("ref").is_null()).height
    print("=" * 60)
    print(f"UNIVERSE rows={df.height:,} by_source={by_src}")
    print(f"VALIDATE dupes={n_dupes} bad_fasta={bad_fasta} resolved_null={resolved_null}")
    print(f"PATH {out}")
    print("=" * 60)
    if n_dupes or bad_fasta or resolved_null:
        print("VALIDATION FAILED — not safe to publish", file=sys.stderr)
        return 1

    # Completeness gate (robustness guarantee #4): the universe is built only from
    # scoring parquets present on disk, so a download/convert gap silently shrinks
    # it. Refuse to publish a short universe.
    from just_prs.ftp import list_all_pgs_ids
    from prs_pipeline.assets import _scoring_parquet_coverage

    n_present, n_catalog, coverage, missing = _scoring_parquet_coverage(
        cache / "scores", args.genome_build, list_all_pgs_ids()
    )
    print(
        f"COVERAGE scoring_parquets={n_present}/{n_catalog} ratio={coverage:.4f} "
        f"missing={len(missing)}{' e.g. ' + str(missing[:10]) if missing else ''}"
    )
    print("=" * 60)
    if args.push and coverage < args.min_coverage:
        print(
            f"COVERAGE {coverage:.4f} < {args.min_coverage} — refusing to publish a "
            f"short universe ({len(missing)} scores missing). Re-run to fill the gap, "
            f"or lower --min-coverage to override.",
            file=sys.stderr,
        )
        return 1

    if args.push:
        from just_prs.hf import (
            DEFAULT_HF_CATALOG_REPO,
            HF_REFERENCE_PREFIX,
            push_reference_allele_universe,
        )

        print(
            f"Uploading to {DEFAULT_HF_CATALOG_REPO}/{HF_REFERENCE_PREFIX}/"
            f"{reference_allele_universe_filename(args.genome_build)} ..."
        )
        push_reference_allele_universe(out, genome_build=args.genome_build)
        print("UPLOADED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

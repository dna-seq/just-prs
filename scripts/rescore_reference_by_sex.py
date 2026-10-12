#!/usr/bin/env python3
"""Rescore the reference panel with copy-aware sex chromosomes and build the per-sex curves.

Scores every PGS ID of the published legacy curves (``{panel}_distributions.parquet``)
against the panel with ``compute_reference_prs_batch``, which now writes three scores
per sample (legacy 0/2, copies, dosage compensation) and the sex. Outputs land in
``<cache>/percentiles/<output-subdir>/``, never over the published files:

* ``{panel}_distributions.parquet`` etc.: the batch's own outputs. The legacy curve
  rebuilt here is a regression check: for a score without PAR rows it must equal the
  published curve, since only PAR matching changed for the legacy coding.
* ``{panel}_distributions_by_sex.parquet``: the new curves (all IDs scored).
* ``{panel}_distributions_by_sex.publish.parquet``: the same, restricted to the PGS IDs
  the published legacy curves still serve (not quarantined) — what would go to HF.
* ``rescore_report.json`` and ``legacy_regression.csv``: what changed, per score.

Nothing is uploaded. The push command is printed at the end.

Resumable: per-PGS raw scores live in ``<cache>/reference_scores/{panel}/<id>/`` and a
rerun reuses every one that already has the sex-aware columns.

Usage:
    uv run python scripts/rescore_reference_by_sex.py --output-subdir sex-aware-2026-10-13
    uv run python scripts/rescore_reference_by_sex.py --pgs-ids PGS000030,PGS000006 --output-subdir trial
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import typer
from dotenv import load_dotenv

from just_prs.hf import distributions_by_sex_filename, distributions_filename, pull_reference_distributions
from just_prs.reference import compute_reference_prs_batch, reference_panel_dir
from just_prs.scoring import resolve_cache_dir

app = typer.Typer(pretty_exceptions_enable=False)


def _published_ids(percentiles_dir: Path, panel: str) -> tuple[pl.DataFrame, set[str]]:
    """The published legacy curves (pulled when missing) and the IDs they still serve."""
    path = percentiles_dir / distributions_filename(panel)
    if not path.exists():
        pull_reference_distributions(percentiles_dir, panel=panel)
    published = pl.read_parquet(path)
    issues_path = percentiles_dir / f"{panel}_distribution_quality_issues.parquet"
    quarantined: set[str] = set()
    if issues_path.exists():
        issues = pl.read_parquet(issues_path)
        quarantined = set(issues.filter(pl.col("severity") == "ERROR")["pgs_id"].to_list())
    return published, set(published["pgs_id"].unique().to_list()) - quarantined


def _per_pgs_match_counts(cache_dir: Path, panel: str, pgs_ids: list[str]) -> pl.DataFrame:
    rows = []
    for pgs_id in pgs_ids:
        path = cache_dir / "reference_scores" / panel / pgs_id / "scores.parquet"
        if not path.exists():
            continue
        first = pl.read_parquet(
            path, columns=["variants_matched", "variants_matched_par", "variants_matched_sex_linked",
                           "variants_matched_one_copy"], n_rows=1,
        )
        rows.append({"pgs_id": pgs_id, **first.row(0, named=True)})
    return pl.DataFrame(rows)


@app.command()
def main(
    output_subdir: str = typer.Option(..., help="Subdirectory of <cache>/percentiles/ for every output."),
    panel: str = typer.Option("1000g", help="Reference panel."),
    genome_build: str = typer.Option("GRCh38", help="Genome build of the panel and scoring files."),
    pgs_ids: str = typer.Option("", help="Comma-separated IDs; default: every ID of the published curves."),
    progress_every: int = typer.Option(25, help="Print a progress line every N PGS IDs."),
    ref_dir: Path = typer.Option(
        None,
        help="Panel directory; default <cache>/reference_panel/<panel>. A copy on local disk avoids "
        "rereading a network share (the panel's genotype reads are random access).",
    ),
) -> None:
    load_dotenv()
    cache_dir = resolve_cache_dir()
    percentiles_dir = cache_dir / "percentiles"
    out_dir = percentiles_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    published, served = _published_ids(percentiles_dir, panel)
    ids = [i.strip() for i in pgs_ids.split(",") if i.strip()] or sorted(published["pgs_id"].unique().to_list())
    print(f"rescoring {len(ids)} PGS IDs on {panel} {genome_build} -> {out_dir}", flush=True)

    started = time.monotonic()

    def progress(event: dict) -> None:
        print(
            f"{event['processed']}/{event['total']} PGS ({100 * event['processed'] / max(event['total'], 1):.1f}%), "
            f"last {event['last_pgs_id']} {event['last_status']}, ok {event['ok']} failed {event['failed']} "
            f"cached {event['cached']}, {time.monotonic() - started:.0f}s, "
            f"eta {event['eta_sec'] or 0:.0f}s",
            flush=True,
        )

    result = compute_reference_prs_batch(
        pgs_ids=ids,
        ref_dir=ref_dir or reference_panel_dir(cache_dir, panel),
        cache_dir=cache_dir,
        genome_build=genome_build,
        panel=panel,
        skip_existing=True,
        output_subdir=output_subdir,
        progress_callback=progress,
        progress_every=progress_every,
    )

    by_sex = result.distributions_by_sex_df
    assert by_sex is not None
    publish = by_sex.filter(pl.col("pgs_id").is_in(list(served)))
    publish_path = out_dir / distributions_by_sex_filename(panel).replace(".parquet", ".publish.parquet")
    publish.write_parquet(publish_path)

    # Regression: the legacy coding rebuilt here against the published curves.
    counts = _per_pgs_match_counts(cache_dir, panel, ids)
    regression = (
        result.distributions_df.select("pgs_id", "superpopulation", "mean", "std", "n")
        .join(
            published.select("pgs_id", "superpopulation", pl.col("mean").alias("published_mean"),
                             pl.col("std").alias("published_std"), pl.col("n").alias("published_n")),
            on=["pgs_id", "superpopulation"],
        )
        .join(counts, on="pgs_id", how="left")
        .with_columns(
            ((pl.col("mean") - pl.col("published_mean")).abs()
             / pl.when(pl.col("published_std") > 0).then(pl.col("published_std")).otherwise(None))
            .alias("mean_shift_sd"),
        )
    )
    regression.write_csv(out_dir / "legacy_regression.csv")
    no_par = regression.filter(pl.col("variants_matched_par") == 0)

    # What the per-sex curves change: a man's copies curve against the legacy curve.
    shift = (
        by_sex.filter((pl.col("sex") == "male") & (pl.col("x_coding") == "copies"))
        .join(published.select("pgs_id", "superpopulation", pl.col("mean").alias("legacy_mean"),
                               pl.col("std").alias("legacy_std")), on=["pgs_id", "superpopulation"])
        .with_columns(((pl.col("mean") - pl.col("legacy_mean")) / pl.col("legacy_std")).alias("male_shift_sd"))
    )

    outcomes = result.quality_df
    report = {
        "panel": panel,
        "genome_build": genome_build,
        "n_ids_requested": len(ids),
        "n_ok": int(outcomes.filter(pl.col("status") == "ok").height),
        "n_failed": int(outcomes.filter(pl.col("status") == "failed").height),
        "failed_examples": outcomes.filter(pl.col("status") == "failed").select("pgs_id", "error").head(10).to_dicts(),
        "elapsed_sec": round(time.monotonic() - started, 1),
        "by_sex_rows": by_sex.height,
        "by_sex_ids": int(by_sex["pgs_id"].n_unique()),
        "by_sex_ids_split_by_sex": int(by_sex.filter(pl.col("sex") == "male")["pgs_id"].n_unique()),
        "publish_ids": int(publish["pgs_id"].n_unique()),
        "ids_with_par_rows": int(counts.filter(pl.col("variants_matched_par") > 0).height) if counts.height else 0,
        "legacy_regression_no_par_rows": no_par.height,
        "legacy_regression_no_par_max_shift_sd": no_par["mean_shift_sd"].max(),
        "legacy_regression_no_par_n_shift_over_1e-6_sd": int(no_par.filter(pl.col("mean_shift_sd") > 1e-6).height),
        "male_copies_shift_sd_median": shift["male_shift_sd"].median(),
        "male_copies_shift_sd_max_abs": shift["male_shift_sd"].abs().max(),
    }
    (out_dir / "rescore_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(
        "\nNot uploaded. To publish the per-sex curves next to the legacy ones:\n"
        f"  uv run python -c \"from pathlib import Path; from just_prs.hf import push_reference_distributions_by_sex; "
        f"push_reference_distributions_by_sex(Path('{publish_path}'), panel='{panel}')\"",
        flush=True,
    )


if __name__ == "__main__":
    app()

"""Attribute the remaining just-prs vs PLINK2 difference of an engine benchmark run.

just-prs side: the engine's own per-variant output (``extract_prs_contributions``
on the same autosomal PASS parquet the benchmark scored).
PLINK2 side: an emulation per scoring row. Each row becomes the ``CHROM:POS:A:B``
IDs ``_prepare_plink2_score_input`` writes (both orders, every other-allele
candidate); an ID present in the PASS pgen's ``.pvar`` contributes
``weight x count of the effect allele in the record's GT``.

Both sides are summed per ``(chrom, pos, effect_allele)`` and every non-zero
difference gets a cause. Whatever is left per score after the variant sums is
PLINK2's six-significant-digit ``.sscore`` rounding.

Usage:
    uv run python just-prs/benchmarks/attribute_residual.py \\
        --run just-prs/benchmarks/results/2026-10-10 \\
        --genotypes <cache>/benchmark/vcf/antonkulaga_normalized_autosomes.parquet \\
        --pvar <cache>/benchmark/pgen/benchmark_pgen_pass.pvar
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import typer

from just_prs.contributions import extract_prs_contributions
from just_prs.scoring import ensure_scoring_file, parse_scoring_file, resolve_cache_dir

app = typer.Typer(pretty_exceptions_enable=False)

KEY = ["chrom", "pos", "effect_allele"]


def _alleles(ref: pl.Expr, alt: pl.Expr) -> pl.Expr:
    return pl.concat_list([ref]).list.concat(alt.str.replace_all(",", "|", literal=True).str.split("|"))


def _gt_dose(alleles: pl.Expr, gt: pl.Expr, effect: pl.Expr) -> pl.Expr:
    parts = gt.str.replace_all(r"\|", "/").str.split("/")
    i1 = parts.list.get(0, null_on_oob=True).cast(pl.Int64, strict=False)
    i2 = pl.when(parts.list.len() == 1).then(i1).otherwise(
        parts.list.get(1, null_on_oob=True).cast(pl.Int64, strict=False)
    )
    a1 = alleles.list.get(i1, null_on_oob=True)
    a2 = alleles.list.get(i2, null_on_oob=True)
    return (a1 == effect).cast(pl.Int64).fill_null(0) + (a2 == effect).cast(pl.Int64).fill_null(0)


def plink2_emulation(scoring: pl.DataFrame, pvar_geno: pl.DataFrame) -> pl.DataFrame:
    """Per (chrom, pos, effect_allele): PLINK2's summed contribution and whether any line matched."""
    other: pl.Expr = pl.lit(None, dtype=pl.Utf8)
    for col in ("reference_allele", "hm_inferOtherAllele", "other_allele"):
        if col in scoring.columns:
            value = pl.col(col).cast(pl.Utf8)
            other = pl.when(value.str.len_chars() > 0).then(value).otherwise(other)
    rows = (
        scoring.filter(
            pl.col("effect_allele").is_not_null()
            & pl.col("effect_weight").is_not_null()
            & pl.col("hm_chr").is_not_null()
            & pl.col("hm_pos").is_not_null()
        )
        .select(
            pl.col("hm_chr").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
            pl.col("hm_pos").cast(pl.Int64).alias("pos"),
            pl.col("effect_allele").cast(pl.Utf8),
            pl.col("effect_weight").cast(pl.Float64),
            other.str.split("/").alias("_other"),
        )
        .with_row_index("_row")
    )
    lines = (
        rows.explode("_other", empty_as_null=True)
        .filter(pl.col("_other").is_not_null() & (pl.col("_other") != pl.col("effect_allele")))
        .with_columns(prefix=pl.col("chrom") + ":" + pl.col("pos").cast(pl.Utf8) + ":")
    )
    ids = pl.concat([
        lines.select("_row", (pl.col("prefix") + pl.col("_other") + ":" + pl.col("effect_allele")).alias("pvar_id")),
        lines.select("_row", (pl.col("prefix") + pl.col("effect_allele") + ":" + pl.col("_other")).alias("pvar_id")),
    ]).unique()
    matched = ids.join(pvar_geno, on="pvar_id", how="inner").join(
        rows.select("_row", "effect_allele", "effect_weight"), on="_row"
    ).with_columns(
        _gt_dose(_alleles(pl.col("pvar_ref"), pl.col("pvar_alt")), pl.col("GT"), pl.col("effect_allele")).alias("pl_dose")
    )
    per_row = matched.group_by("_row").agg(
        (pl.col("pl_dose") * pl.col("effect_weight")).sum().alias("pl_contrib"),
    )
    return (
        rows.join(per_row, on="_row", how="left")
        .with_columns(pl.col("pl_contrib").is_not_null().alias("pl_matched"), pl.col("pl_contrib").fill_null(0.0))
        .group_by(KEY)
        .agg(pl.col("pl_contrib").sum(), pl.col("pl_matched").any(), pl.col("_other").first().alias("other_candidates"))
    )


def classify(diff: pl.DataFrame) -> pl.DataFrame:
    alts = pl.col("vcf_alt").str.replace_all(",", "|", literal=True).str.split("|")
    is_snv_record = (pl.col("vcf_ref").str.len_chars() == 1) & alts.list.eval(pl.element().str.len_chars() == 1).list.all()
    others = pl.col("other_candidates").fill_null(pl.lit([], dtype=pl.List(pl.Utf8)))
    no_other = others.list.len() == 0
    later_alt = alts.list.len() > 1
    cause = (
        pl.when(pl.col("vcf_ref").is_null()).then(pl.lit("no_record_at_position"))
        .when(no_other).then(pl.lit("no_other_allele_plink2_cannot_build_id"))
        .when(later_alt).then(pl.lit("multiallelic_effect_or_other_is_a_later_alt"))
        .when(is_snv_record & ~pl.col("pl_matched"))
        .then(pl.lit("third_allele_snv_just_prs_counts_the_called_copy"))
        .when(~is_snv_record & (pl.col("status") == "observed") & ~pl.col("pl_matched"))
        .then(pl.lit("indel_allele_set_matched_by_just_prs_only"))
        .when(pl.col("status") == "allele_mismatch").then(pl.lit("allele_mismatch_in_just_prs_matched_in_plink2"))
        .otherwise(pl.lit("other"))
    )
    return diff.with_columns(cause.alias("cause"))


@app.command()
def main(
    run: Path = typer.Option(..., help="Benchmark output directory (has pgs_ids.txt, benchmark_per_pgs.csv)."),
    genotypes: Path = typer.Option(..., help="The autosomal PASS parquet the benchmark scored."),
    pvar: Path = typer.Option(..., help="The PASS-only pgen's .pvar."),
    scores_dir: Path = typer.Option(None, help="Scoring-file cache (default <cache>/scores)."),
) -> None:
    scores_dir = scores_dir or resolve_cache_dir() / "scores"
    pgs_ids = [line.strip() for line in (run / "pgs_ids.txt").read_text().splitlines() if line.strip()]
    per_pgs = pl.read_csv(run / "benchmark_per_pgs.csv")

    geno = pl.read_parquet(genotypes, columns=["chrom", "pos", "ref", "alt", "GT"])
    pvar_df = pl.read_csv(
        pvar, separator="\t", comment_prefix="##",
        schema_overrides={"#CHROM": pl.Utf8, "POS": pl.Int64},
        columns=["#CHROM", "POS", "ID", "REF", "ALT"],
    ).rename({"#CHROM": "chrom", "POS": "pos", "ID": "pvar_id", "REF": "pvar_ref", "ALT": "pvar_alt"})
    pvar_geno = pvar_df.join(geno.select("chrom", "pos", "GT"), on=["chrom", "pos"], how="left")
    if pvar_geno["GT"].null_count():
        raise ValueError(f"{pvar_geno['GT'].null_count()} pvar records have no PASS genotype")
    pvar_geno = pvar_geno.select("pvar_id", "pvar_ref", "pvar_alt", "GT")

    variant_rows: list[pl.DataFrame] = []
    score_rows: list[dict[str, object]] = []
    for pgs_id in pgs_ids:
        path = ensure_scoring_file(pgs_id, scores_dir, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as td:
            contrib = extract_prs_contributions(
                vcf_path="__unused__", scoring_file=path, pgs_id=pgs_id,
                genotypes_parquet=genotypes, output=Path(td) / "c.parquet",
            )
            jp = (
                pl.read_parquet(contrib.output_path)
                .group_by(KEY)
                .agg(pl.col("contribution").fill_null(0.0).sum().alias("jp_contrib"), pl.col("status").first())
            )
        pl_side = plink2_emulation(parse_scoring_file(path).collect(), pvar_geno)
        both = jp.join(pl_side, on=KEY, how="full", coalesce=True).with_columns(
            pl.col("jp_contrib").fill_null(0.0), pl.col("pl_contrib").fill_null(0.0),
            pl.col("pl_matched").fill_null(False),
        ).with_columns((pl.col("jp_contrib") - pl.col("pl_contrib")).alias("delta"))
        nz = both.filter(pl.col("delta").abs() > 1e-12).join(
            geno.select("chrom", "pos", pl.col("ref").alias("vcf_ref"), pl.col("alt").alias("vcf_alt"), pl.col("GT").alias("vcf_GT")),
            on=["chrom", "pos"], how="left",
        )
        nz = classify(nz).with_columns(pl.lit(pgs_id).alias("pgs_id"))
        variant_rows.append(nz.drop("other_candidates"))
        score_rows.append({
            "pgs_id": pgs_id,
            "jp_variant_sum": float(both["jp_contrib"].sum()),
            "pl_emulated_sum": float(both["pl_contrib"].sum()),
            "variant_delta_sum": float(both["delta"].sum()),
            "n_nonzero_variants": nz.height,
        })
        print(pgs_id, score_rows[-1], flush=True)

    variants = pl.concat(variant_rows, how="diagonal_relaxed")
    scores = pl.DataFrame(score_rows).join(
        per_pgs.select("pgs_id", "score_polars", "score_duckdb", "score_plink2"), on="pgs_id", how="left"
    ).with_columns(
        (pl.col("score_polars") - pl.col("jp_variant_sum")).alias("jp_engine_vs_variant_sum"),
        (pl.col("score_plink2") - pl.col("pl_emulated_sum")).alias("plink2_print_rounding"),
        (pl.col("score_polars") - pl.col("score_plink2")).alias("delta_polars_plink2"),
    )
    by_cause = (
        variants.filter(pl.col("pgs_id").is_in(scores.filter(pl.col("score_plink2").is_not_null())["pgs_id"].implode()))
        .group_by("cause")
        .agg(
            pl.col("pgs_id").n_unique().alias("n_pgs"),
            pl.len().alias("n_variants"),
            pl.col("delta").sum().alias("net_delta"),
            pl.col("delta").abs().sum().alias("abs_delta"),
        )
        .with_columns((pl.col("abs_delta") / pl.col("abs_delta").sum()).alias("share_abs"))
        .sort("abs_delta", descending=True)
    )
    variants.write_csv(run / "residual_per_variant.csv")
    scores.write_csv(run / "residual_per_pgs.csv")
    by_cause.write_csv(run / "residual_by_cause.csv")
    print(by_cause)
    print(scores.select(
        pl.col("jp_engine_vs_variant_sum").abs().max().alias("max|engine - variant sum|"),
        pl.col("plink2_print_rounding").abs().max().alias("max|plink2 print rounding|"),
    ))


if __name__ == "__main__":
    app()

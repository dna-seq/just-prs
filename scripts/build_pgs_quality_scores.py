#!/usr/bin/env python3
"""Build comprehensive per-PGS quality scoring parquet.

Combines synthetic (metadata-only) scores with practical signals from
smoke test results across all available genomes.

Usage:
    uv run python scripts/build_pgs_quality_scores.py
    uv run python scripts/build_pgs_quality_scores.py --public-only
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import polars as pl

from just_prs import PRSCatalog
from just_prs.quality import synthetic_quality_score

RESULTS_PATH = Path("data/output/smoke_test/all_results.parquet")
PUBLIC_GENOMES = ["antonkulaga", "livia_hard-filtered_vcf"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-PGS quality scoring parquet.")
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--output", type=str, default="data/output/pgs_quality_scores.parquet")
    args = parser.parse_args()

    results = pl.read_parquet(RESULTS_PATH).filter(pl.col("error").is_null())
    if args.public_only:
        results = results.filter(pl.col("genome").is_in(PUBLIC_GENOMES))

    pgs_ids = sorted(results["pgs_id"].unique().to_list())
    n_genomes = results["genome"].n_unique()
    print(f"Building quality scores for {len(pgs_ids)} PGS IDs across {n_genomes} genomes")

    catalog = PRSCatalog()
    best_perf = catalog.best_performance().collect()
    scores_meta = catalog.scores(genome_build="GRCh38").collect()

    # --- Synthetic (theoretical) scoring per PGS ---
    synth_rows = []
    for pgs_id in pgs_ids:
        perf = best_perf.filter(pl.col("pgs_id") == pgs_id)
        kwargs: dict = {}
        tier = "T3_none"
        auroc_val = cindex_val = or_val = hr_val = beta_val = n_ind = None

        if perf.height > 0:
            p = perf.row(0, named=True)
            auroc_val = p.get("auroc_estimate")
            cindex_val = p.get("cindex_estimate")
            or_val = p.get("or_estimate")
            hr_val = p.get("hr_estimate")
            beta_val = p.get("beta_estimate")
            n_ind = p.get("n_individuals")

            kwargs = {
                "auroc": auroc_val, "cindex": cindex_val,
                "or_estimate": or_val, "hr_estimate": hr_val,
                "beta_estimate": beta_val, "n_individuals": n_ind,
            }

            if auroc_val is not None or cindex_val is not None:
                tier = "T1a_auroc"
            elif beta_val is not None:
                tier = "T1b_beta"
            elif or_val is not None or hr_val is not None:
                tier = "T2_or_hr"

        synth_rows.append({
            "pgs_id": pgs_id,
            "synthetic_score": synthetic_quality_score(**kwargs),
            "tier": tier,
            "auroc": auroc_val,
            "cindex": cindex_val,
            "or_estimate": or_val,
            "hr_estimate": hr_val,
            "beta_estimate": beta_val,
            "n_individuals": n_ind,
        })

    synth_df = pl.DataFrame(synth_rows, infer_schema_length=len(synth_rows))

    # --- Practical signals aggregated across genomes ---
    practical = results.group_by("pgs_id").agg([
        pl.col("trait").first(),
        pl.col("match_rate").mean().alias("match_rate_mean"),
        pl.col("match_rate").std().alias("match_rate_std"),
        pl.col("match_rate").min().alias("match_rate_min"),
        pl.col("match_rate").max().alias("match_rate_max"),
        pl.col("variants_matched").first(),
        pl.col("variants_total").first(),
        pl.col("pct_EUR").mean().alias("pct_eur_mean"),
        pl.col("pct_EUR").std().alias("pct_eur_std"),
        pl.col("pct_EUR").min().alias("pct_eur_min"),
        pl.col("pct_EUR").max().alias("pct_eur_max"),
        pl.col("pct_AFR").mean().alias("pct_afr_mean"),
        pl.col("pct_AMR").mean().alias("pct_amr_mean"),
        pl.col("pct_EAS").mean().alias("pct_eas_mean"),
        pl.col("pct_SAS").mean().alias("pct_sas_mean"),
        pl.col("absolute_risk_percent").mean().alias("abs_risk_mean"),
        pl.col("absolute_risk_percent").std().alias("abs_risk_std"),
        pl.col("population_average_percent").first().alias("pop_avg_risk"),
        pl.col("risk_ratio_value").mean().alias("risk_ratio_mean"),
        pl.col("risk_ratio_value").std().alias("risk_ratio_std"),
        pl.col("score").mean().alias("raw_score_mean"),
        pl.col("score").std().alias("raw_score_std"),
        pl.col("has_allele_frequencies").first(),
        pl.col("quality_label").first(),
        pl.col("percentile_method").first(),
        pl.col("heritability").first(),
        pl.col("absolute_risk_method").first(),
        pl.len().alias("n_genomes"),
    ])

    # Practical match score: mean * (1 - cv)  — match_rate is already 0-100
    practical = practical.with_columns(
        (
            pl.col("match_rate_mean")
            * (1.0 - (pl.col("match_rate_std") / pl.col("match_rate_mean").clip(lower_bound=0.01)).clip(upper_bound=0.5))
        ).alias("practical_match_score"),
    )

    # Stability score: (100 - pct_std) * spread_factor
    practical = practical.with_columns(
        (pl.col("pct_eur_max") - pl.col("pct_eur_min")).alias("pct_eur_range"),
    ).with_columns(
        pl.when(pl.col("pct_eur_range") >= 20)
        .then(1.0)
        .otherwise((pl.col("pct_eur_range") / 20.0).clip(lower_bound=0.1))
        .alias("spread_factor"),
    ).with_columns(
        ((100.0 - pl.col("pct_eur_std").fill_null(50.0)) * pl.col("spread_factor")).alias("stability_score"),
    )

    # Risk concordance: bell curve centered on 0.3 relative deviation
    practical = practical.with_columns(
        pl.when(
            pl.col("abs_risk_mean").is_not_null()
            & pl.col("pop_avg_risk").is_not_null()
            & (pl.col("pop_avg_risk") > 0)
        ).then(
            ((pl.col("abs_risk_mean") - pl.col("pop_avg_risk")).abs() / pl.col("pop_avg_risk"))
        ).otherwise(None).alias("mean_rel_deviation"),
    ).with_columns(
        pl.when(pl.col("mean_rel_deviation").is_not_null())
        .then(
            100.0 * (1.0 - ((pl.col("mean_rel_deviation") - 0.3).abs() / 1.0).clip(upper_bound=1.0))
            * (1.0 - (pl.col("abs_risk_std").fill_null(0.0)
                       / pl.col("abs_risk_mean").clip(lower_bound=0.001) / 1.0).clip(upper_bound=0.5))
        ).otherwise(None).alias("risk_concordance_score"),
    )

    # --- EFO trait grouping ---
    efo_map = scores_meta.filter(
        pl.col("pgs_id").is_in(pgs_ids)
    ).select("pgs_id", "trait_reported", "trait_efo_id").unique()

    canonical = (
        efo_map.group_by("trait_efo_id", "trait_reported")
        .len()
        .sort("len", descending=True)
        .group_by("trait_efo_id")
        .first()
        .select("trait_efo_id", pl.col("trait_reported").alias("canonical_trait"))
    )
    resolved = efo_map.join(canonical, on="trait_efo_id", how="left").select(
        "pgs_id",
        pl.coalesce("canonical_trait", "trait_reported").alias("trait_grouped"),
        "trait_efo_id",
    )

    # --- Join ---
    combined = (
        synth_df
        .join(practical, on="pgs_id", how="left")
        .join(resolved, on="pgs_id", how="left")
    )

    # --- Normalize and compute final combined score ---
    def _norm(s: pl.Series) -> pl.Series:
        mn, mx = s.min(), s.max()
        if mn is None or mx is None or mn == mx:
            return pl.Series(s.name, [50.0] * s.len())
        return ((s - mn) / (mx - mn) * 100.0).alias(s.name)

    combined = combined.with_columns(
        _norm(combined["synthetic_score"]).alias("norm_synthetic"),
        _norm(combined["practical_match_score"].fill_null(0.0)).alias("norm_match"),
        _norm(combined["stability_score"].fill_null(0.0)).alias("norm_stability"),
        _norm(combined["risk_concordance_score"].fill_null(0.0)).alias("norm_risk"),
    )

    combined = combined.with_columns(
        (
            pl.col("norm_synthetic") * 0.40
            + pl.col("norm_match") * 0.25
            + pl.col("norm_stability") * 0.15
            + pl.col("norm_risk") * 0.20
        ).alias("combined_quality_score"),
    ).sort("combined_quality_score", descending=True)

    # --- Select final columns ---
    output = combined.select([
        "pgs_id",
        "trait",
        "trait_grouped",
        "trait_efo_id",
        "synthetic_score",
        "tier",
        "auroc",
        "cindex",
        "or_estimate",
        "hr_estimate",
        "beta_estimate",
        "n_individuals",
        "match_rate_mean",
        "match_rate_std",
        "match_rate_min",
        "match_rate_max",
        "variants_matched",
        "variants_total",
        "practical_match_score",
        "pct_eur_mean",
        "pct_eur_std",
        "pct_eur_min",
        "pct_eur_max",
        "pct_afr_mean",
        "pct_amr_mean",
        "pct_eas_mean",
        "pct_sas_mean",
        "stability_score",
        "abs_risk_mean",
        "abs_risk_std",
        "pop_avg_risk",
        "risk_ratio_mean",
        "risk_ratio_std",
        "mean_rel_deviation",
        "risk_concordance_score",
        "norm_synthetic",
        "norm_match",
        "norm_stability",
        "norm_risk",
        "combined_quality_score",
        "quality_label",
        "has_allele_frequencies",
        "percentile_method",
        "heritability",
        "absolute_risk_method",
        "n_genomes",
    ])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(out_path)

    print(f"\nWritten {output.height} PGS quality scores to {out_path}")
    print(f"Columns: {len(output.columns)}")

    print(f"\nScore distribution:")
    print(f"  Combined:  mean={output['combined_quality_score'].mean():.1f}, "
          f"median={output['combined_quality_score'].median():.1f}, "
          f"min={output['combined_quality_score'].min():.1f}, "
          f"max={output['combined_quality_score'].max():.1f}")
    print(f"  Synthetic: mean={output['synthetic_score'].mean():.1f}, "
          f"median={output['synthetic_score'].median():.1f}")

    print(f"\nBy tier:")
    for tier in ["T1a_auroc", "T1b_beta", "T2_or_hr", "T3_none"]:
        sub = output.filter(pl.col("tier") == tier)
        if sub.height > 0:
            print(f"  {tier:12s}: n={sub.height:3d}, "
                  f"combined={sub['combined_quality_score'].median():.1f}, "
                  f"synthetic={sub['synthetic_score'].median():.1f}")

    print(f"\nTop 15:")
    with pl.Config(tbl_rows=15, tbl_width_chars=130, fmt_str_lengths=30):
        print(output.head(15).select(
            "pgs_id", "trait_grouped", "combined_quality_score",
            "synthetic_score", "practical_match_score",
            "stability_score", "risk_concordance_score", "tier",
        ))


if __name__ == "__main__":
    main()

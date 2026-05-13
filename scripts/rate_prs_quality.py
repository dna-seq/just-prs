#!/usr/bin/env python3
"""Rate PRS quality across the catalog using smoke test results.

Produces 5 ratings plus a trait-level demo ranking:

1. **Model Quality** — synthetic_quality_score from catalog metadata only
2. **Practical Match** — actual genotype match rates across real genomes
3. **Percentile Stability** — cross-genome concordance of EUR percentiles
4. **Risk Concordance** — deviation of individual absolute risk from population average
5. **Combined Practical** — weighted blend of ratings 2-4
6. **Trait Demo Ranking** — best traits for showcasing, weighted by quality, match,
   stability, and PGS count (penalizing both too few and too many)

Usage:
    uv run python scripts/rate_prs_quality.py                    # all genomes
    uv run python scripts/rate_prs_quality.py --public-only      # anton + livia only
    uv run python scripts/rate_prs_quality.py --top 30
    uv run python scripts/rate_prs_quality.py --output data/output/ratings.parquet
"""

import argparse
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import polars as pl

from just_prs import PRSCatalog
from just_prs.quality import synthetic_quality_score

RESULTS_PATH = Path("data/output/smoke_test/all_results.parquet")
PUBLIC_GENOMES = ["antonkulaga", "livia_hard-filtered_vcf"]


def load_results(public_only: bool = False) -> pl.DataFrame:
    if not RESULTS_PATH.exists():
        print(f"No results at {RESULTS_PATH} — run smoke_test_all_prs.py first.")
        sys.exit(1)
    df = pl.read_parquet(RESULTS_PATH).filter(pl.col("error").is_null())
    if public_only:
        df = df.filter(pl.col("genome").is_in(PUBLIC_GENOMES))
        label = "PUBLIC-ONLY"
    else:
        label = "ALL GENOMES"
    print(f"[{label}] Loaded {df.height} rows, {df['pgs_id'].n_unique()} PGS IDs, "
          f"{df['genome'].n_unique()} genomes, {df['trait'].n_unique()} traits")
    print(f"Genomes: {', '.join(sorted(df['genome'].unique().to_list()))}")
    return df


def rating_1_model_quality(catalog: PRSCatalog, pgs_ids: list[str]) -> pl.DataFrame:
    """Rating 1: synthetic quality score from catalog metadata only (no real genomes)."""
    best_perf = catalog.best_performance().collect()

    rows: list[dict] = []
    for pgs_id in pgs_ids:
        perf = best_perf.filter(pl.col("pgs_id") == pgs_id)
        kwargs: dict = {}
        if perf.height > 0:
            p = perf.row(0, named=True)
            kwargs["auroc"] = p.get("auroc_estimate")
            kwargs["cindex"] = p.get("cindex_estimate")
            kwargs["or_estimate"] = p.get("or_estimate")
            kwargs["hr_estimate"] = p.get("hr_estimate")
            kwargs["beta_estimate"] = p.get("beta_estimate")
            kwargs["n_individuals"] = p.get("n_individuals")
        score = synthetic_quality_score(**kwargs)
        rows.append({"pgs_id": pgs_id, "model_quality": score})

    return pl.DataFrame(rows).sort("model_quality", descending=True)


def rating_2_practical_match(df: pl.DataFrame) -> pl.DataFrame:
    """Rating 2: actual match rates across genomes.

    Computes mean match rate, min match rate (worst genome), and a practical
    match score that penalizes high variance.
    """
    stats = df.group_by("pgs_id", "trait").agg([
        pl.col("match_rate").mean().alias("match_mean"),
        pl.col("match_rate").min().alias("match_min"),
        pl.col("match_rate").max().alias("match_max"),
        pl.col("match_rate").std().alias("match_std"),
        pl.col("variants_total").first().alias("variants_total"),
    ])

    # Practical match score: mean * (1 - cv), where cv = std/mean, clamped
    return stats.with_columns(
        (
            pl.col("match_mean")
            * (1.0 - (pl.col("match_std") / pl.col("match_mean").clip(lower_bound=1.0)).clip(upper_bound=0.5))
        ).alias("practical_match_score"),
    ).sort("practical_match_score", descending=True)


def rating_3_percentile_stability(df: pl.DataFrame) -> pl.DataFrame:
    """Rating 3: cross-genome concordance of EUR percentiles.

    A good PRS should place different people at reasonably spread percentiles
    (not everyone at 0 or 100), but each person's percentile should make sense.
    We measure stability as low cross-genome standard deviation for each PGS,
    BUT we also penalize PGS IDs where all genomes collapse to the same extreme.
    """
    stats = df.group_by("pgs_id", "trait").agg([
        pl.col("pct_EUR").mean().alias("pct_eur_mean"),
        pl.col("pct_EUR").std().alias("pct_eur_std"),
        pl.col("pct_EUR").min().alias("pct_eur_min"),
        pl.col("pct_EUR").max().alias("pct_eur_max"),
        (pl.col("pct_EUR").max() - pl.col("pct_EUR").min()).alias("pct_eur_range"),
    ])

    # Stability score:
    # - high std = genomes disagree wildly = unstable = lower score
    # - all at 0 or all at 100 = degenerate = also lower score
    # Formula: (100 - std) * spread_factor
    # spread_factor: 1.0 when range >= 20 (healthy spread), reduced when degenerate
    return stats.with_columns(
        pl.when(pl.col("pct_eur_range") >= 20)
        .then(1.0)
        .otherwise((pl.col("pct_eur_range") / 20.0).clip(lower_bound=0.1))
        .alias("spread_factor"),
    ).with_columns(
        ((100.0 - pl.col("pct_eur_std")) * pl.col("spread_factor")).alias("stability_score"),
    ).sort("stability_score", descending=True)


def rating_4_risk_concordance(df: pl.DataFrame) -> pl.DataFrame:
    """Rating 4: deviation of individual absolute risk from population average.

    Measures how much individual risk estimates vary around the population mean.
    Low deviation = all genomes get similar risk ≈ population average = less informative.
    High deviation = the PRS actually discriminates between genomes.

    We want moderate deviation (the PRS discriminates) but not extreme deviation
    (which suggests instability). We also penalize PGS IDs without risk data.
    """
    risk_df = df.filter(
        pl.col("absolute_risk_percent").is_not_null()
        & pl.col("population_average_percent").is_not_null()
        & (pl.col("population_average_percent") > 0)
    )

    if risk_df.height == 0:
        return pl.DataFrame({"pgs_id": [], "trait": [], "risk_concordance_score": []})

    stats = risk_df.with_columns(
        ((pl.col("absolute_risk_percent") - pl.col("population_average_percent")).abs()
         / pl.col("population_average_percent")).alias("rel_deviation"),
    ).group_by("pgs_id", "trait").agg([
        pl.col("rel_deviation").mean().alias("mean_rel_deviation"),
        pl.col("rel_deviation").std().alias("std_rel_deviation"),
        pl.col("risk_ratio_value").mean().alias("mean_risk_ratio"),
        pl.col("risk_ratio_value").std().alias("risk_ratio_std"),
        pl.col("population_average_percent").first().alias("pop_avg_pct"),
        pl.len().alias("n_genomes_with_risk"),
    ])

    # Concordance score:
    # - Moderate mean_rel_deviation (0.1-0.5) = good discrimination = high score
    # - Low std_rel_deviation = consistent discrimination = bonus
    # Use a bell-curve centered on 0.3 relative deviation
    return stats.with_columns(
        (
            100.0
            * (1.0 - ((pl.col("mean_rel_deviation") - 0.3).abs() / 1.0).clip(upper_bound=1.0))
            * (1.0 - (pl.col("std_rel_deviation").fill_null(0.0) / 1.0).clip(upper_bound=0.5))
        ).alias("risk_concordance_score"),
    ).sort("risk_concordance_score", descending=True)


def combined_rating(
    r1: pl.DataFrame,
    r2: pl.DataFrame,
    r3: pl.DataFrame,
    r4: pl.DataFrame,
) -> pl.DataFrame:
    """Rating 5: weighted combination of all practical ratings.

    Weights: model_quality 0.40, practical_match 0.25, stability 0.15, risk 0.20
    Each component is normalized to 0-100 before blending.
    """
    def _normalize(series: pl.Series) -> pl.Series:
        mn, mx = series.min(), series.max()
        if mn == mx or mn is None or mx is None:
            return pl.Series(series.name, [50.0] * series.len())
        return ((series - mn) / (mx - mn) * 100.0).alias(series.name)

    base = r1.select("pgs_id", "model_quality")

    r2_norm = r2.select(
        "pgs_id",
        _normalize(r2["practical_match_score"]).alias("norm_match"),
    )
    r3_norm = r3.select(
        "pgs_id",
        _normalize(r3["stability_score"]).alias("norm_stability"),
    )
    r4_norm = r4.select(
        "pgs_id",
        _normalize(r4["risk_concordance_score"]).alias("norm_risk"),
    )

    joined = (
        base
        .join(r2_norm, on="pgs_id", how="left")
        .join(r3_norm, on="pgs_id", how="left")
        .join(r4_norm, on="pgs_id", how="left")
    ).with_columns(
        pl.col("norm_match").fill_null(0.0),
        pl.col("norm_stability").fill_null(0.0),
        pl.col("norm_risk").fill_null(0.0),
    )

    return joined.with_columns(
        (
            pl.col("model_quality") * 0.40
            + pl.col("norm_match") * 0.25
            + pl.col("norm_stability") * 0.15
            + pl.col("norm_risk") * 0.20
        ).alias("combined_score"),
    ).sort("combined_score", descending=True)


def _build_efo_trait_map(catalog: "PRSCatalog", pgs_ids: list[str]) -> dict[str, str]:
    """Map PGS IDs to canonical trait names by EFO/MONDO ID.

    PGS Catalog uses free-text trait_reported which fragments identical traits
    into variants like "Type 2 diabetes (T2D)" vs "(PheCode 250.2)" vs
    "Type 2 Diabetes Mellitus" — all sharing MONDO_0005148. This function
    resolves each PGS to a canonical trait name (the most common variant per
    EFO ID), so the demo ranking correctly groups them.
    """
    scores = catalog.scores(genome_build="GRCh38").collect()
    efo_map = scores.filter(
        pl.col("pgs_id").is_in(pgs_ids)
    ).select("pgs_id", "trait_reported", "trait_efo_id").unique()

    # For each EFO ID, pick the most common trait_reported as canonical name
    canonical = (
        efo_map.group_by("trait_efo_id", "trait_reported")
        .len()
        .sort("len", descending=True)
        .group_by("trait_efo_id")
        .first()
        .select("trait_efo_id", pl.col("trait_reported").alias("canonical_trait"))
    )
    resolved = efo_map.join(canonical, on="trait_efo_id", how="left").with_columns(
        pl.coalesce("canonical_trait", "trait_reported").alias("grouped_trait"),
    )
    return dict(zip(resolved["pgs_id"].to_list(), resolved["grouped_trait"].to_list()))


def trait_demo_ranking(
    r1: pl.DataFrame,
    r2: pl.DataFrame,
    r3: pl.DataFrame,
    r4: pl.DataFrame,
    df: pl.DataFrame,
    catalog: "PRSCatalog",
    min_pgs: int = 3,
) -> pl.DataFrame:
    """Trait-level demo ranking: best traits for showcasing PRS results.

    Traits are grouped by EFO/MONDO ontology ID so that "Type 2 diabetes
    (T2D)", "(PheCode 250.2)", and "Type 2 Diabetes Mellitus" are merged.

    Trait score = mean_model_quality    * 0.30
               + mean_practical_match  * 0.15
               + mean_stability        * 0.10
               + mean_risk_concordance * 0.15
               + quality_count_weight  * 0.30

    quality_count weights each PGS model by tier:
        best (>=70): 4, normal (>=55): 1, moderate (>=40): 0.4, low (<40): 0.1
    then log2-normalizes the sum (capped at 48 = 12 good models).
    """
    pgs_ids = df["pgs_id"].unique().to_list()
    efo_map = _build_efo_trait_map(catalog, pgs_ids)

    # Map PGS IDs to grouped trait names
    pgs_traits = df.group_by("pgs_id").agg(
        pl.col("trait").first().alias("raw_trait"),
    ).with_columns(
        pl.col("pgs_id").replace_strict(efo_map, default=None).fill_null(pl.col("raw_trait")).alias("trait"),
    )

    n_merged = pgs_traits["trait"].n_unique()
    n_raw = pgs_traits["raw_trait"].n_unique()
    if n_merged < n_raw:
        print(f"  [efo] Grouped {n_raw} raw trait names → {n_merged} by ontology ID")

    trait_counts = pgs_traits.group_by("trait").len().rename({"len": "n_pgs"})
    eligible = trait_counts.filter(pl.col("n_pgs") >= min_pgs)

    if eligible.height == 0:
        return pl.DataFrame({"trait": [], "trait_demo_score": []})

    rated = (
        pgs_traits
        .join(r1.select("pgs_id", "model_quality"), on="pgs_id", how="left")
        .join(r2.select("pgs_id", "practical_match_score"), on="pgs_id", how="left")
        .join(r3.select("pgs_id", "stability_score"), on="pgs_id", how="left")
        .join(r4.select("pgs_id", "risk_concordance_score"), on="pgs_id", how="left")
    )

    trait_agg = rated.group_by("trait").agg([
        pl.col("model_quality").mean().alias("mean_model_q"),
        pl.col("practical_match_score").mean().alias("mean_match"),
        pl.col("stability_score").mean().alias("mean_stability"),
        pl.col("risk_concordance_score").mean().alias("mean_risk"),
        pl.col("pgs_id").n_unique().alias("n_pgs"),
        # Quality-weighted model count: best models count 4x, low nearly zero
        pl.when(pl.col("model_quality") >= 70).then(4.0)
          .when(pl.col("model_quality") >= 55).then(1.0)
          .when(pl.col("model_quality") >= 40).then(0.4)
          .otherwise(0.1)
          .sum().alias("quality_count"),
    ]).join(eligible.select("trait"), on="trait", how="inner")

    def _norm_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
        mn = df[col].min()
        mx = df[col].max()
        if mn is None or mx is None or mn == mx:
            return df.with_columns(pl.lit(50.0).alias(f"n_{col}"))
        return df.with_columns(
            ((pl.col(col).fill_null(0.0) - mn) / (mx - mn) * 100.0).alias(f"n_{col}")
        )

    trait_agg = _norm_col(trait_agg, "mean_model_q")
    trait_agg = _norm_col(trait_agg, "mean_match")
    trait_agg = _norm_col(trait_agg, "mean_stability")
    trait_agg = _norm_col(trait_agg, "mean_risk")

    # Quality-weighted count: log-based, capped at 48 (=12 good models)
    trait_agg = trait_agg.with_columns(
        (100.0 * (pl.col("quality_count").clip(upper_bound=48).cast(pl.Float64).log(2.0)
                   / math.log2(48))).alias("count_weight"),
    )

    # Weights: model 0.30, match 0.15, stability 0.10, risk 0.15, quality_count 0.30
    trait_agg = trait_agg.with_columns(
        pl.when(pl.col("n_mean_risk").is_not_null() & pl.col("n_mean_risk").is_not_nan())
        .then(
            pl.col("n_mean_model_q") * 0.30
            + pl.col("n_mean_match") * 0.15
            + pl.col("n_mean_stability") * 0.10
            + pl.col("n_mean_risk") * 0.15
            + pl.col("count_weight") * 0.30
        )
        .otherwise(
            pl.col("n_mean_model_q") * 0.353
            + pl.col("n_mean_match") * 0.176
            + pl.col("n_mean_stability") * 0.118
            + pl.col("count_weight") * 0.353
        )
        .alias("trait_demo_score"),
    )

    return trait_agg.sort("trait_demo_score", descending=True).select(
        "trait",
        "trait_demo_score",
        "n_pgs",
        "quality_count",
        "mean_model_q",
        "mean_match",
        "mean_stability",
        "mean_risk",
        "count_weight",
        pl.col("mean_risk").is_not_null().alias("has_risk_data"),
    )


def print_rating(title: str, df: pl.DataFrame, score_col: str, top: int = 20) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    show = df.head(top)
    # Determine display columns
    if "trait" in show.columns and "pgs_id" in show.columns:
        id_cols = ["pgs_id", "trait"]
    elif "trait" in show.columns:
        id_cols = ["trait"]
    else:
        id_cols = ["pgs_id"]

    extra_cols = [c for c in show.columns if c not in id_cols and c != score_col]
    display_cols = id_cols + [score_col] + extra_cols[:4]
    # Truncate trait strings for display
    truncated = show.select(display_cols)
    if "trait" in truncated.columns:
        truncated = truncated.with_columns(
            pl.col("trait").str.slice(0, 30).alias("trait")
        )

    with pl.Config(
        tbl_rows=top,
        tbl_cols=len(display_cols),
        tbl_width_chars=120,
        fmt_str_lengths=32,
    ):
        print(truncated)

    print(f"\n  Stats: mean={df[score_col].mean():.1f}, "
          f"median={df[score_col].median():.1f}, "
          f"min={df[score_col].min():.1f}, "
          f"max={df[score_col].max():.1f}")


def print_model_vs_practical_concordance(combined: pl.DataFrame) -> None:
    """Show whether high model quality predicts high practical quality."""
    print(f"\n{'='*70}")
    print("  MODEL vs PRACTICAL CONCORDANCE")
    print(f"{'='*70}")

    # Rank correlation (Spearman via rank-then-pearson)
    ranked = combined.with_columns(
        pl.col("model_quality").rank().alias("rank_model"),
        pl.col("combined_score").rank().alias("rank_combined"),
        pl.col("norm_match").rank().alias("rank_match"),
    )
    n = ranked.height
    # Spearman = pearson of ranks
    model_vs_combined = ranked.select(
        pl.corr("rank_model", "rank_combined").alias("r")
    )["r"][0]
    model_vs_match = ranked.select(
        pl.corr("rank_model", "rank_match").alias("r")
    )["r"][0]

    print(f"\n  Spearman rank correlations (n={n} PGS IDs):")
    print(f"    Model quality vs Combined practical:  r = {model_vs_combined:.3f}")
    print(f"    Model quality vs Practical match:     r = {model_vs_match:.3f}")

    # Quartile concordance table
    q_model = combined["model_quality"].quantile(0.75)
    q_combined = combined["combined_score"].quantile(0.75)
    tagged = combined.with_columns(
        pl.when(pl.col("model_quality") >= q_model).then(pl.lit("High"))
        .otherwise(pl.lit("Low")).alias("model_tier"),
        pl.when(pl.col("combined_score") >= q_combined).then(pl.lit("High"))
        .otherwise(pl.lit("Low")).alias("practical_tier"),
    )
    crosstab = tagged.group_by("model_tier", "practical_tier").len().sort("model_tier", "practical_tier")
    print(f"\n  Top-quartile concordance (model >= {q_model:.1f}, practical >= {q_combined:.1f}):")
    for row in crosstab.iter_rows(named=True):
        print(f"    Model {row['model_tier']:4s} + Practical {row['practical_tier']:4s}: {row['len']:4d} PGS IDs")

    agree = tagged.filter(
        (pl.col("model_tier") == pl.col("practical_tier"))
    ).height
    print(f"    Agreement rate: {agree/n*100:.1f}%")

    # Show biggest disagreements: high model but low practical
    pgs_trait = combined.join(
        combined.select("pgs_id").join(
            pl.read_parquet(RESULTS_PATH).filter(pl.col("error").is_null())
            .group_by("pgs_id").agg(pl.col("trait").first()),
            on="pgs_id", how="left",
        ),
        on="pgs_id", how="left",
    )
    overrated = tagged.join(pgs_trait.select("pgs_id", "trait"), on="pgs_id", how="left").filter(
        (pl.col("model_tier") == "High") & (pl.col("practical_tier") == "Low")
    ).sort("model_quality", descending=True)
    if overrated.height > 0:
        print(f"\n  Overrated by model (High model, Low practical) — top 5:")
        for row in overrated.head(5).iter_rows(named=True):
            t = (row.get("trait") or "")[:30]
            print(f"    {row['pgs_id']} {t:32s} model={row['model_quality']:.1f}  practical={row['combined_score']:.1f}")

    underrated = tagged.join(pgs_trait.select("pgs_id", "trait"), on="pgs_id", how="left").filter(
        (pl.col("model_tier") == "Low") & (pl.col("practical_tier") == "High")
    ).sort("combined_score", descending=True)
    if underrated.height > 0:
        print(f"\n  Underrated by model (Low model, High practical) — top 5:")
        for row in underrated.head(5).iter_rows(named=True):
            t = (row.get("trait") or "")[:30]
            print(f"    {row['pgs_id']} {t:32s} model={row['model_quality']:.1f}  practical={row['combined_score']:.1f}")


def print_stability_summary(df: pl.DataFrame) -> None:
    """Show how stable PRS results are across genomes."""
    print(f"\n{'='*70}")
    print("  CROSS-GENOME STABILITY SUMMARY")
    print(f"{'='*70}")

    stats = df.group_by("pgs_id").agg([
        pl.col("pct_EUR").std().alias("pct_std"),
        pl.col("match_rate").std().alias("match_std"),
        pl.col("score").std().alias("score_std"),
        pl.col("score").mean().alias("score_mean"),
    ]).with_columns(
        (pl.col("score_std") / pl.col("score_mean").abs().clip(lower_bound=0.001)).alias("score_cv"),
    )

    print(f"\n  EUR Percentile std across genomes:")
    print(f"    Median: {stats['pct_std'].median():.1f}")
    print(f"    Mean:   {stats['pct_std'].mean():.1f}")
    print(f"    < 10:   {stats.filter(pl.col('pct_std') < 10).height} PGS IDs ({stats.filter(pl.col('pct_std') < 10).height/stats.height*100:.0f}%)")
    print(f"    < 25:   {stats.filter(pl.col('pct_std') < 25).height} PGS IDs ({stats.filter(pl.col('pct_std') < 25).height/stats.height*100:.0f}%)")

    print(f"\n  Match rate std across genomes:")
    print(f"    Median: {stats['match_std'].median():.1f}")
    print(f"    Mean:   {stats['match_std'].mean():.1f}")
    print(f"    (Match rate varies by genome due to VCF variant coverage)")

    print(f"\n  Raw score CV across genomes:")
    print(f"    Median: {stats['score_cv'].median():.2f}")
    print(f"    Mean:   {stats['score_cv'].mean():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate PRS quality from smoke test results.")
    parser.add_argument("--top", type=int, default=20, help="Show top N entries per rating")
    parser.add_argument("--output", type=str, help="Write combined ratings to parquet")
    parser.add_argument(
        "--public-only", action="store_true",
        help="Use only public genomes (anton + livia) instead of all available",
    )
    args = parser.parse_args()

    df = load_results(public_only=args.public_only)
    catalog = PRSCatalog()

    pgs_ids = sorted(df["pgs_id"].unique().to_list())

    # --- Rating 1: Model quality ---
    print("\nComputing Rating 1: Model Quality (synthetic score)...")
    r1 = rating_1_model_quality(catalog, pgs_ids)
    # Add trait column
    pgs_trait = df.group_by("pgs_id").agg(pl.col("trait").first())
    r1 = r1.join(pgs_trait, on="pgs_id", how="left")
    print_rating("RATING 1: MODEL QUALITY (synthetic, metadata-only)", r1, "model_quality", args.top)

    # --- Rating 2: Practical match ---
    print("\nComputing Rating 2: Practical Match...")
    r2 = rating_2_practical_match(df)
    print_rating("RATING 2: PRACTICAL MATCH (actual genotype coverage)", r2, "practical_match_score", args.top)

    # --- Rating 3: Percentile stability ---
    print("\nComputing Rating 3: Percentile Stability...")
    r3 = rating_3_percentile_stability(df)
    print_rating("RATING 3: PERCENTILE STABILITY (cross-genome concordance)", r3, "stability_score", args.top)

    # --- Rating 4: Risk concordance ---
    print("\nComputing Rating 4: Risk Concordance...")
    r4 = rating_4_risk_concordance(df)
    print_rating("RATING 4: RISK CONCORDANCE (absolute risk discrimination)", r4, "risk_concordance_score", args.top)

    # --- Rating 5: Combined ---
    print("\nComputing Rating 5: Combined Practical...")
    r5 = combined_rating(r1, r2, r3, r4)
    r5 = r5.join(pgs_trait, on="pgs_id", how="left")
    print_rating("RATING 5: COMBINED PRACTICAL SCORE", r5, "combined_score", args.top)

    # --- Model vs Practical concordance ---
    print_model_vs_practical_concordance(r5)

    # --- Stability summary ---
    print_stability_summary(df)

    # --- Trait demo ranking ---
    print("\nComputing Trait Demo Ranking...")
    traits = trait_demo_ranking(r1, r2, r3, r4, df, catalog)
    print_rating("TRAIT DEMO RANKING (best showcase traits)", traits, "trait_demo_score", args.top)

    # Bottom traits for comparison
    if traits.height > args.top:
        print(f"\n  Bottom 10 traits:")
        with pl.Config(tbl_rows=10, tbl_cols=8, tbl_width_chars=120, fmt_str_lengths=32):
            bottom = traits.tail(10).with_columns(
                pl.col("trait").str.slice(0, 30)
            )
            print(bottom)

    # --- Save ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        r5_full = r5.join(
            r2.select("pgs_id", "practical_match_score", "match_mean", "match_std"),
            on="pgs_id", how="left",
        ).join(
            r3.select("pgs_id", "stability_score", "pct_eur_std"),
            on="pgs_id", how="left",
        ).join(
            r4.select("pgs_id", "risk_concordance_score", "mean_rel_deviation"),
            on="pgs_id", how="left",
        )
        r5_full.write_parquet(out_path)
        print(f"\nFull ratings written to {out_path}")

        suffix = "_public" if args.public_only else "_all"
        traits_path = out_path.parent / f"trait_demo_ranking{suffix}.parquet"
        traits.write_parquet(traits_path)
        print(f"Trait ranking written to {traits_path}")


if __name__ == "__main__":
    main()

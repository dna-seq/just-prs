"""PLINK2 comparison tests: validate our PRS against PLINK2 --score output.

PLINK2 is auto-downloaded if not found in PATH (see conftest.plink2_path fixture).
"""

import math
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest

from just_prs.prs import compute_prs
from just_prs.scoring import download_scoring_file, parse_scoring_file

PGS_IDS_GRCH38 = ["PGS000001", "PGS000002", "PGS000003", "PGS000004", "PGS000005"]


def _prepare_plink_scoring_file(scoring_path: Path, output_dir: Path) -> Path:
    """Convert PGS Catalog scoring file to PLINK2 --score format.

    PLINK2 --score expects: variant_id, effect_allele, effect_weight
    We use chr:pos as variant ID for position-based matching.
    """
    lf = parse_scoring_file(scoring_path)
    df = lf.collect()

    columns = df.columns
    if "hm_chr" in columns and "hm_pos" in columns:
        chr_col, pos_col = "hm_chr", "hm_pos"
    else:
        chr_col, pos_col = "chr_name", "chr_position"

    plink_df = df.select([
        (pl.col(chr_col).cast(pl.Utf8) + pl.lit(":") + pl.col(pos_col).cast(pl.Utf8)).alias("ID"),
        pl.col("effect_allele").alias("A1"),
        pl.col("effect_weight").alias("WEIGHT"),
    ]).drop_nulls()

    plink_score_path = output_dir / "plink_score.txt"
    plink_df.write_csv(plink_score_path, separator="\t")
    return plink_score_path


def _run_plink2_prs(
    plink2_bin: Path,
    vcf_path: Path,
    plink_score_path: Path,
    work_dir: Path,
) -> float:
    """Run PLINK2 to compute PRS and return the score.

    Steps:
    1. Convert VCF to PGEN format with chr:pos variant IDs
    2. Run --score to compute PRS
    3. Parse .sscore output
    """
    pgen_prefix = work_dir / "genotypes"
    subprocess.run(
        [
            str(plink2_bin),
            "--vcf", str(vcf_path),
            "--set-all-var-ids", "@:#",
            "--make-pgen",
            "--out", str(pgen_prefix),
            "--allow-extra-chr",
            "--autosome",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result_prefix = work_dir / "prs_result"
    subprocess.run(
        [
            str(plink2_bin),
            "--pfile", str(pgen_prefix),
            "--score", str(plink_score_path), "1", "2", "3",
            "header",
            "no-mean-imputation",
            "cols=+scoresums",
            "--out", str(result_prefix),
            "--allow-extra-chr",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    sscore_path = result_prefix.with_suffix(".sscore")
    sscore_df = pl.read_csv(sscore_path, separator="\t")

    score_col = None
    for col in sscore_df.columns:
        if "SCORESUM" in col.upper() or "SCORE1_SUM" in col.upper():
            score_col = col
            break

    if score_col is None:
        for col in sscore_df.columns:
            if "SCORE" in col.upper() and "AVG" not in col.upper():
                score_col = col
                break

    if score_col is None:
        raise ValueError(f"Could not find score column in {sscore_df.columns}")

    return float(sscore_df[score_col][0])


def _assert_scores_close(
    our_score: float,
    plink_score: float,
    pgs_id: str,
    tolerance: float = 0.05,
) -> None:
    """Assert that our PRS and PLINK2 PRS are within relative tolerance."""
    assert math.isfinite(our_score), f"{pgs_id}: our score is not finite ({our_score})"
    assert math.isfinite(plink_score), f"{pgs_id}: plink score is not finite ({plink_score})"

    if abs(plink_score) > 1e-10:
        relative_diff = abs(our_score - plink_score) / abs(plink_score)
        assert relative_diff < tolerance, (
            f"{pgs_id} PRS mismatch: ours={our_score:.6f}, plink2={plink_score:.6f}, "
            f"relative_diff={relative_diff:.4f}"
        )
    else:
        assert abs(our_score - plink_score) < 1e-6, (
            f"{pgs_id} PRS mismatch near zero: ours={our_score:.6f}, plink2={plink_score:.6f}"
        )


def test_prs_matches_plink2_pgs000001(
    vcf_path: Path, scoring_cache_dir: Path, plink2_path: Path
) -> None:
    """Compare our PRS computation with PLINK2 for PGS000001."""
    scoring_path = download_scoring_file(
        "PGS000001", scoring_cache_dir, genome_build="GRCh38"
    )

    our_result = compute_prs(
        vcf_path=vcf_path,
        scoring_file=scoring_path,
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        plink_score_path = _prepare_plink_scoring_file(scoring_path, work_dir)
        plink_score = _run_plink2_prs(plink2_path, vcf_path, plink_score_path, work_dir)

    _assert_scores_close(our_result.score, plink_score, "PGS000001")


@pytest.mark.parametrize("pgs_id", PGS_IDS_GRCH38)
def test_prs_matches_plink2_grch38(
    pgs_id: str,
    vcf_path: Path,
    scoring_cache_dir: Path,
    plink2_path: Path,
) -> None:
    """Compare our PRS computation with PLINK2 for a GRCh38 PGS score."""
    scoring_path = download_scoring_file(
        pgs_id, scoring_cache_dir, genome_build="GRCh38"
    )

    our_result = compute_prs(
        vcf_path=vcf_path,
        scoring_file=scoring_path,
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id=pgs_id,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        plink_score_path = _prepare_plink_scoring_file(scoring_path, work_dir)
        plink_score = _run_plink2_prs(plink2_path, vcf_path, plink_score_path, work_dir)

    _assert_scores_close(our_result.score, plink_score, pgs_id)


def test_prs_batch_summary_vs_plink2(
    vcf_path: Path,
    scoring_cache_dir: Path,
    plink2_path: Path,
) -> None:
    """Run all GRCh38 PGS IDs and produce a comparison summary table."""
    rows: list[dict[str, object]] = []

    for pgs_id in PGS_IDS_GRCH38:
        scoring_path = download_scoring_file(
            pgs_id, scoring_cache_dir, genome_build="GRCh38"
        )

        our_result = compute_prs(
            vcf_path=vcf_path,
            scoring_file=scoring_path,
            genome_build="GRCh38",
            cache_dir=scoring_cache_dir,
            pgs_id=pgs_id,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            plink_score_path = _prepare_plink_scoring_file(scoring_path, work_dir)
            plink_score = _run_plink2_prs(plink2_path, vcf_path, plink_score_path, work_dir)

        if abs(plink_score) > 1e-10:
            relative_diff = abs(our_result.score - plink_score) / abs(plink_score)
        else:
            relative_diff = abs(our_result.score - plink_score)

        rows.append({
            "pgs_id": pgs_id,
            "our_score": our_result.score,
            "plink_score": plink_score,
            "relative_diff": relative_diff,
            "variants_matched": our_result.variants_matched,
            "variants_total": our_result.variants_total,
            "match_rate": our_result.match_rate,
        })

    summary_df = pl.DataFrame(rows)
    print("\n=== PRS Comparison: just_prs vs PLINK2 ===")
    print(summary_df)

    for row in rows:
        _assert_scores_close(
            float(row["our_score"]),  # type: ignore[arg-type]
            float(row["plink_score"]),  # type: ignore[arg-type]
            str(row["pgs_id"]),
        )

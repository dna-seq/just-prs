"""PLINK2 comparison tests: validate our PRS against PLINK2 --score output.

These tests are skipped if plink2 is not installed.
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest

from just_prs.prs import compute_prs
from just_prs.scoring import download_scoring_file, parse_scoring_file

plink2_available = shutil.which("plink2") is not None
skip_no_plink2 = pytest.mark.skipif(
    not plink2_available, reason="plink2 not found in PATH"
)


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
            "plink2",
            "--vcf", str(vcf_path),
            "--set-all-var-ids", "@:#",
            "--make-pgen",
            "--out", str(pgen_prefix),
            "--allow-extra-chr",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result_prefix = work_dir / "prs_result"
    subprocess.run(
        [
            "plink2",
            "--pfile", str(pgen_prefix),
            "--score", str(plink_score_path), "1", "2", "3",
            "header",
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


@skip_no_plink2
def test_prs_matches_plink2_pgs000001(
    vcf_path: Path, scoring_cache_dir: Path
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
        plink_score = _run_plink2_prs(vcf_path, plink_score_path, work_dir)

    assert math.isfinite(our_result.score)
    assert math.isfinite(plink_score)

    if abs(plink_score) > 1e-10:
        relative_diff = abs(our_result.score - plink_score) / abs(plink_score)
        assert relative_diff < 0.05, (
            f"PRS mismatch: ours={our_result.score:.6f}, plink2={plink_score:.6f}, "
            f"relative_diff={relative_diff:.4f}"
        )
    else:
        assert abs(our_result.score - plink_score) < 1e-6

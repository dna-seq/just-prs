"""Deterministic real-genotype test for normalize_array using the 1000G panel.

Synthesizes a 23andMe-format file from one real 1000G sample's chr22 SNP
genotypes, then verifies normalize_array -> compute_prs reproduces the exact
ALT-dosage sum we compute independently from the raw genotype codes. This
exercises the allele-encoding on real, diverse genotypes (hom-ref / het /
hom-alt / missing) rather than hand-built toy data.

Skips gracefully when the ~7 GB reference panel or pgenlib is unavailable
(clean clone), matching the project's cache-optional test policy.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from just_prs.arrays import normalize_array
from just_prs.prs import compute_prs
from just_prs.scoring import resolve_cache_dir

pytest.importorskip("pgenlib")

_PANEL = resolve_cache_dir() / "reference_panel" / "pgsc_1000G_v1"
_PGEN = _PANEL / "GRCh38_1000G_ALL.pgen"
_PVAR = _PANEL / "GRCh38_1000G_ALL.pvar.zst"
_PSAM = _PANEL / "GRCh38_1000G_ALL.psam"

_N_VARIANTS = 300  # chr22 biallelic SNPs to synthesize onto the "array"


def _require_panel() -> None:
    for p in (_PGEN, _PVAR, _PSAM):
        if not p.exists():
            pytest.skip(f"Reference panel file missing ({p}); run the pipeline first.")


@pytest.fixture(scope="module")
def synthetic_array(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build a synthetic 23andMe file + matching scoring file from real genotypes.

    Returns a dict with the array path, scoring LazyFrame, and the
    independently-computed expected score / matched-variant count for sample 0.
    """
    _require_panel()
    from just_prs.reference import parse_psam, parse_pvar, read_pgen_genotypes

    pvar = parse_pvar(_PVAR)
    # Biallelic SNPs on chr22 only (single-char REF/ALT in ACGT) — array-realistic.
    # Dedupe to ONE variant per position: compute_prs joins on (chrom, pos) only,
    # so multiallelic sites sharing a position would create a many-to-one join.
    snps = (
        pvar.filter(
            (pl.col("chrom").cast(pl.Utf8) == "22")
            & pl.col("REF").is_in(["A", "C", "G", "T"])
            & pl.col("ALT").is_in(["A", "C", "G", "T"])
        )
        .sort("POS")
        .unique(subset=["POS"], keep="first", maintain_order=True)
        .head(_N_VARIANTS)
    )
    if snps.height < _N_VARIANTS:
        pytest.skip("Not enough chr22 biallelic SNPs in panel.")

    n_samples = parse_psam(_PSAM).height
    idx = snps["variant_idx"].cast(pl.UInt32).to_numpy()
    geno = read_pgen_genotypes(
        pgen_path=_PGEN, pvar_zst_path=_PVAR, variant_indices=idx, n_samples=n_samples,
    )  # shape (n_variants, n_samples), values 0/1/2 (ALT dosage), <0 or >2 = missing

    sample_codes = geno[:, 0]  # first sample
    refs = snps["REF"].to_list()
    alts = snps["ALT"].to_list()
    poss = snps["POS"].to_list()

    lines = ["# synthetic 23andMe export from 1000G sample 0", "# rsid\tchromosome\tposition\tgenotype"]
    expected_score = 0.0
    expected_matched = 0
    for i, code in enumerate(sample_codes):
        ref, alt, pos = refs[i], alts[i], poss[i]
        c = int(code)
        if c == 0:
            gt = ref + ref
        elif c == 1:
            gt = ref + alt
        elif c == 2:
            gt = alt + alt
        else:
            gt = "--"  # missing → dropped by normalize_array, contributes nothing
        lines.append(f"rs{i}\t22\t{pos}\t{gt}")
        if c in (0, 1, 2):
            expected_score += float(c)  # effect_allele = ALT, weight 1.0 → dosage == code
            expected_matched += 1

    array_dir = tmp_path_factory.mktemp("synthetic_array")
    array_path = array_dir / "sample0.txt"
    array_path.write_text("\n".join(lines) + "\n")

    scoring = pl.DataFrame({
        "rsID": [f"rs{i}" for i in range(_N_VARIANTS)],
        "hm_chr": ["22"] * _N_VARIANTS,
        "hm_pos": poss,
        "effect_allele": alts,
        "other_allele": refs,
        "effect_weight": [1.0] * _N_VARIANTS,
    }).lazy()

    return {
        "array_path": array_path,
        "scoring": scoring,
        "expected_score": expected_score,
        "expected_matched": expected_matched,
        "out_dir": array_dir,
    }


def test_normalize_array_reproduces_alt_dosage_sum(synthetic_array: dict) -> None:
    """compute_prs over the normalized array equals the independent ALT-dosage sum."""
    geno_parquet = normalize_array(
        synthetic_array["array_path"],
        synthetic_array["out_dir"] / "norm.parquet",
        genome_build="GRCh38",
    )
    result = compute_prs(
        vcf_path="",
        scoring_file=synthetic_array["scoring"],
        genome_build="GRCh38",
        cache_dir=synthetic_array["out_dir"],
        pgs_id="PGSSYNTH",
        genotypes_lf=pl.scan_parquet(geno_parquet),
    )
    # Effect allele = ALT, weight 1.0 → score is exactly the summed ALT dosage,
    # and every non-missing synthesized SNP must match the scoring file.
    assert result.variants_matched == synthetic_array["expected_matched"]
    assert result.score == pytest.approx(synthetic_array["expected_score"])
    # Real genotypes must contain a mix (not all hom-ref) for this to be meaningful.
    assert synthetic_array["expected_score"] > 0

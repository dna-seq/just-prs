"""Privé / bigsnpr worldwide-reference ancestry proportions (Level-2, finer resolution).

A third, independent ancestry view to complement the in-house 1000G/HGDP models: the
UK-Biobank-derived worldwide reference from Privé 2022 ("Using the UK Biobank as a global
reference of worldwide populations", Bioinformatics). It estimates **admixture
proportions** over ~21 fine ancestry groups by projecting a sample onto 16 published PCs
(with a per-PC shrinkage `correction`) and solving a simplex-constrained least squares
against the groups' allele-frequency PC-centroids — then rolling the 21 groups up to the
canonical continental super-pops for the consensus.

Method ported from the bigsnpr `ancestry` vignette (GPL-3 R code — we reimplement the
math in numpy; only the **data** files are consumed). The reference is **GRCh37**, so a
GRCh38 sample is lifted 38→37 first. Reference data files (figshare, ~1.7 GB) and their
re-hosting license must be confirmed before publishing to HF; built/consumed locally here.

Reference files (figshare):
- allele frequencies: files/31620968  (chr,pos,a0,a1,rsid + 21 group freqs; 5,816,590 rows)
- PC loadings:        files/31620953  (chr,pos,a0,a1,rsid + PC1..PC16)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from eliot import start_action

from just_prs.ancestry._projection import DEFAULT_MISSING  # noqa: F401  (kept for parity)
from just_prs.ancestry import _simplex_lstsq
from just_prs.vcf import compute_dosage_expr

PRIVE_FREQ_URL = "https://figshare.com/ndownloader/files/31620968"
PRIVE_PROJ_URL = "https://figshare.com/ndownloader/files/31620953"
PRIVE_BUILD = "GRCh37"
N_PCS = 16

# Per-PC shrinkage correction for projecting NEW samples onto Privé's loadings (from the
# bigsnpr vignette; specific to these published loadings — do not recompute).
CORRECTION = (
    1.0, 1.0, 1.0, 1.008, 1.021, 1.034, 1.052, 1.074,
    1.099, 1.123, 1.15, 1.195, 1.256, 1.321, 1.382, 1.443,
)

CANONICAL_SUPERPOPS = ("AFR", "AMR", "EAS", "EUR", "SAS")


def prive_group_to_continental(name: str) -> str | None:
    """Roll a Privé fine ancestry-group name up to a canonical continental super-pop.

    Keyword-based so it is robust to the exact group spellings. Middle Eastern groups
    have no clean 1000G super-pop and return None (dropped + renormalized downstream,
    mirroring the HGDP ``MID`` handling).
    """
    n = name.lower()
    if "africa" in n:
        return "AFR"
    if "america" in n or "latin" in n or "hispanic" in n:
        return "AMR"
    if any(k in n for k in ("east asia", "asia (east", "japan", "china", "korea", "philippin")):
        return "EAS"
    if any(k in n for k in ("south asia", "asia (south", "pakistan", "bangladesh", "sri lanka", "india")):
        return "SAS"
    if any(k in n for k in ("middle east", "near east")):
        return None  # no 1000G equivalent
    if any(k in n for k in (
        "europe", "scandinav", "united kingdom", "ireland", "italy", "finland",
        "ashkenazi", "spain", "portugal", "france", "germany",
    )):
        return "EUR"
    return None


# ---------------------------------------------------------------------------
# Build (offline): repackage the two figshare CSVs into one parquet + meta.
# ---------------------------------------------------------------------------


def _ref_paths(ref_dir: Path) -> dict[str, Path]:
    return {
        "parquet": ref_dir / "prive_reference.parquet",
        "meta": ref_dir / "prive_reference_meta.json",
    }


def build_prive_reference(freq_csv: Path, proj_csv: Path, out_dir: Path) -> dict[str, object]:
    """Merge the freq + loadings CSVs into a single reference parquet + meta in ``out_dir``.

    Output parquet columns: ``chr, pos, a0, a1`` + ``load0..load{N_PCS-1}`` (loadings) +
    one column per fine ancestry group (allele frequency of ``a1``). Meta records the
    group names, the ``correction`` vector, and the group→continental rollup.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _ref_paths(out_dir)
    with start_action(action_type="prive:build_reference", out_dir=str(out_dir)):
        freq = pl.read_csv(freq_csv)
        proj = pl.read_csv(proj_csv)
        key = ["chr", "pos", "a0", "a1"]
        groups = [c for c in freq.columns if c not in ("chr", "pos", "a0", "a1", "rsid")]
        pcs = [c for c in proj.columns if c not in ("chr", "pos", "a0", "a1", "rsid")]
        if len(pcs) != N_PCS:
            raise ValueError(f"expected {N_PCS} PC loadings, found {len(pcs)}: {pcs}")

        proj_ren = proj.select(key + pcs).rename({c: f"load{i}" for i, c in enumerate(pcs)})
        merged = (
            freq.select(key + groups)
            .join(proj_ren, on=key, how="inner")
            .with_columns(
                pl.col("chr").cast(pl.Utf8).str.replace("(?i)^chr", ""),
                pl.col("pos").cast(pl.Int64),
            )
        )
        merged.write_parquet(p["parquet"])

        rollup = {g: prive_group_to_continental(g) for g in groups}
        meta = {
            "build": PRIVE_BUILD,
            "n_pcs": N_PCS,
            "correction": list(CORRECTION),
            "groups": groups,
            "group_to_continental": rollup,
            "n_variants": merged.height,
        }
        p["meta"].write_text(json.dumps(meta, indent=2))
        return {"parquet": str(p["parquet"]), "n_variants": merged.height, "n_groups": len(groups)}


# ---------------------------------------------------------------------------
# Runtime: project a sample, solve the QP, return proportions.
# ---------------------------------------------------------------------------


def estimate_prive_proportions(
    ref_dir: Path, genotypes_lf: pl.LazyFrame, sample_build: str = "GRCh38"
) -> dict[str, object]:
    """Estimate a sample's ancestry proportions against the Privé reference.

    Lifts the sample to GRCh37 if needed, matches by (chr,pos) allele-aware (orienting to
    the reference ``a1``), projects onto the 16 PCs with the shrinkage ``correction``,
    builds the group PC-centroids on the matched subset, solves the simplex-constrained
    least squares (Privé's QP), and returns both the 21-group ``proportions`` and the
    continental rollup ``continental`` (canonical 5 super-pops; Middle-East mass dropped).

    Uses only variants present in the sample ∩ reference (Privé's documented method,
    which assumes dense genotypes); for a variant-only VCF this is the recorded-variant
    intersection — finer-resolution but less hom-ref-robust than the 1000G path, hence
    offered as an opt-in complement, not the default.
    """
    p = _ref_paths(ref_dir)
    if not p["parquet"].exists():
        raise FileNotFoundError(f"Privé reference not built: {p['parquet']}")
    meta = json.loads(p["meta"].read_text())
    groups: list[str] = meta["groups"]
    correction = np.asarray(meta["correction"], dtype=np.float64)
    n_pcs = int(meta["n_pcs"])

    with start_action(action_type="prive:estimate", sample_build=sample_build):
        if sample_build != PRIVE_BUILD:
            from just_prs.liftover import lift_frame

            lifted, _ = lift_frame(
                genotypes_lf.collect(), sample_build, PRIVE_BUILD,
                chrom_col="chrom", pos_col="pos",
            )
            genotypes_lf = lifted.lazy()

        ref = pl.read_parquet(p["parquet"])
        sample = (
            genotypes_lf.select(
                pl.col("chrom").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chr"),
                pl.col("pos").cast(pl.Int64),
                pl.col("ref").cast(pl.Utf8).alias("ref_s"),
                pl.col("alt").cast(pl.Utf8).alias("alt_s"),
                pl.col("GT").cast(pl.Utf8),
            )
            .unique(subset=["chr", "pos"], keep="first")
            .collect()
        )
        # Match by (chr,pos) + biallelic allele set; orient to reference a1.
        joined = ref.join(sample, on=["chr", "pos"], how="inner").with_columns(
            effect_allele=pl.col("a1")
        )
        allele_ok = (
            ((pl.col("a0") == pl.col("ref_s")) & (pl.col("a1") == pl.col("alt_s")))
            | ((pl.col("a0") == pl.col("alt_s")) & (pl.col("a1") == pl.col("ref_s")))
        )
        dose_a1 = compute_dosage_expr(
            gt_col="GT", ref_col="ref_s", alt_col="alt_s", effect_allele_col="effect_allele"
        )
        joined = joined.filter(allele_ok).with_columns(dose_a1.cast(pl.Float64).alias("_dose"))
        n_used = joined.height
        if n_used == 0:
            return {"proportions": {}, "continental": {}, "n_variants_used": 0}

        loadings = joined.select([f"load{i}" for i in range(n_pcs)]).to_numpy()  # (m x 16)
        dose = joined["_dose"].to_numpy()                                        # (m,)
        freqs = joined.select(groups).to_numpy()                                 # (m x 21)

        # Individual projection: y_pc = Σ_v dose(v) · loading(v,pc) · correction_pc/2
        y = (dose @ loadings) * (correction / 2.0)                               # (16,)
        # Group centroids: X[:, j] = Σ_v loading(v,pc) · freq(v,j)  (no correction)
        X = loadings.T @ freqs                                                    # (16 x 21)

        w = _simplex_lstsq(X, y)                                                  # (21,)
        proportions = {g: round(float(wi), 4) for g, wi in zip(groups, w)}

        # Roll up to canonical continental super-pops (drop unmapped/Middle-East, renorm).
        cont = {p_: 0.0 for p_ in CANONICAL_SUPERPOPS}
        for g, wi in zip(groups, w):
            c = prive_group_to_continental(g)
            if c in cont:
                cont[c] += float(wi)
        total = sum(cont.values())
        continental = {k: round(v / total, 4) if total > 0 else 0.0 for k, v in cont.items()}

        return {
            "proportions": proportions,
            "continental": continental,
            "n_variants_used": n_used,
        }

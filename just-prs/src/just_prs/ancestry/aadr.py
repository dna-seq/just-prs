"""Build the `aadr_ho` ancestry model from AADR Human Origins present-day individuals.

A West-Eurasian-focused fine-population model for Slavic / Eastern-European resolution
(Russian / Ukrainian / Belarusian / Czech / Bulgarian / Serbian / Croatian / Romanian /
Balt / Finnic …), which the 1000G/HGDP super-pop models can't reach. Built **offline** from
local EIGENSTRAT files (GPL-adjacent academic data — not HF-published); GRCh37 reference, so
GRCh38 samples lift 38→37 at inference (handled by the existing lift path).

Pipeline: read present-day West-Eurasian individuals from the packed TGENO `.geno`
(`eigenstrat.py`) → flip reference-allele count to ALT dosage (`2-g`) so it matches the
`build_ancestry_model` ALT convention → QC (autosomes, biallelic ACGT, drop strand-ambiguous,
MAF) → reuse `build_ancestry_model` (numpy SVD + LOO + persist) with `dim_ref=20` (fine
within-Europe structure needs deeper PCs). No LD-prune: the HO set is already ascertained.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from eliot import start_action

from just_prs.ancestry import build_ancestry_model
from just_prs.ancestry.eigenstrat import (
    parse_anno_present_day,
    parse_ind,
    parse_snp,
    read_tgeno,
)

# West-Eurasian bounding box (Europe + N-Africa edge + Near East + Caucasus + W/Central Asia).
_WEST_EURASIA = {"lat": (30.0, 75.0), "lon": (-25.0, 90.0)}
_AMBIGUOUS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
# QC-flagged AADR group suffixes/markers to exclude.
_QC_MARKERS = ("Ignore", "QCremove", "outlier", "_dup", "_rel", "_contam", "_lc", "_fail", "..")
# Fine groups that roll up to SAS (Central/South-Asian present-day in the bbox); everything
# else West-Eurasian rolls up to EUR (the broad 1000G-vocab call for coherence).
_SAS_PATTERNS = ("pakistan", "baloch", "brahui", "makrani", "sindhi", "pathan", "kalash",
                 "bengali", "punjabi", "gujarati", "_gih", "pjl", "beb", "stu", "itu", "indian")


def _superpop(group: str) -> str:
    g = group.lower()
    return "SAS" if any(p in g for p in _SAS_PATTERNS) else "EUR"


def _is_qc_flagged(group: str) -> bool:
    return any(m.lower() in group.lower() for m in _QC_MARKERS) or group.endswith(("-o", "_o", "-oPCA"))


def build_aadr_ho_model(
    eigenstrat_prefix,  # e.g. /data/genomes/dataverse_66p1/v66.p1_compatibility_HO.aadr.patch.PUB
    anno_path,
    model_dir,
    *,
    dim_ref: int = 20,
    maf: float = 0.02,
) -> dict:
    """Build + persist the aadr_ho GRCh37 model from EIGENSTRAT files. Returns build metadata."""
    from pathlib import Path

    pre = Path(eigenstrat_prefix)
    geno_p, snp_p, ind_p = Path(f"{pre}.geno"), Path(f"{pre}.snp"), Path(f"{pre}.ind")
    with start_action(action_type="aadr:build", prefix=str(pre)):
        ind = parse_ind(ind_p)
        anno = parse_anno_present_day(Path(anno_path))
        snp = parse_snp(snp_p)
        n_ind_total, n_snp = ind.height, snp.height

        # Select present-day, non-QC, West-Eurasian individuals.
        keep_idx, groups, superpops = [], [], []
        for r in ind.iter_rows(named=True):
            a = anno.get(r["iid"])
            if a is None or a["date_bp"] != 0 or _is_qc_flagged(r["group"]):
                continue
            lat, lon = a["lat"], a["lon"]
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            if not (_WEST_EURASIA["lat"][0] <= lat <= _WEST_EURASIA["lat"][1]
                    and _WEST_EURASIA["lon"][0] <= lon <= _WEST_EURASIA["lon"][1]):
                continue
            keep_idx.append(r["idx"]); groups.append(r["group"]); superpops.append(_superpop(r["group"]))
        keep_idx = np.asarray(keep_idx, dtype=np.int64)
        if keep_idx.size < 4 * dim_ref * 2:
            raise ValueError(f"only {keep_idx.size} individuals selected (need ≥ {8 * dim_ref})")

        # Read genotypes (ref-allele count) → ALT dosage (2-g), missing preserved.
        g = read_tgeno(geno_p, n_ind_total, n_snp, keep_idx)  # (n_sel x n_snp), 0/1/2/-9
        g_alt = np.where(g == -9, -9, 2 - g).astype(np.int8).T  # (n_snp x n_sel) ALT dosage

        # SNP QC: autosomes, biallelic ACGT, drop strand-ambiguous, MAF.
        chrom = snp["chrom"].to_numpy()
        ref = snp["ref"].to_numpy(); alt = snp["alt"].to_numpy()
        acgt = {"A", "C", "G", "T"}
        autosome = np.array([str(c).isdigit() and 1 <= int(c) <= 22 for c in chrom])
        biallelic = np.array([r in acgt and a in acgt for r, a in zip(ref, alt)])
        ambiguous = np.array([(r, a) in _AMBIGUOUS for r, a in zip(ref, alt)])
        valid = autosome & biallelic & ~ambiguous
        # MAF over selected samples (ALT dosage; missing excluded).
        masked = np.where(g_alt == -9, np.nan, g_alt)
        af = np.nanmean(masked, axis=1) / 2.0
        valid &= np.isfinite(af) & (np.minimum(af, 1 - af) >= maf)

        g_alt = g_alt[valid]
        sites = snp.filter(pl.Series(valid)).select(
            pl.col("chrom").cast(pl.Utf8), pl.col("pos").cast(pl.Int64),
            pl.col("ref").cast(pl.Utf8), pl.col("alt").cast(pl.Utf8),
        )
        labels = pl.DataFrame({
            "iid": [ind[int(i), "iid"] for i in keep_idx],
            "superpop": superpops,
            "population": groups,
        })

        info = build_ancestry_model(
            g_alt, sites, labels, panel="aadr_ho", build="GRCh37",
            model_dir=Path(model_dir), dim_ref=dim_ref,
        )
        info.update({"n_individuals": int(keep_idx.size), "n_sites": int(g_alt.shape[0]),
                     "n_populations": labels["population"].n_unique()})
        return info

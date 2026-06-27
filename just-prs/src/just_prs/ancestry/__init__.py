"""Sample genetic-ancestry inference (Level 1): reference-PCA build + projection + KNN.

Runtime is pure-Python (numpy + polars); no plink2, no compiled binary, no GPL — the
reference-PCA *model* is built offline (pipeline) and published to HF, then pulled and
consumed here. Projection uses the vendored FRAPOSA OADP math (``_projection``).

Artifact layout (under a model directory), per (panel, build):
- ``ancestry_model_<panel>_<build>.parquet`` — pruned sites:
  ``chrom, pos, ref, alt, effect_allele, mean, std, u0..u{dim_online-1}`` (U loadings).
- ``ancestry_refpcs_<panel>_<build>.parquet`` — reference samples:
  ``iid, superpop, population, v0..v{dim_online-1}`` (V eigenvectors).
- ``ancestry_model_<panel>_<build>_meta.json`` — ``s`` (singular values), ``dim_ref``,
  ``dim_stu``, ``dim_online``, ``knn_k``, ``loo_accuracy``, ``n_reference``,
  ``n_variants``, ``superpopulations``, ``fine_to_broad``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from eliot import start_action

from just_prs.ancestry._projection import (
    DEFAULT_MISSING,
    ReferencePCA,
    build_reference_pca,
    project_samples,
)
from just_prs.models import AncestryInference
from just_prs.vcf import compute_dosage_expr

__all__ = [
    "build_ancestry_model",
    "infer_ancestry",
    "load_ancestry_model",
    "artifact_paths",
    "DEFAULT_KNN_K",
    "DEFAULT_DIM_REF",
    "DEFAULT_COVERAGE_FLOOR",
]

DEFAULT_KNN_K = 20
DEFAULT_DIM_REF = 10
# Below this fraction of model sites matched, the call is treated as UNKNOWN.
DEFAULT_COVERAGE_FLOOR = 0.2
_UNKNOWN = "UNKNOWN"


def artifact_paths(model_dir: Path, panel: str, build: str) -> dict[str, Path]:
    """Return the three artifact paths for a (panel, build) model under ``model_dir``."""
    stem = f"ancestry_model_{panel}_{build}"
    return {
        "sites": model_dir / f"{stem}.parquet",
        "refpcs": model_dir / f"ancestry_refpcs_{panel}_{build}.parquet",
        "meta": model_dir / f"{stem}_meta.json",
    }


# ---------------------------------------------------------------------------
# Classification (pure-numpy KNN over reference PC scores)
# ---------------------------------------------------------------------------


def _knn_predict(
    ref_pcs: np.ndarray, ref_labels: np.ndarray, query: np.ndarray, k: int
) -> tuple[str, dict[str, float]]:
    """Majority-vote KNN. Returns (label, {label: fraction-of-k})."""
    d = np.linalg.norm(ref_pcs - query, axis=1)
    nn = np.argsort(d)[:k]
    labs = ref_labels[nn]
    uniq, counts = np.unique(labs, return_counts=True)
    probs = {str(u): float(c) / float(k) for u, c in zip(uniq, counts)}
    label = str(uniq[int(np.argmax(counts))])
    return label, probs


def _loo_accuracy(ref_pcs: np.ndarray, ref_labels: np.ndarray, k: int) -> float:
    """Leave-one-out KNN accuracy on the reference PC scores (model-quality proxy)."""
    n = ref_pcs.shape[0]
    if n <= k:
        return 0.0
    correct = 0
    for i in range(n):
        d = np.linalg.norm(ref_pcs - ref_pcs[i], axis=1)
        d[i] = np.inf  # exclude self
        nn = np.argsort(d)[:k]
        labs = ref_labels[nn]
        uniq, counts = np.unique(labs, return_counts=True)
        correct += uniq[int(np.argmax(counts))] == ref_labels[i]
    return float(correct) / float(n)


# ---------------------------------------------------------------------------
# Build (offline / pipeline)
# ---------------------------------------------------------------------------


def build_ancestry_model(
    genotypes: np.ndarray,
    sites: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    panel: str,
    build: str,
    model_dir: Path,
    dim_ref: int = DEFAULT_DIM_REF,
    knn_k: int = DEFAULT_KNN_K,
    fine_to_broad: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a reference-PCA ancestry model and persist its artifact.

    Args:
        genotypes: (n_variants x n_samples) int8 ALT-allele dosage (0/1/2, -9 missing),
            row order matching ``sites``, column order matching ``labels``.
        sites: pruned variants with columns ``chrom, pos, ref, alt`` (effect = ALT).
        labels: reference samples with columns ``iid, superpop, population``.
        fine_to_broad: optional map from panel labels to the broad 5 super-pops.

    Returns metadata dict (paths, ``loo_accuracy``, ``n_variants``, ``n_reference``).
    """
    if genotypes.shape[0] != sites.height:
        raise ValueError(f"genotypes rows {genotypes.shape[0]} != sites {sites.height}")
    if genotypes.shape[1] != labels.height:
        raise ValueError(f"genotypes cols {genotypes.shape[1]} != labels {labels.height}")

    model = build_reference_pca(genotypes, dim_ref=dim_ref)
    ref_pcs = model.pcs_ref  # (n_samples x dim_ref)
    superpops = labels["superpop"].to_numpy()
    loo = _loo_accuracy(ref_pcs, superpops, knn_k)

    model_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(model_dir, panel, build)

    # sites parquet: site cols + per-variant mean/std + U loadings (u0..)
    u_cols = {f"u{j}": model.U[:, j] for j in range(model.dim_online)}
    sites_out = sites.select(
        pl.col("chrom").cast(pl.Utf8),
        pl.col("pos").cast(pl.Int64),
        pl.col("ref").cast(pl.Utf8),
        pl.col("alt").cast(pl.Utf8),
    ).with_columns(
        effect_allele=pl.col("alt"),
        mean=pl.Series(model.mean.flatten()),
        std=pl.Series(model.std.flatten()),
        **{c: pl.Series(v) for c, v in u_cols.items()},
    )
    sites_out.write_parquet(paths["sites"])

    # refpcs parquet: labels + V eigenvectors (v0..)
    v_cols = {f"v{j}": model.V[:, j] for j in range(model.dim_online)}
    refpcs_out = labels.select(
        pl.col("iid").cast(pl.Utf8),
        pl.col("superpop").cast(pl.Utf8),
        pl.col("population").cast(pl.Utf8),
    ).with_columns(**{c: pl.Series(v) for c, v in v_cols.items()})
    refpcs_out.write_parquet(paths["refpcs"])

    meta = {
        "panel": panel,
        "build": build,
        "dim_ref": model.dim_ref,
        "dim_stu": model.dim_stu,
        "dim_online": model.dim_online,
        "s": model.s.tolist(),
        "knn_k": knn_k,
        "loo_accuracy": loo,
        "n_reference": labels.height,
        "n_variants": sites.height,
        "superpopulations": sorted({str(x) for x in superpops.tolist()}),
        "fine_to_broad": fine_to_broad or {},
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))

    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "loo_accuracy": loo,
        "n_variants": sites.height,
        "n_reference": labels.height,
    }


# ---------------------------------------------------------------------------
# Load + infer (runtime)
# ---------------------------------------------------------------------------


class _LoadedModel:
    __slots__ = ("pca", "sites", "ref_labels", "ref_pcs", "meta")

    def __init__(self, pca, sites, ref_labels, ref_pcs, meta):
        self.pca = pca
        self.sites = sites
        self.ref_labels = ref_labels
        self.ref_pcs = ref_pcs
        self.meta = meta


def load_ancestry_model(model_dir: Path, panel: str, build: str) -> _LoadedModel:
    """Load a persisted ancestry model into a runtime-ready in-memory form."""
    paths = artifact_paths(model_dir, panel, build)
    for key in ("sites", "refpcs", "meta"):
        if not paths[key].exists():
            raise FileNotFoundError(f"ancestry model artifact missing: {paths[key]}")
    meta = json.loads(paths["meta"].read_text())
    dim_online = int(meta["dim_online"])

    sites = pl.read_parquet(paths["sites"])
    u = sites.select([f"u{j}" for j in range(dim_online)]).to_numpy()
    mean = sites["mean"].to_numpy().reshape((-1, 1))
    std = sites["std"].to_numpy().reshape((-1, 1))

    refpcs_df = pl.read_parquet(paths["refpcs"])
    V = refpcs_df.select([f"v{j}" for j in range(dim_online)]).to_numpy()
    s = np.asarray(meta["s"], dtype=np.float64)
    pca = ReferencePCA(
        mean=mean, std=std, s=s, V=V, U=u,
        dim_ref=int(meta["dim_ref"]), dim_stu=int(meta["dim_stu"]), dim_online=dim_online,
    )
    ref_labels = refpcs_df["superpop"].to_numpy()
    return _LoadedModel(pca, sites, ref_labels, pca.pcs_ref, meta)


def _sample_dosage_vector(
    sites: pl.DataFrame, genotypes_lf: pl.LazyFrame, assume_homref_absent: bool = True
) -> np.ndarray:
    """Effect-allele (ALT) dosage of the sample at each model site, in sites order.

    Allele-aware: counts the model ALT allele as observed in the sample (both
    orientations). Site handling:

    - present & alleles match -> observed ALT dosage (0/1/2);
    - present but allele-set mismatch -> missing (``DEFAULT_MISSING``);
    - **absent from the sample** -> hom-ref (ALT dosage 0) when ``assume_homref_absent``
      (the correct default for a **variant-only VCF**, where a common site the sample is
      hom-ref at is simply not emitted — the F15 hom-ref theme), else missing.

    Treating absent common sites as missing→mean would shrink the projection toward the
    population centroid and collapse every sparse WGS sample onto the central (Admixed
    American) cluster — so hom-ref imputation of absent sites is essential here. (For a
    genotyping *array*, absent means untyped, not hom-ref; pass
    ``assume_homref_absent=False`` for array input.)
    """
    sample = (
        genotypes_lf.select(
            pl.col("chrom").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
            pl.col("pos").cast(pl.Int64),
            pl.col("ref").cast(pl.Utf8).alias("ref_s"),
            pl.col("alt").cast(pl.Utf8).alias("alt_s"),
            pl.col("GT").cast(pl.Utf8),
        )
        .unique(subset=["chrom", "pos"], keep="first")
        .collect()
    )
    idx = sites.with_row_index("_idx").select(
        "_idx", "chrom", "pos", "ref", "alt", pl.col("alt").alias("effect_allele")
    )
    joined = idx.join(sample, on=["chrom", "pos"], how="left").sort("_idx")
    dose = compute_dosage_expr(
        gt_col="GT", ref_col="ref_s", alt_col="alt_s", effect_allele_col="effect_allele"
    )
    present = pl.col("GT").is_not_null()
    allele_match = present & (
        ((pl.col("ref") == pl.col("ref_s")) & (pl.col("alt") == pl.col("alt_s")))
        | ((pl.col("ref") == pl.col("alt_s")) & (pl.col("alt") == pl.col("ref_s")))
    )
    absent_value = 0.0 if assume_homref_absent else DEFAULT_MISSING
    joined = joined.with_columns(
        pl.when(allele_match).then(dose)
        .when(~present).then(absent_value)  # absent in variant-only VCF -> hom-ref
        .otherwise(DEFAULT_MISSING)          # present but allele mismatch -> missing
        .alias("_dose")
    )
    return joined["_dose"].cast(pl.Float64).to_numpy()


def infer_ancestry(
    model_dir: Path,
    genotypes_lf: pl.LazyFrame,
    *,
    panel: str,
    build: str,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    assume_homref_absent: bool = True,
) -> AncestryInference:
    """Infer a single sample's ancestry against a (panel, build) reference model.

    ``assume_homref_absent`` (default True) treats model sites absent from the sample as
    hom-ref — correct for a variant-only VCF; set False for genotyping-array input.
    """
    with start_action(action_type="ancestry:infer", panel=panel, build=build):
        m = load_ancestry_model(model_dir, panel, build)
        dose = _sample_dosage_vector(m.sites, genotypes_lf, assume_homref_absent)
        n_model = int(m.sites.height)
        n_used = int(np.sum(dose != DEFAULT_MISSING))
        coverage = n_used / n_model if n_model else 0.0

        if n_used == 0 or coverage < coverage_floor:
            return AncestryInference(
                panel=panel, genome_build=build, superpopulation=_UNKNOWN,
                probabilities={}, pc_coords=[],
                n_variants_used=n_used, n_variants_model=n_model,
                coverage=coverage, confidence=0.0,
            )

        W = dose.reshape((-1, 1))
        pcs = project_samples(m.pca, W)[0]  # (dim_ref,)
        k = int(m.meta.get("knn_k", DEFAULT_KNN_K))
        label, probs = _knn_predict(m.ref_pcs, m.ref_labels, pcs, k)
        return AncestryInference(
            panel=panel, genome_build=build, superpopulation=label,
            probabilities=probs, pc_coords=[float(x) for x in pcs],
            n_variants_used=n_used, n_variants_model=n_model,
            coverage=coverage, confidence=float(probs.get(label, 0.0)),
        )

"""Offline tests for Level-1 sample genetic-ancestry inference.

All synthetic + local — no plink2, no network, no reference panel. The plink2 build
path is validated separately by the pipeline (``check_ancestry_model_valid`` + a real
``pipeline ancestry-model`` run), not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from just_prs import PRSCatalog
from just_prs.ancestry import build_ancestry_model, infer_ancestry, load_ancestry_model
from just_prs.ancestry._projection import build_reference_pca, project_samples
from just_prs.prs_catalog import _broad_to_superpop

RNG = np.random.default_rng(7)
_GT = {0: "0/0", 1: "0/1", 2: "1/1"}


def _two_pop_panel(p=1500, na=200, nb=200, n_div=400):
    """Synthetic 2-population genotype matrix (variants x samples) + labels."""
    freqA = RNG.uniform(0.05, 0.5, p)
    freqB = freqA.copy()
    div = RNG.choice(p, n_div, replace=False)
    freqB[div] = RNG.uniform(0.05, 0.5, n_div)

    def draw(freq, m):
        return RNG.binomial(2, freq[:, None] * np.ones((1, m))).astype(np.int8)

    X = np.hstack([draw(freqA, na), draw(freqB, nb)])
    return X, freqA, freqB


def _genotypes_lf(dosage: np.ndarray, p: int) -> pl.LazyFrame:
    return pl.DataFrame({
        "chrom": ["1"] * p,
        "pos": list(range(1, p + 1)),
        "ref": ["A"] * p,
        "alt": ["G"] * p,
        "GT": [_GT[int(d)] for d in dosage],
    }).lazy()


def test_projection_separates_and_classifies():
    """OADP-projected held-out samples land near their source population."""
    X, freqA, freqB = _two_pop_panel()
    labels = np.array(["A"] * 200 + ["B"] * 200)
    model = build_reference_pca(X, dim_ref=10)
    assert model.U.shape[1] == model.dim_online
    assert model.pcs_ref.shape == (400, 10)

    test = np.hstack([
        RNG.binomial(2, freqA[:, None] * np.ones((1, 20))).astype(np.int8),
        RNG.binomial(2, freqB[:, None] * np.ones((1, 20))).astype(np.int8),
    ])
    tlab = np.array(["A"] * 20 + ["B"] * 20)
    pcs = project_samples(model, test)
    ref = model.pcs_ref
    acc = sum(
        labels[np.argmin(np.linalg.norm(ref[:, :4] - pcs[i, :4], axis=1))] == tlab[i]
        for i in range(len(tlab))
    ) / len(tlab)
    assert acc >= 0.95


def _build_synthetic_model(model_dir: Path, panel="1000g", build="GRCh38"):
    p, na, nb = 1500, 250, 250
    X, freqA, freqB = _two_pop_panel(p=p, na=na, nb=nb)
    sites = pl.DataFrame({"chrom": ["1"] * p, "pos": list(range(1, p + 1)),
                          "ref": ["A"] * p, "alt": ["G"] * p})
    labels = pl.DataFrame({
        "iid": [f"s{i}" for i in range(na + nb)],
        "superpop": ["EUR"] * na + ["EAS"] * nb,
        "population": ["x"] * (na + nb),
    })
    info = build_ancestry_model(X, sites, labels, panel=panel, build=build, model_dir=model_dir)
    return info, freqB, p


def test_build_persists_and_infers(tmp_path):
    md = tmp_path / "ancestry"
    info, freqB, p = _build_synthetic_model(md)
    assert info["loo_accuracy"] >= 0.95
    assert info["n_variants"] == p
    # artifact round-trips
    m = load_ancestry_model(md, "1000g", "GRCh38")
    assert m.ref_pcs.shape[1] == 10

    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    res = infer_ancestry(md, _genotypes_lf(eas, p), panel="1000g", build="GRCh38")
    assert res.superpopulation == "EAS"
    assert res.coverage == pytest.approx(1.0)
    assert res.confidence > 0.5
    assert res.probabilities


def test_coverage_floor_returns_unknown(tmp_path):
    md = tmp_path / "ancestry"
    _info, freqB, p = _build_synthetic_model(md)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    sparse = _genotypes_lf(eas, p).collect().sample(fraction=0.1, seed=3).lazy()
    res = infer_ancestry(md, sparse, panel="1000g", build="GRCh38", coverage_floor=0.2)
    assert res.superpopulation == "UNKNOWN"
    assert res.coverage < 0.2


def test_allele_swap_is_matched(tmp_path):
    """A sample with ref/alt swapped relative to the model still scores (orientation-aware)."""
    md = tmp_path / "ancestry"
    _info, freqB, p = _build_synthetic_model(md)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    # swap ref<->alt and invert dosage accordingly
    swapped = pl.DataFrame({
        "chrom": ["1"] * p, "pos": list(range(1, p + 1)),
        "ref": ["G"] * p, "alt": ["A"] * p,
        "GT": [_GT[int(2 - d)] for d in eas],
    }).lazy()
    res = infer_ancestry(md, swapped, panel="1000g", build="GRCh38")
    assert res.coverage == pytest.approx(1.0)
    assert res.superpopulation == "EAS"


def test_broad_to_superpop_mapping():
    assert _broad_to_superpop("European") == "EUR"
    assert _broad_to_superpop("East Asian") == "EAS"
    assert _broad_to_superpop("Hispanic or Latin American") == "AMR"
    assert _broad_to_superpop("European, East Asian") is None  # admixed -> ambiguous
    assert _broad_to_superpop(None) is None
    assert _broad_to_superpop("Greys from Zeta Reticuli") is None


def _catalog_with_dev(tmp_path: Path, dev_broad: str, dist: dict) -> PRSCatalog:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    pl.DataFrame({"pgs_id": ["PGS000001"], "genome_build": ["GRCh37"],
                  "trait_reported": ["t2d"]}).write_parquet(meta / "scores.parquet")
    for f in ("performance.parquet", "best_performance.parquet"):
        pl.DataFrame({"pgs_id": ["PGS000001"]}).write_parquet(meta / f)
    pl.DataFrame({
        "pgs_id": ["PGS000001"], "dev_ancestry_broad": [dev_broad],
        "dev_ancestry_distribution": [json.dumps(dist)], "dev_is_multi_ancestry": [len(dist) > 1],
        "dev_ancestries": [list(dist)], "dev_sample_size": [1000],
        "dev_gwas_sample_size": [1000], "dev_training_sample_size": [0],
        "dev_n_ancestries": [len(dist)], "gwas_ancestry_broad": [dev_broad],
        "training_ancestry_broad": [None],
    }).write_parquet(meta / "score_development_ancestry.parquet")
    # Local distributions so reference_data_status reads locally (no HF pull).
    pctl = tmp_path / "percentiles"
    pctl.mkdir(parents=True)
    pl.DataFrame({
        "pgs_id": ["PGS000001"] * 2, "superpopulation": ["EUR", "EAS"],
        "mean": [0.0, 0.0], "std": [1.0, 1.0],
    }).write_parquet(pctl / "1000g_distributions.parquet")
    return PRSCatalog(cache_dir=tmp_path)


def test_coherence_dev_mismatch(tmp_path):
    cat = _catalog_with_dev(tmp_path, "East Asian", {"East Asian": 0.9, "European": 0.1})
    v = cat.assess_ancestry_coherence("PGS000001", "EUR")
    assert v.level == "dev_mismatch"
    assert v.dev_ancestry == "EAS"
    assert v.dev_sample_fraction == pytest.approx(0.1)
    assert v.reliable is False
    assert "less accurate" in v.message


def test_coherence_coherent(tmp_path):
    cat = _catalog_with_dev(tmp_path, "East Asian", {"East Asian": 0.9, "European": 0.1})
    v = cat.assess_ancestry_coherence("PGS000001", "EAS")
    assert v.level == "coherent"
    assert v.reliable is True


def test_coherence_unknown_sample(tmp_path):
    cat = _catalog_with_dev(tmp_path, "European", {"European": 1.0})
    v = cat.assess_ancestry_coherence("PGS000001", None)
    assert v.level == "unknown"
    assert v.reliable is True


def test_prscatalog_infer_defaults_grch38(tmp_path):
    """PRSCatalog.infer_ancestry defaults the build to GRCh38 and loads the local model."""
    md = tmp_path / "ancestry"
    _info, freqB, p = _build_synthetic_model(md, build="GRCh38")
    # minimal cleaned metadata so PRSCatalog constructs without FTP
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    for f in ("scores.parquet", "performance.parquet", "best_performance.parquet"):
        pl.DataFrame({"pgs_id": ["PGS000001"]}).write_parquet(meta / f)
    cat = PRSCatalog(cache_dir=tmp_path)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    res = cat.infer_ancestry(genotypes_lf=_genotypes_lf(eas, p), panel="1000g")
    assert res.genome_build == "GRCh38"
    assert res.superpopulation == "EAS"

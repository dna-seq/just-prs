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


def test_variant_only_vcf_imputes_homref_absent(tmp_path):
    """A variant-only VCF (hom-ref sites omitted) must still classify correctly.

    Regression: treating absent common sites as missing→mean shrinks the projection
    toward the centroid and collapses every sparse WGS sample onto the central (AMR)
    cluster. Absent sites must be imputed hom-ref (ALT dosage 0).
    """
    md = tmp_path / "ancestry"
    _info, freqB, p = _build_synthetic_model(md)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    # variant-only: keep only sites where the sample carries an ALT (dosage > 0)
    full = _genotypes_lf(eas, p).collect()
    variant_only = full.filter(pl.col("GT") != "0/0").lazy()

    res = infer_ancestry(md, variant_only, panel="1000g", build="GRCh38")
    assert res.superpopulation == "EAS"
    assert res.coverage > 0.9  # absent hom-ref sites recovered

    # With array semantics (absent = untyped), coverage drops to the observed fraction.
    res_array = infer_ancestry(md, variant_only, panel="1000g", build="GRCh38",
                               assume_homref_absent=False)
    assert res_array.coverage < res.coverage


def test_mixture_proportions_populated(tmp_path):
    """Level-2: infer returns simplex ancestry proportions dominated by the true pop."""
    md = tmp_path / "ancestry"
    _info, freqB, p = _build_synthetic_model(md)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    res = infer_ancestry(md, _genotypes_lf(eas, p), panel="1000g", build="GRCh38")
    assert res.mixture_method == "pca_nnls"
    assert res.mixture is not None
    assert sum(res.mixture.values()) == pytest.approx(1.0, abs=0.01)
    assert all(v >= -1e-6 for v in res.mixture.values())
    # dominant component is the true (EAS) population
    assert max(res.mixture, key=res.mixture.get) == "EAS"


def test_bayesian_consensus_and_canonical():
    from just_prs.ancestry import bayesian_consensus, to_canonical_superpops

    # HGDP gnomAD labels fold to the canonical 5: CSA->SAS, MID dropped + renormalized.
    canon = to_canonical_superpops({"EUR": 0.5, "CSA": 0.3, "MID": 0.2})
    assert canon["SAS"] == pytest.approx(0.3 / 0.8)  # MID's 0.2 dropped, renormalized
    assert canon["EUR"] == pytest.approx(0.5 / 0.8)
    assert sum(canon.values()) == pytest.approx(1.0)

    # Agreement across methods sharpens the posterior to ~1.
    agree = [{"EUR": 0.9, "AMR": 0.1}, {"EUR": 0.85, "SAS": 0.15}, {"EUR": 1.0}]
    label, post = bayesian_consensus(agree)
    assert label == "EUR" and post["EUR"] > 0.95 and sum(post.values()) == pytest.approx(1.0)

    # Disagreement flattens it (lower top posterior than the agreement case).
    _, post_d = bayesian_consensus([{"EUR": 0.9}, {"EAS": 0.9}])
    assert max(post_d.values()) < 0.95


def test_infer_ancestry_consensus(tmp_path):
    """Fuse two synthetic panels' KNN + mixture into a Bayesian consensus."""
    md = tmp_path / "ancestry"
    _info, freqB, p = _build_synthetic_model(md, panel="1000g")
    _build_synthetic_model(md, panel="hgdp_1kg")  # second panel, same EUR/EAS structure
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    for f in ("scores.parquet", "performance.parquet", "best_performance.parquet"):
        pl.DataFrame({"pgs_id": ["PGS000001"]}).write_parquet(meta / f)
    cat = PRSCatalog(cache_dir=tmp_path)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    con = cat.infer_ancestry_consensus(genotypes_lf=_genotypes_lf(eas, p),
                                       panels=("1000g", "hgdp_1kg"))
    assert con.consensus_superpopulation == "EAS"
    assert sum(con.posterior.values()) == pytest.approx(1.0, abs=0.01)
    assert len(con.per_panel) == 2
    assert len(con.methods) == 4  # 2 panels x (knn + mixture)


def test_coverage_floor_returns_unknown(tmp_path):
    md = tmp_path / "ancestry"
    _info, freqB, p = _build_synthetic_model(md)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    sparse = _genotypes_lf(eas, p).collect().sample(fraction=0.1, seed=3).lazy()
    # array semantics (absent = untyped, not hom-ref) so sparse input is genuinely low-coverage
    res = infer_ancestry(md, sparse, panel="1000g", build="GRCh38", coverage_floor=0.2,
                         assume_homref_absent=False)
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


def test_prive_group_to_continental():
    from just_prs.ancestry.prive import prive_group_to_continental as g
    # All 21 real Privé group names roll up correctly.
    for n in ("Africa (West)", "Africa (South)", "Africa (East)", "Africa (North)"):
        assert g(n) == "AFR"
    for n in ("Ashkenazi", "Italy", "Europe (South East)", "Europe (North East)", "Finland",
              "Scandinavia", "United Kingdom", "Ireland", "Europe (South West)"):
        assert g(n) == "EUR"
    for n in ("Sri Lanka", "Pakistan", "Bangladesh"):
        assert g(n) == "SAS"
    for n in ("Asia (East)", "Japan", "Philippines"):
        assert g(n) == "EAS"
    assert g("South America") == "AMR"
    assert g("Middle East") is None  # no clean 1000G super-pop


def test_build_prive_reference(tmp_path):
    import json
    from just_prs.ancestry.prive import build_prive_reference, _ref_paths

    freq = tmp_path / "f.csv"
    proj = tmp_path / "p.csv"
    base = {"chr": ["1", "1", "2"], "pos": [10, 20, 30],
            "a0": ["A", "C", "G"], "a1": ["G", "T", "A"], "rsid": ["r1", "r2", "r3"]}
    pl.DataFrame({**base, "Italy": [0.1, 0.2, 0.3], "Japan": [0.8, 0.7, 0.6]}).write_csv(freq)
    pcols = dict(base)
    for i in range(1, 17):
        pcols[f"PC{i}"] = [0.01 * i, 0.02 * i, 0.03 * i]
    pl.DataFrame(pcols).write_csv(proj)

    out = tmp_path / "out"
    info = build_prive_reference(freq, proj, out)
    assert info["n_variants"] == 3 and info["n_groups"] == 2
    meta = json.loads(_ref_paths(out)["meta"].read_text())
    assert meta["groups"] == ["Italy", "Japan"]
    assert meta["group_to_continental"] == {"Italy": "EUR", "Japan": "EAS"}
    assert len(meta["correction"]) == 16
    ref = pl.read_parquet(_ref_paths(out)["parquet"])
    assert {"load0", "load15", "Italy", "Japan", "chr", "pos", "a0", "a1"} <= set(ref.columns)


def test_population_resolution(tmp_path):
    """resolution='population' classifies fine pops + keeps the broad super-pop rollup."""
    md = tmp_path / "ancestry"
    p, na, nb = 1200, 200, 200
    X, freqA, freqB = _two_pop_panel(p=p, na=na, nb=nb)
    sites = pl.DataFrame({"chrom": ["1"] * p, "pos": list(range(1, p + 1)),
                          "ref": ["A"] * p, "alt": ["G"] * p})
    # distinct populations nested under super-pops
    labels = pl.DataFrame({
        "iid": [f"s{i}" for i in range(na + nb)],
        "superpop": ["EUR"] * na + ["EAS"] * nb,
        "population": ["French"] * na + ["Japanese"] * nb,
    })
    build_ancestry_model(X, sites, labels, panel="hgdp_1kg", build="GRCh38", model_dir=md)
    eas = RNG.binomial(2, freqB[:, None]).astype(np.int8).flatten()
    gt = _genotypes_lf(eas, p)

    fine = infer_ancestry(md, gt, panel="hgdp_1kg", build="GRCh38", resolution="population")
    assert fine.fine_population == "Japanese"      # fine call
    assert fine.superpopulation == "EAS"           # broad rollup kept for coherence/consumers
    assert "Japanese" in fine.probabilities

    # default resolution unchanged: super-pop label, no fine call
    broad = infer_ancestry(md, gt, panel="hgdp_1kg", build="GRCh38", resolution="superpop")
    assert broad.superpopulation == "EAS" and broad.fine_population is None


def test_tgeno_reader_roundtrip(tmp_path):
    """Write a tiny packed TGENO, read it back, verify genotypes + missing handling."""
    import math
    from just_prs.ancestry.eigenstrat import read_tgeno, TGENO_HEADER_BYTES
    n_ind, n_snp = 5, 7
    rng = np.random.default_rng(0)
    # 0/1/2 genotypes + some missing (3) — what read_tgeno returns as 0/1/2/-9
    truth = rng.integers(0, 3, size=(n_ind, n_snp)).astype(np.int8)
    truth[0, 1] = -9  # a missing call
    rlen = math.ceil(n_snp / 4)
    geno = tmp_path / "t.geno"
    with open(geno, "wb") as fh:
        fh.write(b"TGENO".ljust(TGENO_HEADER_BYTES, b"\x00"))
        for i in range(n_ind):
            vals = [3 if truth[i, j] == -9 else int(truth[i, j]) for j in range(n_snp)]
            vals += [0] * (rlen * 4 - n_snp)
            rec = bytearray(rlen)
            for j, v in enumerate(vals):
                rec[j // 4] |= (v & 3) << (6 - 2 * (j % 4))  # MSB-first
            fh.write(bytes(rec))
    got = read_tgeno(geno, n_ind, n_snp, np.array([0, 2, 4]))
    assert np.array_equal(got, truth[[0, 2, 4]])
    assert (got == -9).sum() == 1  # the injected missing


def test_aadr_panel_lifts_to_grch37():
    """PRSCatalog routes the aadr_ho panel to its GRCh37 model build (not GRCh38)."""
    from just_prs.prs_catalog import _ANCESTRY_PANEL_BUILD
    assert _ANCESTRY_PANEL_BUILD.get("aadr_ho") == "GRCh37"
    assert _ANCESTRY_PANEL_BUILD.get("1000g", "GRCh38") == "GRCh38"

"""Dagster lineage for the sample-ancestry reference-PCA model.

    pgsc_reference_panel (SourceAsset)
        -> ancestry_pca_model (build: plink2 QC/LD-prune + numpy SVD, per panel x build)
        -> hf_ancestry_model (upload to HuggingFace)

plink2 is **build-time only** (QC + LD-pruning). The published artifact is small
(pruned-site loadings + reference PC scores + meta); runtime projection is pure-Python
(``just_prs.ancestry``), so no plink2 is needed after this pipeline runs.

Build approach (avoids variant-ID / psam-column pitfalls): plink2 QC + ``--indep-pairwise``
then ``--make-pgen`` writes a small pruned panel, which we read back via the existing
``parse_pvar`` / ``read_pgen_genotypes`` utilities; sample labels are joined from the
original ``.psam`` by IID (plink2 may drop custom psam columns).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from dagster import AssetDep, AssetExecutionContext, AssetIn, Output, SourceAsset, asset
from eliot import start_action

from just_prs.ancestry import build_ancestry_model
from just_prs.hf import push_ancestry_model
from just_prs.reference import (
    REFERENCE_PANELS,
    _build_name_tokens,
    _ResolvedRefPanel,
    download_reference_panel,
    parse_pvar,
    read_pgen_genotypes,
    reference_panel_dir,
)
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource
from prs_pipeline.runtime import resource_tracker

_AMBIGUOUS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
# Ancestry models are built in GRCh38 only; GRCh37 samples are lifted to GRCh38 at
# inference time (the hom-ref-absent imputation that makes projection work is inline at
# the model's pruned sites, so a native GRCh37 model would be redundant compute).
_DEFAULT_PANELS = ("1000g", "hgdp_1kg")
_DEFAULT_BUILDS = ("GRCh38",)


def _selected(env_var: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(env_var, "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else list(default)


def _resolve_plink2(cache_dir: Path) -> str:
    """Resolve the plink2 binary: PLINK2_BIN env > PATH > <cache>/plink2/plink2."""
    env = os.environ.get("PLINK2_BIN", "").strip()
    if env:
        return env
    which = shutil.which("plink2")
    if which:
        return which
    cached = cache_dir / "plink2" / "plink2"
    if cached.exists():
        return str(cached)
    raise FileNotFoundError(
        "plink2 not found. Install it (apt/conda) or set PLINK2_BIN. "
        "plink2 is build-time only; runtime never needs it."
    )


def _panel_files(panel: str, build: str, cache_dir: Path):
    """Resolve (pfile_prefix, psam_df, king_remove_file) for a panel x build.

    Downloads the panel if missing (HGDP+1kGP is ~16 GB), then uses the robust
    ``_ResolvedRefPanel`` resolver (handles GRCh38/hg38 token naming) so it works for
    both 1000G and HGDP+1kGP. plink2 ``--pfile`` needs the pgen/pvar/psam to share a
    prefix, which the PGS Catalog panels do.
    """
    download_reference_panel(cache_dir, panel=panel)  # no-op if already extracted
    ref_dir = reference_panel_dir(cache_dir, panel=panel)
    resolved = _ResolvedRefPanel(ref_dir, genome_build=build)
    prefix = str(resolved.pgen_path)[: -len(".pgen")]
    tokens = _build_name_tokens(build)
    kings = [k for k in sorted(ref_dir.glob("*king.cutoff.out.id"))
             if any(t in k.name for t in tokens)]
    king = kings[0] if kings else None
    return prefix, resolved.psam_df, king


def _psam_iids(psam_path: Path) -> list[str]:
    """Read the IID column from a .psam in file order (handles plink2-minimal psam)."""
    df = pl.read_csv(psam_path, separator="\t", comment_prefix="##", infer_schema=False)
    col = "#IID" if "#IID" in df.columns else ("IID" if "IID" in df.columns else df.columns[0])
    return df[col].to_list()


def _run_plink2(plink2: str, args: list[str], log) -> None:
    cmd = [plink2, *args]
    log(f"plink2 {' '.join(args)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"plink2 failed ({res.returncode}): {res.stderr[-2000:]}")


def _ld_pruned_panel(plink2: str, prefix: str, king: Path | None, workdir: Path, log) -> str:
    """QC + LD-prune the panel, return a small pruned pfile prefix (vzs .pvar.zst)."""
    remove = ["--remove", str(king)] if king else []
    qc = [
        "--allow-extra-chr", "--autosome",
        "--maf", "0.05", "--min-alleles", "2", "--max-alleles", "2",
        "--snps-only", "just-acgt", "--rm-dup", "exclude-all", "--hwe", "1e-4",
    ]
    step1 = str(workdir / "step1")
    _run_plink2(plink2, ["--pfile", prefix, "vzs", *remove, *qc,
                         "--indep-pairwise", "1000", "50", "0.05", "--out", step1], log)
    pruned = str(workdir / "pruned")
    _run_plink2(plink2, ["--pfile", prefix, "vzs", "--allow-extra-chr", *remove,
                         "--extract", f"{step1}.prune.in",
                         "--make-pgen", "vzs", "--out", pruned], log)
    return pruned


def _build_one(panel: str, build: str, cache_dir: Path, plink2: str, model_dir: Path, log) -> dict:
    """Build + persist one (panel, build) ancestry model. Returns build metadata."""
    prefix, psam_df, king = _panel_files(panel, build, cache_dir)
    with tempfile.TemporaryDirectory(prefix=f"anc_{panel}_{build}_") as tmp:
        workdir = Path(tmp)
        pruned = _ld_pruned_panel(plink2, prefix, king, workdir, log)
        pruned_pvar = Path(f"{pruned}.pvar.zst")
        pruned_pgen = Path(f"{pruned}.pgen")
        pruned_psam = Path(f"{pruned}.psam")

        pvar = parse_pvar(pruned_pvar)  # variant_idx, chrom, POS, ID, REF, ALT
        keep = pvar.filter(
            ~pl.struct(["REF", "ALT"]).map_elements(
                lambda s: (s["REF"], s["ALT"]) in _AMBIGUOUS, return_dtype=pl.Boolean
            )
        )
        sites = keep.select(
            pl.col("chrom").cast(pl.Utf8),
            pl.col("POS").cast(pl.Int64).alias("pos"),
            pl.col("REF").cast(pl.Utf8).alias("ref"),
            pl.col("ALT").cast(pl.Utf8).alias("alt"),
        )
        var_idx = keep["variant_idx"].cast(pl.UInt32).to_numpy()

        iids = _psam_iids(pruned_psam)
        genos = read_pgen_genotypes(pruned_pgen, pruned_pvar, var_idx, len(iids))  # (n_var x n_samp)

        # Labels from the ORIGINAL psam (plink2 may drop SuperPop/Population), aligned to IID order.
        lab = {r["iid"]: (r["superpop"], r["population"]) for r in psam_df.iter_rows(named=True)}
        labels = pl.DataFrame({
            "iid": iids,
            "superpop": [lab.get(i, ("NR", "NR"))[0] for i in iids],
            "population": [lab.get(i, ("NR", "NR"))[1] for i in iids],
        })

        info = build_ancestry_model(
            genos, sites, labels, panel=panel, build=build, model_dir=model_dir
        )
        info.update({"panel": panel, "build": build})
        return info


pgsc_reference_panel = SourceAsset(
    key="pgsc_reference_panel",
    group_name="external",
    description=(
        "PGS Catalog processed reference panels at EBI FTP (1000G, HGDP+1kGP; both "
        "GRCh37 and GRCh38). Genotypes + KING unrelated-sample lists used to build the "
        "ancestry-PCA model. Already consumed by reference scoring."
    ),
    metadata={
        "url_1000g": REFERENCE_PANELS["1000g"]["url"],
        "url_hgdp_1kg": REFERENCE_PANELS["hgdp_1kg"]["url"],
    },
)


@asset(
    group_name="compute",
    description=(
        "Builds the sample-ancestry reference-PCA model per panel x build: plink2 QC + "
        "LD-prune (build-time only), numpy SVD (OADP-compatible), KNN leave-one-out "
        "validation. Writes small loadings/refpcs/meta artifacts to <cache>/ancestry/. "
        "Panels/builds selectable via PRS_ANCESTRY_PANELS / PRS_ANCESTRY_BUILDS."
    ),
)
def ancestry_pca_model(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[list[dict]]:
    cache_dir = cache_dir_resource.get_path()
    model_dir = cache_dir / "ancestry"
    plink2 = _resolve_plink2(cache_dir)
    panels = _selected("PRS_ANCESTRY_PANELS", _DEFAULT_PANELS)
    builds = _selected("PRS_ANCESTRY_BUILDS", _DEFAULT_BUILDS)

    results: list[dict] = []
    with resource_tracker("ancestry_pca_model", context=context):
        for panel in panels:
            for build in builds:
                with start_action(action_type="pipeline:ancestry_pca_model", panel=panel, build=build):
                    try:
                        info = _build_one(panel, build, cache_dir, plink2, model_dir, context.log.info)
                        context.log.info(
                            f"{panel}/{build}: {info['n_variants']:,} pruned variants, "
                            f"{info['n_reference']:,} refs, LOO={info['loo_accuracy']:.3f}"
                        )
                        results.append(info)
                    except Exception as exc:  # noqa: BLE001 - keep building other models
                        context.log.error(f"{panel}/{build} build failed: {exc}")
                        results.append({"panel": panel, "build": build, "error": str(exc)})

    ok = [r for r in results if "error" not in r]
    context.add_output_metadata({
        "n_models": len(ok),
        "n_failed": len(results) - len(ok),
        "models": str([(r["panel"], r["build"]) for r in ok]),
        "min_loo_accuracy": round(min((r["loo_accuracy"] for r in ok), default=0.0), 4),
        "model_dir": str(model_dir),
    })
    return Output(results)


@asset(
    group_name="upload",
    ins={"models": AssetIn("ancestry_pca_model")},
    description="Pushes each built ancestry-PCA model to HuggingFace (data/ancestry/).",
)
def hf_ancestry_model(
    context: AssetExecutionContext,
    models: list[dict],
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    model_dir = cache_dir_resource.get_path() / "ancestry"
    repo_id = hf_resource.percentiles_repo
    token = hf_resource.get_token()
    ok = [r for r in models if "error" not in r]

    if os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip():
        context.add_output_metadata({"test_mode": True, "hf_push_skipped": True, "n_models": len(ok)})
        return Output(str(model_dir))

    pushed = 0
    with resource_tracker("hf_ancestry_model", context=context):
        for r in ok:
            push_ancestry_model(model_dir, r["panel"], r["build"], repo_id=repo_id, token=token)
            pushed += 1

    url = f"https://huggingface.co/datasets/{repo_id}"
    context.add_output_metadata({"repo_id": repo_id, "url": url, "n_pushed": pushed})
    return Output(url)

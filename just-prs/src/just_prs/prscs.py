"""Build a polygenic score from GWAS summary statistics with PRS-CS.

PRS-CS (Ge et al. 2019, MIT licence) shrinks marginal GWAS effects with a
continuous-shrinkage prior and an external LD reference. ``PRS-CS-auto`` (no
``phi``) learns the global shrinkage from the data, so no tuning cohort is
needed — the right default for a personal-genomics tool where no validation
phenotypes exist.

The flow, each step cached and resumable:

1. :func:`ensure_prscs` — the pinned PRS-CS source (:data:`PRSCS_COMMIT`).
2. :func:`download_prscs_ld_reference` — HapMap3 LD blocks (UK Biobank or
   1000 Genomes, per ancestry), tarball discarded after extraction.
3. :func:`write_prscs_inputs` — normalized sumstats (:mod:`just_prs.sumstats`)
   filtered to the reference SNPs, plus a synthetic ``.bim`` that lists the
   whole reference. Using the reference as the "target" set makes the score
   sample-agnostic: one build scores any genome.
4. :func:`run_prscs` — one subprocess per chromosome, largest first, bounded
   workers, one BLAS thread each, progress forwarded to ``log``.
5. :func:`prscs_to_scoring_file` — concatenate posterior effects, attach the
   GRCh38 coordinates from the harmonised sumstats (the LD reference is
   GRCh37, so no liftover is needed), the reference-panel REF base and the
   reference-population effect-allele frequency, and write a just-prs scoring
   parquet with header metadata.

:func:`build_prscs_score` chains all of it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import psutil
from eliot import log_message, start_action
from pydantic import BaseModel

from just_prs.custom_scores import custom_score_path, custom_scores_dir
from just_prs.scoring import _write_scoring_parquet, resolve_cache_dir
from just_prs.sumstats import (
    GwasSumstatsSource,
    SumstatsNormalization,
    _duckdb_connect,
    download_gwas_sumstats,
    normalize_gwas_sumstats,
)

PRSCS_COMMIT = "53303906784c3360b9a1a7fa000b32c73eda9d5a"
PRSCS_ARCHIVE_URL = f"https://github.com/getian107/PRScs/archive/{PRSCS_COMMIT}.tar.gz"

_BROAD = "https://personal.broadinstitute.org/hhuang/public/PRS-CSx/Reference"
#: ``(panel, population) -> tarball URL``. The Broad mirror serves byte ranges,
#: which the README's Dropbox links do not do reliably (measured ~110 KB/s).
PRSCS_LD_REFERENCES: dict[tuple[str, str], str] = {
    **{("ukbb", pop): f"{_BROAD}/UKBB/ldblk_ukbb_{pop}.tar.gz" for pop in ("afr", "amr", "eas", "eur", "sas")},
    **{("1kg", pop): f"{_BROAD}/1KG/ldblk_1kg_{pop}.tar.gz" for pop in ("afr", "amr", "eas", "eur", "sas")},
}

#: Chromosomes ordered by HapMap3 SNP count so the longest jobs start first.
_CHROM_ORDER: tuple[int, ...] = (1, 2, 6, 3, 4, 5, 7, 11, 10, 8, 12, 9, 13, 14, 16, 15, 18, 20, 17, 19, 22, 21)
#: Resident memory one PRS-CS chromosome job can reach (chr1/chr2 LD blocks).
_PRSCS_JOB_GB = float(os.environ.get("PRS_PRSCS_JOB_MEMORY_GB", "3"))

Log = Callable[[str], None]


class PrscsInputs(BaseModel):
    """Files handed to PRS-CS for one score."""

    work_dir: Path
    sst_file: Path
    bim_prefix: Path
    variants_file: Path
    n_sumstats_rows: int
    n_reference_snps: int
    n_matched: int


class PrscsRun(BaseModel):
    """Per-chromosome posterior-effect files from one PRS-CS run."""

    outputs: dict[int, Path]
    n_cached: int
    n_computed: int
    phi_by_chrom: dict[int, float] = {}


class CustomScoreBuild(BaseModel):
    """Result of :func:`build_prscs_score`."""

    score_id: str
    scoring_file: Path
    n_variants: int
    n_gwas: int
    gcst: str
    trait_reported: str
    inputs: PrscsInputs
    run: PrscsRun
    header: dict[str, str]


def _print(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# PRS-CS code + LD reference
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, log: Log) -> None:
    """Resumable download; uses ``aria2c`` (parallel ranges) when on PATH."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    aria2 = shutil.which("aria2c")
    if aria2:
        subprocess.run(
            [aria2, "-q", "-x16", "-s16", "-k20M", "--max-tries=0", "--retry-wait=10",
             "--async-dns=false", "--auto-file-renaming=false", "-c",
             "-d", str(dest.parent), "-o", dest.name, url],
            check=True,
        )
        return
    part = dest.with_name(dest.name + ".part")
    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True) as http:
        for attempt in range(20):
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with http.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416:
                        break
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with part.open(mode) as handle:
                        for chunk in response.iter_bytes(chunk_size=1 << 22):
                            handle.write(chunk)
                break
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                log(f"download retry {attempt + 1} for {url}: {exc}")
                time.sleep(10)
    os.replace(part, dest)


def ensure_prscs(cache_dir: Path | None = None, log: Log = _print) -> Path:
    """Return the directory holding the pinned ``PRScs.py``, downloading it once."""
    root = (cache_dir or resolve_cache_dir()) / "prscs"
    target = root / f"PRScs-{PRSCS_COMMIT}"
    if (target / "PRScs.py").is_file():
        return target
    with start_action(action_type="prscs:install", commit=PRSCS_COMMIT):
        tarball = root / f"PRScs-{PRSCS_COMMIT}.tar.gz"
        _download(PRSCS_ARCHIVE_URL, tarball, log)
        with tarfile.open(tarball) as tar:
            tar.extractall(root, filter="data")
        tarball.unlink()
    return target


def download_prscs_ld_reference(
    panel: str = "ukbb",
    population: str = "eur",
    cache_dir: Path | None = None,
    log: Log = _print,
) -> Path:
    """Download and extract a PRS-CS LD reference; return ``ldblk_<panel>_<pop>/``.

    The panel must match the ancestry of the **GWAS**, not of the person being
    scored. The tarball (4–7 GB) is discarded once extraction succeeds.
    """
    key = (panel.lower(), population.lower())
    if key not in PRSCS_LD_REFERENCES:
        raise ValueError(f"Unknown PRS-CS LD reference {key}; known: {sorted(PRSCS_LD_REFERENCES)}")
    ld_root = (cache_dir or resolve_cache_dir()) / "prscs" / "ld"
    target = ld_root / f"ldblk_{key[0]}_{key[1]}"
    if (target / f"snpinfo_{key[0]}_hm3").is_file() and len(list(target.glob("*.hdf5"))) == 22:
        return target
    with start_action(action_type="prscs:ld_reference", panel=key[0], population=key[1]):
        tarball = ld_root / f"ldblk_{key[0]}_{key[1]}.tar.gz"
        if not tarball.exists() or tarball.with_name(tarball.name + ".aria2").exists():
            log(f"Downloading PRS-CS LD reference {key[0]}/{key[1]} (several GB)…")
            _download(PRSCS_LD_REFERENCES[key], tarball, log)
        log(f"Extracting {tarball.name}…")
        with tarfile.open(tarball) as tar:
            tar.extractall(ld_root, filter="data")
        tarball.unlink()
    return target


def _snpinfo_path(ld_dir: Path) -> Path:
    panel = "1kg" if "1kg" in ld_dir.name else "ukbb"
    return ld_dir / f"snpinfo_{panel}_hm3"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def write_prscs_inputs(
    normalized_sumstats: Path,
    ld_dir: Path,
    work_dir: Path,
    drop_palindromic: bool = True,
    min_info: float | None = 0.8,
    memory_limit: str | None = None,
) -> PrscsInputs:
    """Write the PRS-CS summary-statistics file and a reference-wide ``.bim``.

    Rows are kept when they are harmonised, single-base A/C/G/T, carry a
    positive standard error, clear ``min_info`` when an INFO score exists, and
    their rsID is in the LD reference. Strand-ambiguous A/T and C/G SNPs are
    dropped by default: their orientation in the harmonised file is inferred,
    and a wrong guess flips the sign of the effect. Duplicate rsIDs keep the
    row with the smallest standard error.

    ``variants.parquet`` keeps each used SNP's GRCh38 coordinates and alleles so
    :func:`prscs_to_scoring_file` can place the posterior effects in GRCh38.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    snpinfo = _snpinfo_path(ld_dir)
    sst_file = work_dir / "sumstats_prscs.txt"
    bim_prefix = work_dir / "reference_snps"
    variants_file = work_dir / "variants.parquet"
    info_filter = f"AND (s.info IS NULL OR s.info >= {float(min_info)})" if min_info is not None else ""
    palindrome_filter = "AND NOT s.palindromic" if drop_palindromic else ""
    con = _duckdb_connect(memory_limit)
    try:
        con.execute(
            "CREATE TEMP TABLE ref AS SELECT * FROM read_csv(?, delim='\t', header=true, "
            "columns={'CHR': 'INTEGER', 'SNP': 'VARCHAR', 'BP': 'BIGINT', 'A1': 'VARCHAR', 'A2': 'VARCHAR', 'MAF': 'DOUBLE'})",
            [str(snpinfo)],
        )
        con.execute(
            f"""
            CREATE TEMP TABLE used AS
            SELECT s.rsid, s.chrom, s.pos, s.effect_allele, s.other_allele, s.beta, s.se, s.eaf
            FROM read_parquet(?) s
            SEMI JOIN ref r ON r.SNP = s.rsid
            WHERE s.pos IS NOT NULL
              AND s.beta IS NOT NULL AND s.se IS NOT NULL AND s.se > 0
              AND regexp_full_match(s.effect_allele, '[ACGT]')
              AND regexp_full_match(s.other_allele, '[ACGT]')
              {palindrome_filter}
              {info_filter}
            QUALIFY row_number() OVER (PARTITION BY s.rsid ORDER BY s.se) = 1
            """,
            [str(normalized_sumstats)],
        )
        con.execute(
            f"COPY (SELECT rsid AS SNP, effect_allele AS A1, other_allele AS A2, beta AS BETA, se AS SE "
            f"FROM used ORDER BY rsid) TO '{sst_file}' (FORMAT csv, DELIMITER '\t', HEADER true)"
        )
        con.execute(f"COPY (SELECT * FROM used) TO '{variants_file}' (FORMAT parquet, COMPRESSION zstd)")
        con.execute(
            f"COPY (SELECT CHR, SNP, 0, BP, A1, A2 FROM ref) TO '{bim_prefix}.bim' (FORMAT csv, DELIMITER '\t', HEADER false)"
        )
        n_rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(normalized_sumstats)]).fetchone()[0]
        n_ref = con.execute("SELECT count(*) FROM ref").fetchone()[0]
        n_used = con.execute("SELECT count(*) FROM used").fetchone()[0]
    finally:
        con.close()
    inputs = PrscsInputs(
        work_dir=work_dir,
        sst_file=sst_file,
        bim_prefix=bim_prefix,
        variants_file=variants_file,
        n_sumstats_rows=n_rows,
        n_reference_snps=n_ref,
        n_matched=n_used,
    )
    log_message(message_type="prscs:inputs", **inputs.model_dump(mode="json"))
    return inputs


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _default_workers(n_jobs: int) -> int:
    available_gb = psutil.virtual_memory().available / 1024**3
    by_ram = int(available_gb * 0.6 // _PRSCS_JOB_GB)
    return max(1, min(n_jobs, os.cpu_count() or 1, by_ram))


def run_prscs(
    inputs: PrscsInputs,
    ld_dir: Path,
    n_gwas: int,
    out_prefix: Path,
    chroms: tuple[int, ...] = _CHROM_ORDER,
    workers: int | None = None,
    phi: float | None = None,
    seed: int = 42,
    n_iter: int = 1000,
    n_burnin: int = 500,
    prscs_dir: Path | None = None,
    log: Log = _print,
) -> PrscsRun:
    """Run PRS-CS per chromosome in parallel subprocesses.

    A chromosome is skipped when its output exists and is newer than the
    summary-statistics input, so an interrupted build resumes where it stopped.
    Each job gets one BLAS thread; ``workers`` defaults to what fits in 60% of
    available RAM at ``PRS_PRSCS_JOB_MEMORY_GB`` (default 3) per job.

    Args:
        inputs: From :func:`write_prscs_inputs`.
        ld_dir: The extracted ``ldblk_*`` directory.
        n_gwas: GWAS sample size (effective N for case/control traits).
        out_prefix: PRS-CS ``--out_dir`` prefix; files become
            ``<prefix>_pst_eff_a1_b0.5_phiauto_chr<N>.txt``.
        phi: Fix the global shrinkage instead of learning it (``None`` = auto).
        log: Receives one line per chromosome start/finish and every
            100 MCMC iterations.
    """
    prscs_dir = prscs_dir or ensure_prscs(log=log)
    phi_tag = "phiauto" if phi is None else f"phi{phi:1.0e}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def output_for(chrom: int) -> Path:
        return out_prefix.parent / f"{out_prefix.name}_pst_eff_a1_b0.5_{phi_tag}_chr{chrom}.txt"

    sst_mtime = inputs.sst_file.stat().st_mtime
    todo = [c for c in chroms if not (output_for(c).exists() and output_for(c).stat().st_mtime >= sst_mtime)]
    n_cached = len(chroms) - len(todo)
    workers = workers or _default_workers(len(todo))
    log(f"PRS-CS: {len(todo)} chromosome(s) to fit, {n_cached} cached, {workers} worker(s), n_gwas={n_gwas}")
    env = {**os.environ, "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    phi_by_chrom: dict[int, float] = {}

    def run_one(chrom: int) -> tuple[int, float | None, float]:
        started = time.monotonic()
        cmd = [
            sys.executable, str(prscs_dir / "PRScs.py"),
            f"--ref_dir={ld_dir}", f"--bim_prefix={inputs.bim_prefix}", f"--sst_file={inputs.sst_file}",
            f"--n_gwas={int(n_gwas)}", f"--out_dir={out_prefix}", f"--chrom={chrom}", f"--seed={seed}",
            f"--n_iter={n_iter}", f"--n_burnin={n_burnin}",
        ]
        if phi is not None:
            cmd.append(f"--phi={phi}")
        log_file = out_prefix.parent / f"{out_prefix.name}_chr{chrom}.log"
        phi_est: float | None = None
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as proc, \
                log_file.open("w") as log_handle:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_handle.write(line)
                text = line.strip()
                if text.startswith("--- iter-"):
                    log(f"  chr{chrom}: MCMC {text.strip('- ')}/{n_iter}")
                elif "Estimated global shrinkage" in text:
                    phi_est = float(text.split(":")[1].strip(" .\n"))
            if proc.wait() != 0:
                raise RuntimeError(f"PRS-CS failed on chr{chrom}; see {log_file}")
        return chrom, phi_est, time.monotonic() - started

    with start_action(action_type="prscs:run", n_chroms=len(todo), workers=workers):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, c): c for c in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                chrom, phi_est, seconds = future.result()
                if phi_est is not None:
                    phi_by_chrom[chrom] = phi_est
                log(f"PRS-CS chr{chrom} done ({done}/{len(todo)}, {seconds / 60:.1f} min"
                    + (f", phi={phi_est:.2e}" if phi_est is not None else "") + ")")

    return PrscsRun(
        outputs={c: output_for(c) for c in chroms},
        n_cached=n_cached,
        n_computed=len(todo),
        phi_by_chrom=phi_by_chrom,
    )


# ---------------------------------------------------------------------------
# Scoring file
# ---------------------------------------------------------------------------


def _reference_panel_pvar(genome_build: str) -> Path | None:
    from just_prs.reference import reference_panel_dir

    candidate = reference_panel_dir(panel="1000g") / f"{genome_build}_1000G_ALL_pvar.parquet"
    return candidate if candidate.exists() else None


def prscs_to_scoring_file(
    run: PrscsRun,
    inputs: PrscsInputs,
    ld_dir: Path,
    out_path: Path,
    header: dict[str, str],
    panel_pvar_path: Path | None = None,
) -> tuple[Path, int]:
    """Convert PRS-CS posterior effects into a just-prs scoring parquet (GRCh38).

    Columns: ``rsID``, ``chr_name``, ``chr_position``, ``effect_allele``,
    ``other_allele``, ``effect_weight``, ``reference_allele``,
    ``allelefrequency_effect``. The effect-allele frequency comes from the LD
    reference population, which enables just-prs's theoretical percentile even
    without a reference-panel distribution. ``reference_allele`` (from the
    1000G panel ``.pvar`` when available) lets a variant-only VCF score absent
    loci as homozygous-reference instead of dropping them.

    Variants whose PRS-CS alleles are not the harmonised GRCh38 alleles (strand
    or build disagreement) are dropped and counted in the header.
    """
    from just_prs.reference_allele import resolve_reference_alleles

    frames = []
    for chrom, path in sorted(run.outputs.items()):
        if not path.exists():
            raise FileNotFoundError(f"Missing PRS-CS output for chr{chrom}: {path}")
        frames.append(
            pl.read_csv(path, separator="\t", has_header=False,
                        new_columns=["chr37", "rsID", "pos37", "a1", "a2", "effect_weight"],
                        schema_overrides={"chr37": pl.Utf8, "a1": pl.Utf8, "a2": pl.Utf8})
        )
    posterior = pl.concat(frames)
    snpinfo = pl.read_csv(_snpinfo_path(ld_dir), separator="\t", schema_overrides={"CHR": pl.Utf8}).select(
        pl.col("SNP").alias("rsID"), pl.col("A1").alias("ref_a1"), pl.col("MAF").alias("ref_a1_freq")
    )
    variants = pl.read_parquet(inputs.variants_file).rename({"rsid": "rsID"})
    joined = (
        posterior.join(variants, on="rsID", how="inner")
        .join(snpinfo, on="rsID", how="left")
        .with_columns(
            pl.when(pl.col("a1") == pl.col("ref_a1")).then(pl.col("ref_a1_freq"))
            .otherwise(1.0 - pl.col("ref_a1_freq")).alias("allelefrequency_effect"),
            (((pl.col("a1") == pl.col("effect_allele")) & (pl.col("a2") == pl.col("other_allele")))
             | ((pl.col("a1") == pl.col("other_allele")) & (pl.col("a2") == pl.col("effect_allele")))).alias("allele_ok"),
        )
    )
    n_mismatch = joined.filter(~pl.col("allele_ok")).height
    scored = joined.filter(pl.col("allele_ok")).select(
        "rsID",
        pl.col("chrom").alias("chr_name"),
        pl.col("pos").alias("chr_position"),
        pl.col("a1").alias("effect_allele"),
        pl.col("a2").alias("other_allele"),
        "effect_weight",
        "allelefrequency_effect",
    )

    pvar = panel_pvar_path or _reference_panel_pvar(header.get("genome_build", "GRCh38"))
    if pvar is not None:
        refs = resolve_reference_alleles(
            scored.select(pl.col("chr_name").alias("chrom"), pl.col("chr_position").alias("pos")),
            header.get("genome_build", "GRCh38"),
            panel_pvar_path=pvar,
        ).select(pl.col("chrom").alias("chr_name"), pl.col("pos").alias("chr_position"), pl.col("ref").alias("reference_allele"))
        scored = scored.join(refs, on=["chr_name", "chr_position"], how="left")
        # A REF that is neither allele means the panel disagrees with the sumstats — do not assume it.
        scored = scored.with_columns(
            pl.when((pl.col("reference_allele") != pl.col("effect_allele"))
                    & (pl.col("reference_allele") != pl.col("other_allele")))
            .then(None).otherwise(pl.col("reference_allele")).alias("reference_allele")
        )
    else:
        scored = scored.with_columns(pl.lit(None, dtype=pl.Utf8).alias("reference_allele"))

    scored = scored.sort("chr_name", "chr_position")
    header = {
        **header,
        "variants_number": str(scored.height),
        "variants_dropped_allele_mismatch": str(n_mismatch),
        "variants_with_reference_allele": str(scored.filter(pl.col("reference_allele").is_not_null()).height),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_scoring_parquet(scored, header, out_path)
    log_message(message_type="prscs:scoring_file", path=str(out_path), **{k: header[k] for k in ("variants_number", "variants_dropped_allele_mismatch")})
    return out_path, scored.height


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def build_prscs_score(
    gcst: str,
    score_id: str,
    trait_reported: str | None = None,
    n_gwas: int | None = None,
    ld_panel: str = "ukbb",
    population: str = "eur",
    workers: int | None = None,
    phi: float | None = None,
    direction: str | None = None,
    cache_dir: Path | None = None,
    memory_limit: str | None = None,
    log: Log = _print,
) -> CustomScoreBuild:
    """Download a GWAS Catalog study and turn it into a PRS-CS-auto scoring file.

    Every step is cached (sumstats download, normalization, PRS-CS inputs,
    per-chromosome posteriors), so re-running after an interruption only redoes
    what is missing. The result lands at :func:`custom_score_path` and is scored
    like any other file: ``prs compute --scoring-file <path>``.

    Args:
        gcst: GWAS Catalog accession with harmonised summary statistics.
        score_id: Short identifier used for file names and result labels.
        trait_reported: Human-readable trait (defaults to the GWAS Catalog trait).
        n_gwas: GWAS sample size; defaults to the study metadata.
        ld_panel / population: PRS-CS LD reference matching the GWAS ancestry.
        direction: Plain-English meaning of a higher score, stored in the header.
    """
    cache_dir = cache_dir or resolve_cache_dir()
    with start_action(action_type="prscs:build", gcst=gcst, score_id=score_id):
        log(f"[1/5] Summary statistics for {gcst}")
        source: GwasSumstatsSource = download_gwas_sumstats(gcst, cache_dir / "sumstats" / "gwas_catalog")
        normalized: SumstatsNormalization = normalize_gwas_sumstats(
            source.path, cache_dir / "sumstats" / "normalized" / f"{source.gcst}.parquet", memory_limit=memory_limit
        )
        log(f"      {normalized.n_rows:,} rows ({normalized.layout.value}; beta={normalized.beta_source}; se={normalized.se_source})")
        n = n_gwas or source.sample_size or (int(normalized.median_n) if normalized.median_n else None)
        if not n:
            raise ValueError(f"No sample size for {gcst}; pass n_gwas explicitly")

        log(f"[2/5] LD reference {ld_panel}/{population}")
        ld_dir = download_prscs_ld_reference(ld_panel, population, cache_dir, log=log)
        prscs_dir = ensure_prscs(cache_dir, log=log)

        score_dir = custom_scores_dir(cache_dir) / score_id
        work = score_dir / "prscs"
        log("[3/5] PRS-CS inputs")
        inputs = write_prscs_inputs(normalized.output, ld_dir, work, memory_limit=memory_limit)
        log(f"      {inputs.n_matched:,} of {inputs.n_reference_snps:,} reference SNPs usable")

        log("[4/5] PRS-CS-auto")
        run = run_prscs(inputs, ld_dir, n, work / score_id, workers=workers, phi=phi, prscs_dir=prscs_dir, log=log)

        log("[5/5] Scoring file")
        header = {
            "pgs_id": score_id,
            "pgs_name": score_id,
            "trait_reported": trait_reported or source.trait or gcst,
            "genome_build": "GRCh38",
            "weight_type": "beta",
            "method_name": "PRS-CS-auto" if phi is None else f"PRS-CS (phi={phi:g})",
            "method_params": f"ld_reference=ldblk_{ld_panel}_{population}; n_gwas={n}; prscs_commit={PRSCS_COMMIT[:7]}; seed=42",
            "source_gwas": gcst,
            "source_url": source.url,
            "gwas_ancestry": ", ".join(source.ancestry),
            "build_date": date.today().isoformat(),
        }
        if direction:
            header["direction"] = direction
        out_path, n_variants = prscs_to_scoring_file(
            run, inputs, ld_dir, custom_score_path(score_id, "GRCh38", cache_dir), header
        )
        log(f"      {n_variants:,} variants -> {out_path}")
        return CustomScoreBuild(
            score_id=score_id,
            scoring_file=out_path,
            n_variants=n_variants,
            n_gwas=n,
            gcst=source.gcst,
            trait_reported=header["trait_reported"],
            inputs=inputs,
            run=run,
            header=header,
        )

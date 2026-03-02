"""Reference panel utilities and PLINK2-free pgen operations.

Pure Python functions for working with PLINK2 binary format files
(.pgen/.pvar.zst/.psam) and the PGS Catalog reference panels:
  - Parsing .pvar.zst and .psam files (no PLINK2 binary needed)
  - Reading genotypes from .pgen via pgenlib
  - Variant matching between scoring files and .pvar
  - PRS scoring via pgenlib + polars + numpy
  - Downloading and extracting reference panel tarballs
  - Batch scoring across multiple PGS IDs with error/quality tracking
  - Aggregating per-superpopulation distribution statistics
  - Ancestry-matched percentile estimation

A legacy ``compute_reference_prs_plink2()`` function is retained for
cross-validation against the PLINK2 binary via ``prs reference compare``.
"""

from __future__ import annotations

import math
import re as _re
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import httpx
import polars as pl
from eliot import log_message, start_action
from pydantic import BaseModel, ConfigDict

from just_prs.scoring import resolve_cache_dir

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Reference panel registry
# ---------------------------------------------------------------------------

REFERENCE_PANELS: dict[str, dict[str, str]] = {
    "1000g": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_1000G_v1.tar.zst",
        "dir_name": "pgsc_1000G_v1",
        "description": "1000 Genomes Project (3,202 individuals, 5 superpopulations)",
    },
    "hgdp_1kg": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst",
        "dir_name": "pgsc_HGDP+1kGP_v1",
        "description": "HGDP + 1000 Genomes merged panel",
    },
}
DEFAULT_PANEL = "1000g"

REFERENCE_PANEL_URL = REFERENCE_PANELS[DEFAULT_PANEL]["url"]

SUPERPOPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")


# ---------------------------------------------------------------------------
# Batch scoring result models
# ---------------------------------------------------------------------------

class ScoringOutcome(BaseModel):
    """Per-PGS-ID outcome from batch reference scoring."""

    pgs_id: str
    status: str
    variants_total: int | None = None
    variants_matched: int | None = None
    match_rate: float | None = None
    n_samples: int | None = None
    score_mean: float | None = None
    score_std: float | None = None
    elapsed_sec: float | None = None
    error: str | None = None


class BatchScoringResult(BaseModel):
    """Result of batch reference PRS scoring across multiple PGS IDs.

    Does NOT hold raw per-sample scores — those are written to disk per PGS ID
    and discarded to avoid OOM when scoring thousands of IDs.  Only the small
    aggregated distribution summaries (~5 rows per PGS ID) are kept in memory.
    """

    panel: str
    distributions_df: pl.DataFrame
    outcomes: list[ScoringOutcome]
    quality_df: pl.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def reference_panel_dir(cache_dir: Path | None = None, panel: str = DEFAULT_PANEL) -> Path:
    """Return the local directory where the reference panel is (or will be) extracted."""
    base = cache_dir or resolve_cache_dir()
    panel_info = REFERENCE_PANELS.get(panel)
    if panel_info is None:
        raise ValueError(f"Unknown panel {panel!r}. Known panels: {list(REFERENCE_PANELS)}")
    return base / "reference_panel" / panel_info["dir_name"]


def download_reference_panel(
    cache_dir: Path | None = None,
    overwrite: bool = False,
    panel: str = DEFAULT_PANEL,
) -> Path:
    """Download and extract a reference panel tarball from the PGS Catalog FTP.

    Args:
        cache_dir: Root cache directory. Defaults to resolve_cache_dir().
        overwrite: Re-download even if already extracted.
        panel: Panel identifier (``1000g`` or ``hgdp_1kg``).

    Returns:
        Path to the extracted reference panel directory.
    """
    panel_info = REFERENCE_PANELS.get(panel)
    if panel_info is None:
        raise ValueError(f"Unknown panel {panel!r}. Known panels: {list(REFERENCE_PANELS)}")

    base = cache_dir or resolve_cache_dir()
    dest = reference_panel_dir(cache_dir, panel=panel)
    url = panel_info["url"]
    tarball_name = url.rsplit("/", 1)[-1]

    if dest.exists() and not overwrite:
        log_message(
            message_type="reference:panel_already_exists",
            path=str(dest),
            panel=panel,
        )
        return dest

    tarball = base / "reference_panel" / tarball_name
    tarball.parent.mkdir(parents=True, exist_ok=True)

    with start_action(
        action_type="reference:download_panel",
        url=url,
        tarball=str(tarball),
        panel=panel,
    ):
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with tarball.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        log_message(
                            message_type="reference:download_progress",
                            downloaded_mb=round(downloaded / 1e6, 1),
                            total_mb=round(total / 1e6, 1),
                        )

    with start_action(action_type="reference:extract_panel", tarball=str(tarball)):
        import zstandard as zstd

        dest.mkdir(parents=True, exist_ok=True)
        with tarball.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(dest)

    log_message(message_type="reference:panel_extracted", dest=str(dest), panel=panel)
    return dest


def parse_psam(psam_path: Path) -> pl.DataFrame:
    """Parse a PLINK2 .psam file and return a DataFrame with sample population labels.

    Expected columns: #IID (or IID), SuperPop (or SUP), Population (or POP).
    The '#' prefix on the first column header is stripped automatically.

    Returns:
        DataFrame with columns: iid, superpop, population
    """
    with start_action(action_type="reference:parse_psam", path=str(psam_path)):
        df = pl.read_csv(
            psam_path, separator="\t", comment_prefix="##", infer_schema=False,
        )
        df = df.rename({c: c.lstrip("#") for c in df.columns})

        col_map: dict[str, str] = {}
        for raw, canonical in [("IID", "iid"), ("SuperPop", "superpop"), ("Population", "population")]:
            match = next((c for c in df.columns if c.upper() == raw.upper()), None)
            if match:
                col_map[match] = canonical

        df = df.rename(col_map).select(["iid", "superpop", "population"])
        log_message(
            message_type="reference:psam_loaded",
            n_samples=df.height,
            superpops=df["superpop"].unique().sort().to_list(),
        )
        return df


def _parse_plink2_timing(log_text: str) -> dict[str, object]:
    """Extract timing, memory, and variant match stats from a PLINK2 log.

    Returns a dict with keys like ``ram_mb``, ``variants_loaded``,
    ``variants_matched``, ``variants_skipped``, ``samples_loaded``.
    Missing fields are omitted (caller can safely ``**`` the result).
    """
    info: dict[str, object] = {}

    m = _re.search(r"(\d+)\s+MiB RAM detected,\s*~(\d+)\s+available", log_text)
    if m:
        info["ram_total_mb"] = int(m.group(1))
        info["ram_available_mb"] = int(m.group(2))

    m = _re.search(r"reserving\s+(\d+)\s+MiB", log_text)
    if m:
        info["ram_reserved_mb"] = int(m.group(1))

    m = _re.search(r"(\d+)\s+samples?\b.*loaded from", log_text)
    if m:
        info["samples_loaded"] = int(m.group(1))

    m = _re.search(r"(\d+)\s+variants?\s+loaded from", log_text)
    if m:
        info["variants_loaded"] = int(m.group(1))

    m = _re.search(r"--score:\s+(\d+)\s+variants?\s+processed", log_text)
    if m:
        info["variants_matched"] = int(m.group(1))

    m = _re.search(r"--score:\s+(\d+)\s+valid predictor", log_text)
    if m:
        info["variants_matched"] = int(m.group(1))

    m = _re.search(r"(\d+)\s+entries?\s+.*were skipped due to missing variant IDs", log_text)
    if m:
        info["variants_skipped"] = int(m.group(1))

    m = _re.search(r"Using up to\s+(\d+)\s+compute threads", log_text)
    if m:
        info["threads_used"] = int(m.group(1))

    return info


class ReferencePanelError(RuntimeError):
    """Raised when reference panel files are missing or unusable."""


class ScoringFileError(RuntimeError):
    """Raised when a PGS scoring file cannot be parsed."""


# Keep old name for compute_reference_prs_plink2 internals
Plink2Error = ReferencePanelError


def compute_reference_prs_plink2(
    pgs_id: str,
    scoring_file: Path,
    ref_dir: Path,
    out_dir: Path,
    plink2_bin: Path,
    genome_build: str = "GRCh38",
    memory_mb: int = 16384,
    threads: int = 4,
) -> pl.DataFrame:
    """Run PLINK2 --score on the 1000G reference panel for a single PGS ID.

    The scoring file must be a tab-separated PGS Catalog harmonized file with
    columns: chr_name / hm_chr, chr_position / hm_pos, effect_allele, effect_weight.
    We convert it to the 3-column format PLINK2 expects: variant_id, allele, weight.

    Args:
        pgs_id: PGS Catalog Score ID (used for output file naming and labeling).
        scoring_file: Path to the downloaded harmonized scoring file (.txt.gz).
        ref_dir: Path to extracted reference panel directory (contains .pgen, .pvar, .psam).
        out_dir: Directory for PLINK2 output files.
        plink2_bin: Path to the plink2 binary.
        genome_build: Genome build to select the correct reference panel files (GRCh37 or GRCh38).
        memory_mb: PLINK2 --memory limit in MiB (default 16384 = 16 GB).
        threads: Number of PLINK2 compute threads (default 4).

    Returns:
        DataFrame with columns: iid, superpop, population, score, pgs_id

    Raises:
        ScoringFileError: If the scoring file cannot be parsed.
        Plink2Error: If PLINK2 fails or produces unusable output.
    """
    t_total_start = time.monotonic()
    with start_action(
        action_type="reference:compute_plink2_score",
        pgs_id=pgs_id,
        scoring_file=str(scoring_file),
        genome_build=genome_build,
        memory_mb=memory_mb,
        threads=threads,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)

        build_suffix = "hg38" if genome_build in ("GRCh38", "hg38") else "hg19"
        pgen_files = list(ref_dir.rglob(f"*{build_suffix}*.pgen")) + list(
            ref_dir.rglob("*.pgen")
        )
        if not pgen_files:
            raise Plink2Error(
                f"[{pgs_id}] No .pgen file found in {ref_dir} for build suffix '{build_suffix}'"
            )
        pgen = pgen_files[0]
        pfile_prefix = str(pgen).removesuffix(".pgen")

        t0 = time.monotonic()
        score_input = _prepare_plink2_score_input(scoring_file, pgs_id, out_dir)
        t_prepare = time.monotonic() - t0
        if score_input is None:
            raise ScoringFileError(
                f"[{pgs_id}] Could not parse scoring file {scoring_file} into PLINK2 input"
            )
        score_input_lines = sum(1 for _ in score_input.open()) - 1
        log_message(
            message_type="reference:phase_prepare_score_input",
            pgs_id=pgs_id,
            elapsed_sec=round(t_prepare, 3),
            score_input_rows=score_input_lines,
            score_input_bytes=score_input.stat().st_size,
        )

        out_prefix = out_dir / pgs_id
        cmd = [
            str(plink2_bin),
            "--pfile", pfile_prefix, "vzs",
            "--score", str(score_input), "1", "2", "3", "header", "no-mean-imputation",
            "--out", str(out_prefix),
            "--memory", str(memory_mb),
            "--threads", str(threads),
        ]
        log_message(message_type="reference:plink2_cmd", cmd=" ".join(cmd))

        t0 = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        t_plink2 = time.monotonic() - t0

        log_path = Path(str(out_prefix) + ".log")
        plink2_log = log_path.read_text() if log_path.exists() else ""

        plink2_stats = _parse_plink2_timing(plink2_log)
        log_message(
            message_type="reference:phase_plink2_execution",
            pgs_id=pgs_id,
            elapsed_sec=round(t_plink2, 3),
            returncode=result.returncode,
            **plink2_stats,
        )

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            log_tail = plink2_log[-2000:] if plink2_log else ""
            raise Plink2Error(
                f"[{pgs_id}] PLINK2 exited with code {result.returncode}\n"
                f"--- stderr ---\n{stderr_tail}\n"
                f"--- log tail ---\n{log_tail}"
            )

        sscore_file = Path(str(out_prefix) + ".sscore")
        if not sscore_file.exists():
            raise Plink2Error(
                f"[{pgs_id}] PLINK2 completed but no .sscore file at {sscore_file}\n"
                f"--- log tail ---\n{plink2_log[-2000:]}"
            )

        t0 = time.monotonic()
        scores_df = pl.read_csv(sscore_file, separator="\t")
        scores_df = scores_df.rename({c: c.lstrip("#") for c in scores_df.columns})
        iid_col = next((c for c in scores_df.columns if c.upper() in ("IID", "#IID")), None)
        score_col = next(
            (c for c in scores_df.columns if "SCORE" in c.upper() and "SUM" in c.upper()),
            None,
        )
        if score_col is None:
            score_col = next(
                (c for c in scores_df.columns if "SCORE" in c.upper() and "AVG" in c.upper()),
                None,
            )
        if iid_col is None or score_col is None:
            raise Plink2Error(
                f"[{pgs_id}] .sscore file missing expected columns. "
                f"Found: {scores_df.columns}, need IID and *SCORE*SUM* or *SCORE*AVG*"
            )

        scores_df = scores_df.rename({iid_col: "iid", score_col: "score"}).select(
            ["iid", "score"]
        )

        psam_files = list(ref_dir.rglob("*.psam"))
        if not psam_files:
            raise Plink2Error(f"[{pgs_id}] No .psam file found in {ref_dir}")
        psam_df = parse_psam(psam_files[0])

        joined = scores_df.join(psam_df, on="iid", how="inner").with_columns(
            pl.lit(pgs_id).alias("pgs_id")
        )
        t_parse = time.monotonic() - t0
        t_total = time.monotonic() - t_total_start

        log_message(
            message_type="reference:phase_parse_results",
            pgs_id=pgs_id,
            elapsed_sec=round(t_parse, 3),
            sscore_bytes=sscore_file.stat().st_size,
            n_samples=joined.height,
        )
        log_message(
            message_type="reference:plink2_score_done",
            pgs_id=pgs_id,
            n_samples=joined.height,
            total_elapsed_sec=round(t_total, 3),
            prepare_sec=round(t_prepare, 3),
            plink2_sec=round(t_plink2, 3),
            parse_sec=round(t_parse, 3),
            **plink2_stats,
        )
        return joined


def _resolve_other_allele_col(df: pl.DataFrame) -> str | None:
    """Find the column containing the non-effect allele, if available."""
    if "other_allele" in df.columns:
        return "other_allele"
    if "hm_inferOtherAllele" in df.columns:
        return "hm_inferOtherAllele"
    if "reference_allele" in df.columns:
        return "reference_allele"
    return None


def _prepare_plink2_score_input(
    scoring_file: Path,
    pgs_id: str,
    out_dir: Path,
) -> Path | None:
    """Convert a PGS Catalog harmonized scoring file to the 3-column PLINK2 --score format.

    PLINK2 --score expects: variant_id, effect_allele, effect_weight (tab-separated).

    The PGS Catalog 1000G reference panel uses variant IDs in ``CHROM:POS:REF:ALT``
    format (e.g. ``1:10397:C:A``).  Since the scoring file doesn't specify which
    allele is REF vs ALT, we emit **both orderings** for each variant (e.g.
    ``11:69516650:C:T`` and ``11:69516650:T:C``).  PLINK2 matches by string
    equality, so exactly one ordering will match the .pvar ID and the other will
    be counted as "missing" and silently skipped.

    When neither ``other_allele`` nor ``hm_inferOtherAllele`` is available, falls
    back to the 2-part ``CHROM:POS`` format (which will only work with reference
    panels that use that simpler scheme).

    Returns path to the prepared score input file, or None if parsing fails.
    """
    import gzip

    open_fn = gzip.open if str(scoring_file).endswith(".gz") else open
    with open_fn(scoring_file, "rt") as f:  # type: ignore[call-overload]
        header_lines = []
        data_lines = []
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
            else:
                data_lines.append(line)

    if not data_lines:
        log_message(
            message_type="reference:score_no_data_lines",
            pgs_id=pgs_id,
            scoring_file=str(scoring_file),
        )
        return None

    import io
    raw = "".join(data_lines)
    df = pl.read_csv(
        io.StringIO(raw),
        separator="\t",
        infer_schema_length=10000,
        schema_overrides={
            "chr_name": pl.Utf8,
            "chr_position": pl.Utf8,
            "hm_chr": pl.Utf8,
            "hm_pos": pl.Utf8,
        },
    )
    df = df.rename({c: c.lstrip("#") for c in df.columns})

    chr_col = "hm_chr" if "hm_chr" in df.columns else "chr_name"
    pos_col = "hm_pos" if "hm_pos" in df.columns else "chr_position"

    if chr_col not in df.columns or pos_col not in df.columns:
        log_message(
            message_type="reference:score_missing_coords",
            pgs_id=pgs_id,
            columns=df.columns,
        )
        return None

    chrom_expr = pl.col(chr_col).cast(pl.Utf8).str.replace("(?i)^chr", "")
    pos_expr = pl.col(pos_col).cast(pl.Utf8)

    other_col = _resolve_other_allele_col(df)

    if other_col is not None:
        df = df.filter(
            pl.col("effect_allele").is_not_null()
            & pl.col("effect_weight").is_not_null()
            & pl.col(chr_col).is_not_null()
            & pl.col(pos_col).is_not_null()
            & pl.col(other_col).is_not_null()
            & (pl.col(other_col).cast(pl.Utf8).str.len_chars() > 0)
        )

        other_clean = pl.col(other_col).cast(pl.Utf8).str.split("/").list.first()

        id_fwd = (
            chrom_expr + pl.lit(":") + pos_expr
            + pl.lit(":") + other_clean
            + pl.lit(":") + pl.col("effect_allele").cast(pl.Utf8)
        )
        id_rev = (
            chrom_expr + pl.lit(":") + pos_expr
            + pl.lit(":") + pl.col("effect_allele").cast(pl.Utf8)
            + pl.lit(":") + other_clean
        )

        df_fwd = df.with_columns(id_fwd.alias("variant_id")).select(
            ["variant_id", "effect_allele", "effect_weight"]
        )
        df_rev = df.with_columns(id_rev.alias("variant_id")).select(
            ["variant_id", "effect_allele", "effect_weight"]
        )
        out_df = pl.concat([df_fwd, df_rev])

        log_message(
            message_type="reference:score_input_4part_ids",
            pgs_id=pgs_id,
            n_variants=df.height,
            n_rows_with_both_orderings=out_df.height,
            other_allele_source=other_col,
        )
    else:
        df = df.filter(
            pl.col("effect_allele").is_not_null()
            & pl.col("effect_weight").is_not_null()
            & pl.col(chr_col).is_not_null()
            & pl.col(pos_col).is_not_null()
        ).with_columns(
            (chrom_expr + pl.lit(":") + pos_expr).alias("variant_id")
        )
        out_df = df.select(["variant_id", "effect_allele", "effect_weight"])

        log_message(
            message_type="reference:score_input_2part_ids_fallback",
            pgs_id=pgs_id,
            n_variants=out_df.height,
        )

    score_input = out_dir / f"{pgs_id}_score_input.txt"
    out_df.write_csv(score_input, separator="\t", include_header=True)
    return score_input


def _allele_offsets_cache_path(pvar_zst_path: Path) -> Path:
    """Return the path where cached allele-count diffs are stored."""
    return pvar_zst_path.parent / (pvar_zst_path.stem.replace(".pvar", "") + "_allele_cts.u8.zst")


def _build_allele_offsets_cache(pvar_zst_path: Path) -> Path:
    """Build a compact allele-count cache from a .pvar.zst file (one-time cost).

    Stores per-variant allele counts as uint8 diffs compressed with zstd.
    The resulting file is typically <1 MB (99.65% of variants are biallelic).
    These diffs can be reconstructed into the ``allele_idx_offsets`` array
    needed by ``pgenlib.PgenReader`` without re-parsing the .pvar.zst.

    Returns the path to the cache file.
    """
    import numpy as np
    import pgenlib
    import zstandard as zstd

    cache_path = _allele_offsets_cache_path(pvar_zst_path)
    if cache_path.exists():
        return cache_path

    log_message(
        message_type="reference:building_allele_offsets_cache",
        pvar_zst=str(pvar_zst_path),
        cache_path=str(cache_path),
    )

    with pgenlib.PvarReader(str(pvar_zst_path).encode("utf-8")) as pvar:
        offsets = pvar.get_allele_idx_offsets()
        variant_ct = pvar.get_variant_ct()

    offsets_arr = np.array(offsets, dtype=np.uint64)
    diffs = np.diff(offsets_arr).astype(np.uint8)

    cctx = zstd.ZstdCompressor(level=3)
    cache_path.write_bytes(cctx.compress(diffs.tobytes()))

    log_message(
        message_type="reference:allele_offsets_cache_built",
        variant_ct=variant_ct,
        cache_bytes=cache_path.stat().st_size,
    )
    return cache_path


def _load_allele_idx_offsets(cache_path: Path, variant_ct: int) -> object:
    """Load cached allele-count diffs and reconstruct allele_idx_offsets.

    Returns a numpy array of shape ``(variant_ct + 1,)`` with dtype ``uintp``
    suitable for passing to ``pgenlib.PgenReader``.
    """
    import numpy as np
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(cache_path.read_bytes())
    diffs = np.frombuffer(raw, dtype=np.uint8)

    offsets = np.empty(variant_ct + 1, dtype=np.uintp)
    offsets[0] = 0
    np.cumsum(diffs, out=offsets[1:])
    return offsets


def _pvar_parquet_cache_path(pvar_zst_path: Path) -> Path:
    """Return the parquet cache path for a .pvar.zst file."""
    return pvar_zst_path.parent / (pvar_zst_path.stem.replace(".pvar", "") + "_pvar.parquet")


def parse_pvar(pvar_zst_path: Path) -> pl.DataFrame:
    """Load a PLINK2 .pvar.zst variant table, using a parquet cache when available.

    On first call, decompresses and parses the .pvar.zst (~2.5 GB text, ~7s)
    and writes a parquet cache (~455 MB, subsequent reads ~0.5s).

    This is a pure Python alternative to ``plink2 --make-just-pvar``.

    Args:
        pvar_zst_path: Path to the .pvar.zst file.

    Returns:
        DataFrame with columns: variant_idx (u32), chrom (str),
        POS (i64), REF (str), ALT (str).  The variant_idx column corresponds
        to the 0-based row index in the .pgen file (needed by pgenlib).
    """
    parquet_cache = _pvar_parquet_cache_path(pvar_zst_path)

    if parquet_cache.exists():
        return pl.read_parquet(parquet_cache)

    import io

    import zstandard as zstd

    log_message(
        message_type="reference:building_pvar_parquet_cache",
        pvar_zst=str(pvar_zst_path),
        parquet_cache=str(parquet_cache),
    )

    dctx = zstd.ZstdDecompressor()
    with pvar_zst_path.open("rb") as fh:
        reader = dctx.stream_reader(fh)
        chunks: list[bytes] = []
        while True:
            chunk = reader.read(64 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)

    df = pl.read_csv(
        io.BytesIO(raw),
        separator="\t",
        comment_prefix="##",
        has_header=True,
        schema_overrides={
            "#CHROM": pl.Utf8,
            "POS": pl.Int64,
            "REF": pl.Utf8,
            "ALT": pl.Utf8,
        },
        columns=["#CHROM", "POS", "REF", "ALT"],
    )
    del raw, chunks

    result = df.with_row_index("variant_idx").with_columns(
        pl.col("#CHROM").str.replace("(?i)^chr", "").alias("chrom"),
    )

    result.write_parquet(parquet_cache, compression="zstd", compression_level=3)
    log_message(
        message_type="reference:pvar_parquet_cache_built",
        rows=result.height,
        cache_bytes=parquet_cache.stat().st_size,
    )

    return result


def match_scoring_to_pvar(
    pvar_df: pl.DataFrame,
    scoring_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join scoring file variants with .pvar variants by position and alleles.

    Matches on (chrom, pos) and then verifies that effect_allele and
    other_allele correspond to the REF/ALT pair in the .pvar.  Adds an
    ``effect_is_alt`` boolean column indicating whether the effect allele
    is the ALT allele (True) or the REF allele (False).

    This is the variant-matching step of PRS computation — a pure polars
    replacement for PLINK2's internal variant matching in ``--score``.

    Args:
        pvar_df: DataFrame from ``parse_pvar()`` (columns: variant_idx, chrom, POS, REF, ALT).
        scoring_df: Scoring DataFrame with columns: chr_name_norm, chr_pos_norm,
            effect_allele, effect_weight, and optionally other_allele.

    Returns:
        DataFrame with columns from both inputs plus ``effect_is_alt``.
    """
    joined = pvar_df.join(
        scoring_df,
        left_on=["chrom", "POS"],
        right_on=["chr_name_norm", "chr_pos_norm"],
        how="inner",
    )

    if "other_allele" in joined.columns:
        matched = joined.filter(
            (
                (pl.col("effect_allele") == pl.col("ALT"))
                & (pl.col("other_allele") == pl.col("REF"))
            )
            | (
                (pl.col("effect_allele") == pl.col("REF"))
                & (pl.col("other_allele") == pl.col("ALT"))
            )
        )
    else:
        matched = joined.filter(
            (pl.col("effect_allele") == pl.col("ALT"))
            | (pl.col("effect_allele") == pl.col("REF"))
        )

    return matched.with_columns(
        (pl.col("effect_allele") == pl.col("ALT")).alias("effect_is_alt")
    )


def read_pgen_genotypes(
    pgen_path: Path,
    pvar_zst_path: Path,
    variant_indices: "np.ndarray[Any, np.dtype[np.uint32]]",
    n_samples: int,
    pvar_variant_ct: int | None = None,
) -> "np.ndarray[Any, np.dtype[np.int8]]":
    """Read genotype data for specific variants from a PLINK2 .pgen file.

    Pure Python alternative to PLINK2's genotype extraction. Uses ``pgenlib``
    to read the binary .pgen format directly, with an automatic allele-offset
    cache for multiallelic variant support.

    Args:
        pgen_path: Path to the .pgen binary genotype file.
        pvar_zst_path: Path to the corresponding .pvar.zst file (used for
            allele offset caching — needed by pgenlib for multiallelic sites).
        variant_indices: 0-based variant row indices to extract (from
            ``parse_pvar()``'s ``variant_idx`` column). Must be uint32.
        n_samples: Number of samples in the .pgen file (from .psam).
        pvar_variant_ct: Total variant count from pvar. If provided, avoids
            re-reading the entire pvar just to get its row count.

    Returns:
        int8 numpy array of shape ``(len(variant_indices), n_samples)`` where
        values are ALT allele counts: 0 = hom-ref, 1 = het, 2 = hom-alt,
        -9 = missing.
    """
    import numpy as np
    import pgenlib

    if pvar_variant_ct is None:
        pvar_variant_ct = pl.scan_parquet(
            _pvar_parquet_cache_path(pvar_zst_path)
        ).select(pl.len()).collect().item()

    sort_order = np.argsort(variant_indices)
    sorted_indices = variant_indices[sort_order]

    offsets_cache = _allele_offsets_cache_path(pvar_zst_path)
    if not offsets_cache.exists():
        _build_allele_offsets_cache(pvar_zst_path)
    allele_offsets = _load_allele_idx_offsets(offsets_cache, variant_ct=pvar_variant_ct)

    with pgenlib.PgenReader(
        str(pgen_path).encode("utf-8"),
        raw_sample_ct=n_samples,
        variant_ct=pvar_variant_ct,
        allele_idx_offsets=allele_offsets,
    ) as greader:
        actual_samples = greader.get_raw_sample_ct()
        geno_buf = np.empty((len(sorted_indices), actual_samples), dtype=np.int8)
        greader.read_list(sorted_indices, geno_buf)

    unsort_order = np.argsort(sort_order)
    return geno_buf[unsort_order]


class _ResolvedRefPanel:
    """Pre-resolved reference panel paths and shared data.

    Resolves file paths once and caches the pvar variant count + psam.
    The 75M-row pvar is NOT loaded into memory — variant matching uses
    DuckDB to query the pvar parquet directly (~400 MB scan vs 6 GB polars).
    """

    __slots__ = (
        "pvar_variant_ct", "psam_df", "pgen_path",
        "pvar_zst_path", "pvar_parquet_path",
    )

    def __init__(self, ref_dir: Path, genome_build: str = "GRCh38") -> None:
        build_suffix = "hg38" if genome_build in ("GRCh38", "hg38") else "hg19"

        pvar_zst_files = list(ref_dir.rglob(f"*{build_suffix}*.pvar.zst")) + list(
            ref_dir.rglob("*.pvar.zst")
        )
        if not pvar_zst_files:
            raise ReferencePanelError(f"No .pvar.zst file in {ref_dir} for build '{build_suffix}'")
        self.pvar_zst_path = pvar_zst_files[0]

        self.pvar_parquet_path = _pvar_parquet_cache_path(self.pvar_zst_path)
        if not self.pvar_parquet_path.exists():
            parse_pvar(self.pvar_zst_path)

        pgen_files = list(ref_dir.rglob(f"*{build_suffix}*.pgen")) + list(
            ref_dir.rglob("*.pgen")
        )
        if not pgen_files:
            raise ReferencePanelError(f"No .pgen file in {ref_dir} for build '{build_suffix}'")
        self.pgen_path = pgen_files[0]

        psam_files = list(ref_dir.rglob("*.psam"))
        if not psam_files:
            raise ReferencePanelError(f"No .psam file found in {ref_dir}")
        self.psam_df = parse_psam(psam_files[0])

        import duckdb
        self.pvar_variant_ct = duckdb.sql(
            f"SELECT count(*) FROM '{self.pvar_parquet_path}'"
        ).fetchone()[0]  # type: ignore[index]

        log_message(
            message_type="reference:panel_resolved",
            pvar_variants=self.pvar_variant_ct,
            n_samples=self.psam_df.height,
            pvar_parquet=str(self.pvar_parquet_path),
        )

    def match_scoring(self, scoring_df: pl.DataFrame) -> pl.DataFrame:
        """Join scoring file variants with pvar using DuckDB (memory-efficient).

        Scans the 434 MB pvar parquet with DuckDB instead of loading 75M rows
        into polars (which spikes to 6+ GB). Returns a small DataFrame with
        only the matched variants, including ``variant_idx`` and ``effect_is_alt``.
        """
        import duckdb

        has_other = "other_allele" in scoring_df.columns
        con = duckdb.connect()
        con.register("scoring", scoring_df.to_arrow())
        pvar = str(self.pvar_parquet_path)

        if has_other:
            query = f"""
                SELECT
                    p.variant_idx,
                    p.chrom,
                    p."POS",
                    p."REF",
                    p."ALT",
                    s.effect_allele,
                    s.effect_weight,
                    s.other_allele,
                    CASE
                        WHEN s.effect_allele = p."ALT" AND s.other_allele = p."REF" THEN true
                        WHEN s.effect_allele = p."REF" AND s.other_allele = p."ALT" THEN false
                        ELSE NULL
                    END AS effect_is_alt
                FROM '{pvar}' p
                INNER JOIN scoring s
                    ON p.chrom = s.chr_name_norm AND p."POS" = s.chr_pos_norm
                WHERE
                    (s.effect_allele = p."ALT" AND s.other_allele = p."REF")
                    OR (s.effect_allele = p."REF" AND s.other_allele = p."ALT")
            """
        else:
            query = f"""
                SELECT
                    p.variant_idx,
                    p.chrom,
                    p."POS",
                    p."REF",
                    p."ALT",
                    s.effect_allele,
                    s.effect_weight,
                    CASE
                        WHEN s.effect_allele = p."ALT" THEN true
                        WHEN s.effect_allele = p."REF" THEN false
                        ELSE NULL
                    END AS effect_is_alt
                FROM '{pvar}' p
                INNER JOIN scoring s
                    ON p.chrom = s.chr_name_norm AND p."POS" = s.chr_pos_norm
                WHERE s.effect_allele = p."ALT" OR s.effect_allele = p."REF"
            """

        result = con.sql(query).pl()
        con.close()
        return result


def compute_reference_prs_polars(
    pgs_id: str,
    scoring_file: Path,
    ref_dir: Path,
    out_dir: Path,
    genome_build: str = "GRCh38",
    _panel: "_ResolvedRefPanel | None" = None,
) -> pl.DataFrame:
    """Score a PGS ID against the 1000G panel using pgenlib + polars (no PLINK2).

    Uses pgenlib to extract genotypes from the .pgen binary, computes
    dosage-weighted PRS per sample in numpy, and joins with .psam for
    population labels.

    When called from ``compute_reference_prs_batch``, a pre-loaded
    ``_ResolvedRefPanel`` is passed to avoid re-reading the 75M-row pvar
    (1.3 GB) on every PGS ID.

    Args:
        pgs_id: PGS Catalog Score ID.
        scoring_file: Path to the downloaded harmonized scoring file (.txt.gz).
        ref_dir: Path to the extracted reference panel directory.
        out_dir: Directory for intermediate files.
        genome_build: Genome build (GRCh37 or GRCh38).
        _panel: Pre-resolved panel data. If None, resolved from ref_dir
            (loads pvar into memory — fine for single calls, wasteful for batches).

    Returns:
        DataFrame with columns: iid, superpop, population, score, pgs_id
    """
    import numpy as np

    from just_prs.prs import _normalize_scoring_columns
    from just_prs.scoring import parse_scoring_file

    panel = _panel or _ResolvedRefPanel(ref_dir, genome_build)

    t_total_start = time.monotonic()
    with start_action(
        action_type="reference:compute_polars_score",
        pgs_id=pgs_id,
        scoring_file=str(scoring_file),
        genome_build=genome_build,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        scoring_lf = parse_scoring_file(scoring_file)
        scoring_norm = _normalize_scoring_columns(scoring_lf)
        scoring_df = scoring_norm.collect()
        variants_total = scoring_df.height
        t_scoring = time.monotonic() - t0
        log_message(
            message_type="reference:polars_phase_scoring",
            pgs_id=pgs_id,
            elapsed_sec=round(t_scoring, 3),
            variants_total=variants_total,
        )

        t0 = time.monotonic()
        matched = panel.match_scoring(scoring_df)
        del scoring_df
        variants_matched = matched.height
        t_pvar = time.monotonic() - t0
        log_message(
            message_type="reference:polars_phase_pvar",
            pgs_id=pgs_id,
            elapsed_sec=round(t_pvar, 3),
            variants_matched=variants_matched,
        )

        t0 = time.monotonic()
        variant_indices = matched["variant_idx"].cast(pl.UInt32).to_numpy()

        geno_ordered = read_pgen_genotypes(
            pgen_path=panel.pgen_path,
            pvar_zst_path=panel.pvar_zst_path,
            variant_indices=variant_indices,
            n_samples=panel.psam_df.height,
            pvar_variant_ct=panel.pvar_variant_ct,
        )
        n_samples = geno_ordered.shape[1]
        t_geno = time.monotonic() - t0
        log_message(
            message_type="reference:polars_phase_genotypes",
            pgs_id=pgs_id,
            elapsed_sec=round(t_geno, 3),
            n_samples=n_samples,
            n_variants_read=len(variant_indices),
        )

        t0 = time.monotonic()
        weights = matched["effect_weight"].to_numpy()
        is_alt = matched["effect_is_alt"].to_numpy()
        missing_mask = geno_ordered == -9
        geno_float = geno_ordered.astype(np.float64)
        del geno_ordered

        dosage = np.where(is_alt[:, np.newaxis], geno_float, 2.0 - geno_float)
        del geno_float
        dosage = np.where(missing_mask, 0.0, dosage)
        del missing_mask

        prs_sum = (dosage * weights[:, np.newaxis]).sum(axis=0)
        del dosage, weights, is_alt, matched
        allele_ct = 2 * variants_matched
        prs_avg = prs_sum / allele_ct
        t_compute = time.monotonic() - t0
        log_message(
            message_type="reference:polars_phase_compute",
            pgs_id=pgs_id,
            elapsed_sec=round(t_compute, 3),
            variants_matched=variants_matched,
            n_samples=n_samples,
        )

        t0 = time.monotonic()
        result = (
            pl.DataFrame({"iid": panel.psam_df["iid"].to_list(), "score": prs_avg.tolist()})
            .join(panel.psam_df, on="iid", how="inner")
            .with_columns(pl.lit(pgs_id).alias("pgs_id"))
        )
        del prs_avg
        t_join = time.monotonic() - t0
        t_total = time.monotonic() - t_total_start

        log_message(
            message_type="reference:polars_score_done",
            pgs_id=pgs_id,
            n_samples=result.height,
            variants_total=variants_total,
            variants_matched=variants_matched,
            total_elapsed_sec=round(t_total, 3),
            scoring_sec=round(t_scoring, 3),
            pvar_sec=round(t_pvar, 3),
            genotypes_sec=round(t_geno, 3),
            compute_sec=round(t_compute, 3),
            join_sec=round(t_join, 3),
        )
        return result


class _SinglePgsAgg(BaseModel):
    """Aggregated distribution + outcome for a single cached PGS ID."""
    df: pl.DataFrame
    outcome: ScoringOutcome
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _aggregate_single_pgs(parquet_path: Path, pgs_id: str) -> _SinglePgsAgg | None:
    """Read a cached per-PGS scores parquet, aggregate distributions, discard raw scores.

    Returns None if the parquet is empty or unreadable.
    """
    lf = pl.scan_parquet(parquet_path)
    schema = lf.collect_schema()
    if "score" not in schema.names():
        return None
    stats = lf.select(
        pl.col("score").mean().alias("mean"),
        pl.col("score").std().alias("std"),
        pl.len().alias("n"),
    ).collect()
    if stats["n"][0] == 0:
        return None
    df = lf.collect()
    dist = aggregate_distributions(df)
    del df
    return _SinglePgsAgg(
        df=dist,
        outcome=ScoringOutcome(
            pgs_id=pgs_id,
            status="ok",
            n_samples=int(stats["n"][0]),
            score_mean=float(stats["mean"][0]) if stats["mean"][0] is not None else None,
            score_std=float(stats["std"][0]) if stats["std"][0] is not None else None,
        ),
    )


def compute_reference_prs_batch(
    pgs_ids: list[str],
    ref_dir: Path,
    cache_dir: Path,
    genome_build: str = "GRCh38",
    panel: str = DEFAULT_PANEL,
    skip_existing: bool = True,
    match_rate_threshold: float = 0.1,
    output_subdir: str | None = None,
) -> BatchScoringResult:
    """Score multiple PGS IDs against a reference panel in a single process.

    Iterates over *pgs_ids*, downloading scoring files and calling
    ``compute_reference_prs_polars`` for each.  Per-sample scores are
    aggregated into distribution summaries (5 rows per PGS ID) immediately
    and then discarded to avoid OOM.  Failures are logged and recorded
    in the returned ``outcomes`` list — the loop never aborts.

    Writes both the quality report and the distributions parquet to
    *cache_dir/percentiles/* (or *cache_dir/percentiles/{output_subdir}/*).

    Args:
        pgs_ids: PGS Catalog Score IDs to score.
        ref_dir: Extracted reference panel directory (contains .pgen/.pvar/.psam).
        cache_dir: Root cache directory for scoring files and output.
        genome_build: Genome build (GRCh37 or GRCh38).
        panel: Panel identifier used for output file naming.
        skip_existing: Skip PGS IDs whose ``scores.parquet`` already exists.
        match_rate_threshold: Flag scores with match rate below this as ``low_match``.
        output_subdir: Optional subdirectory under ``percentiles/`` for output isolation
            (e.g. ``"test"`` to avoid overwriting full-panel results).

    Returns:
        ``BatchScoringResult`` with aggregated distributions, per-ID outcomes,
        and a quality DataFrame.  Raw per-sample scores are NOT held in
        memory — they are written to disk per PGS ID and discarded.
    """
    from just_prs.scoring import download_scoring_file

    scores_cache = cache_dir / "scores"
    scores_cache.mkdir(parents=True, exist_ok=True)

    resolved = _ResolvedRefPanel(ref_dir, genome_build)

    outcomes: list[ScoringOutcome] = []
    dist_parts: list[pl.DataFrame] = []

    with start_action(
        action_type="reference:batch_score",
        panel=panel,
        genome_build=genome_build,
        n_pgs_ids=len(pgs_ids),
    ):
        for i, pgs_id in enumerate(pgs_ids):
            out_dir = cache_dir / "reference_scores" / panel / pgs_id
            out_dir.mkdir(parents=True, exist_ok=True)
            result_parquet = out_dir / "scores.parquet"

            if skip_existing and result_parquet.exists():
                dist = _aggregate_single_pgs(result_parquet, pgs_id)
                if dist is not None:
                    dist_parts.append(dist.df)
                    outcomes.append(dist.outcome)
                    log_message(
                        message_type="reference:batch_skip_cached",
                        pgs_id=pgs_id,
                        progress=f"{i + 1}/{len(pgs_ids)}",
                    )
                    continue

            t0 = time.monotonic()
            try:
                scoring_file = download_scoring_file(
                    pgs_id=pgs_id,
                    output_dir=scores_cache,
                    genome_build=genome_build,
                )
                df = compute_reference_prs_polars(
                    pgs_id=pgs_id,
                    scoring_file=scoring_file,
                    ref_dir=ref_dir,
                    out_dir=out_dir,
                    genome_build=genome_build,
                    _panel=resolved,
                )
            except Exception as exc:
                elapsed = round(time.monotonic() - t0, 3)
                log_message(
                    message_type="reference:batch_score_failed",
                    pgs_id=pgs_id,
                    error=str(exc),
                    elapsed_sec=elapsed,
                    progress=f"{i + 1}/{len(pgs_ids)}",
                )
                outcomes.append(ScoringOutcome(
                    pgs_id=pgs_id,
                    status="failed",
                    elapsed_sec=elapsed,
                    error=str(exc),
                ))
                continue

            elapsed = round(time.monotonic() - t0, 3)
            df.write_parquet(result_parquet)

            n_samples = df.height
            mean_val = df["score"].mean()
            std_val = df["score"].std()

            status = "ok"
            if std_val is not None and std_val < 1e-10:
                status = "zero_variance"

            per_pgs_dist = aggregate_distributions(df)
            dist_parts.append(per_pgs_dist)
            del df

            outcomes.append(ScoringOutcome(
                pgs_id=pgs_id,
                status=status,
                n_samples=n_samples,
                score_mean=mean_val,
                score_std=std_val,
                elapsed_sec=elapsed,
            ))

            log_message(
                message_type="reference:batch_score_done",
                pgs_id=pgs_id,
                status=status,
                n_samples=n_samples,
                elapsed_sec=elapsed,
                progress=f"{i + 1}/{len(pgs_ids)}",
            )

        quality_rows = [o.model_dump() for o in outcomes]
        quality_df = pl.DataFrame(quality_rows) if quality_rows else pl.DataFrame(
            schema={
                "pgs_id": pl.Utf8, "status": pl.Utf8, "variants_total": pl.Int64,
                "variants_matched": pl.Int64, "match_rate": pl.Float64,
                "n_samples": pl.Int64, "score_mean": pl.Float64, "score_std": pl.Float64,
                "elapsed_sec": pl.Float64, "error": pl.Utf8,
            }
        )

        if dist_parts:
            distributions_df = pl.concat(dist_parts, how="diagonal_relaxed")
        else:
            distributions_df = pl.DataFrame(schema={
                "pgs_id": pl.Utf8, "superpopulation": pl.Utf8,
                "mean": pl.Float64, "std": pl.Float64, "n": pl.UInt32,
                "median": pl.Float64, "p5": pl.Float64, "p25": pl.Float64,
                "p75": pl.Float64, "p95": pl.Float64,
            })

        percentiles_dir = cache_dir / "percentiles"
        if output_subdir:
            percentiles_dir = percentiles_dir / output_subdir
        percentiles_dir.mkdir(parents=True, exist_ok=True)
        quality_df.write_parquet(percentiles_dir / f"{panel}_quality.parquet")
        distributions_df.write_parquet(percentiles_dir / f"{panel}_distributions.parquet")

        n_ok = sum(1 for o in outcomes if o.status == "ok")
        n_failed = sum(1 for o in outcomes if o.status == "failed")
        n_low = sum(1 for o in outcomes if o.status == "low_match")
        n_zero = sum(1 for o in outcomes if o.status == "zero_variance")
        log_message(
            message_type="reference:batch_complete",
            panel=panel,
            n_total=len(pgs_ids),
            n_ok=n_ok,
            n_failed=n_failed,
            n_low_match=n_low,
            n_zero_variance=n_zero,
            n_distributions=distributions_df.height,
        )

        return BatchScoringResult(
            panel=panel,
            distributions_df=distributions_df,
            outcomes=outcomes,
            quality_df=quality_df,
        )


def aggregate_distributions(scores_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-individual PRS scores into per-superpopulation distribution statistics.

    Args:
        scores_df: DataFrame with columns: pgs_id, iid, superpop, score

    Returns:
        DataFrame with columns: pgs_id, superpopulation, mean, std, n,
        median, p5, p25, p75, p95
    """
    with start_action(action_type="reference:aggregate_distributions"):
        agg = (
            scores_df.group_by(["pgs_id", "superpop"])
            .agg(
                pl.col("score").mean().alias("mean"),
                pl.col("score").std().alias("std"),
                pl.col("score").count().alias("n"),
                pl.col("score").median().alias("median"),
                pl.col("score").quantile(0.05).alias("p5"),
                pl.col("score").quantile(0.25).alias("p25"),
                pl.col("score").quantile(0.75).alias("p75"),
                pl.col("score").quantile(0.95).alias("p95"),
            )
            .rename({"superpop": "superpopulation"})
            .sort(["pgs_id", "superpopulation"])
        )
        log_message(
            message_type="reference:distributions_aggregated",
            n_rows=agg.height,
            pgs_ids=agg["pgs_id"].unique().len(),
        )
        return agg


def ancestry_percentile(
    prs_score: float,
    pgs_id: str,
    superpopulation: str,
    distributions_lf: pl.LazyFrame,
) -> float | None:
    """Estimate percentile relative to a 1000G ancestry reference group.

    Looks up the mean and std for (pgs_id, superpopulation) in the pre-computed
    reference distributions, then computes Phi((score - mean) / std) * 100.

    Args:
        prs_score: The computed PRS value.
        pgs_id: PGS Catalog Score ID.
        superpopulation: One of AFR, AMR, EAS, EUR, SAS.
        distributions_lf: LazyFrame from reference_distributions.parquet.

    Returns:
        Percentile (0-100) or None if (pgs_id, superpopulation) not found or std==0.
    """
    row_df = (
        distributions_lf.filter(
            (pl.col("pgs_id") == pgs_id)
            & (pl.col("superpopulation") == superpopulation.upper())
        )
        .select(["mean", "std"])
        .collect()
    )
    if row_df.height == 0:
        return None
    row = row_df.row(0, named=True)
    mean = float(row["mean"])
    std = float(row["std"] or 0.0)
    if std <= 0:
        return None
    z = (prs_score - mean) / std
    return round(_norm_cdf(z) * 100.0, 2)


def enrich_distributions(
    distributions_df: pl.DataFrame,
    metadata_dir: Path,
) -> pl.DataFrame:
    """Join distribution statistics with cleaned PGS Catalog metadata.

    Adds trait info (name, trait, EFO terms) and best-available performance
    metrics (AUROC, OR, C-index, ancestry) from the cleaned metadata parquets
    so that the published distributions parquet is self-contained.

    Args:
        distributions_df: Per-superpopulation distribution stats with pgs_id column.
        metadata_dir: Directory containing scores.parquet and best_performance.parquet.

    Returns:
        Enriched DataFrame with metadata columns joined on pgs_id.
    """
    scores_path = metadata_dir / "scores.parquet"
    best_perf_path = metadata_dir / "best_performance.parquet"

    if not scores_path.exists() or not best_perf_path.exists():
        log_message(
            message_type="reference:enrich_distributions_skipped",
            reason="cleaned metadata parquets not found",
            metadata_dir=str(metadata_dir),
        )
        return distributions_df

    scores_cols = ["pgs_id", "name", "trait_reported", "trait_efo", "trait_efo_id",
                   "genome_build", "n_variants", "pgp_id", "pmid"]
    scores_lf = pl.scan_parquet(scores_path)
    available = scores_lf.collect_schema().names()
    scores_cols = [c for c in scores_cols if c in available]
    scores_meta = scores_lf.select(scores_cols).collect()

    perf_cols_wanted = ["pgs_id", "auroc_estimate", "auroc_ci_lower", "auroc_ci_upper",
                        "or_estimate", "or_ci_lower", "or_ci_upper",
                        "hr_estimate", "hr_ci_lower", "hr_ci_upper",
                        "cindex_estimate", "cindex_ci_lower", "cindex_ci_upper",
                        "ancestry_broad", "n_individuals"]
    best_perf_lf = pl.scan_parquet(best_perf_path)
    available_perf = best_perf_lf.collect_schema().names()
    perf_cols = [c for c in perf_cols_wanted if c in available_perf]
    best_perf = best_perf_lf.select(perf_cols).collect()

    enriched = distributions_df.join(scores_meta, on="pgs_id", how="left")
    enriched = enriched.join(best_perf, on="pgs_id", how="left", suffix="_perf")

    log_message(
        message_type="reference:distributions_enriched",
        n_rows=enriched.height,
        n_columns=len(enriched.columns),
        added_columns=sorted(set(enriched.columns) - set(distributions_df.columns)),
    )
    return enriched

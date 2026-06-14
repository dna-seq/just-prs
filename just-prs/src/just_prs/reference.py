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

import gc
import math
import re as _re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Literal, TYPE_CHECKING

import httpx
import polars as pl
from eliot import log_message, start_action
from pydantic import BaseModel, ConfigDict

from just_prs.scoring import resolve_cache_dir

if TYPE_CHECKING:
    import numpy as np


def _require_pgenlib() -> None:
    """Raise ImportError with a helpful message if pgenlib is not installed."""
    try:
        import pgenlib as _pgenlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "pgenlib is required for reference panel operations but is not installed. "
            "On Linux/macOS install it with: pip install just-prs[reference] (or uv sync). "
            "On Windows pgenlib is intentionally excluded (no Windows wheels and its "
            "bundled C fails to compile with MSVC) — use WSL or Linux for reference-panel "
            "/ .pgen features. The UI and VCF-based PRS computation do not need pgenlib."
        ) from None


def _require_duckdb() -> None:
    """Raise ImportError with a helpful message if duckdb is not installed."""
    try:
        import duckdb as _duckdb  # noqa: F401
    except ImportError:
        raise ImportError(
            "duckdb is required for reference panel variant matching but is not installed. "
            "Install it with: pip install just-prs[reference]"
        ) from None


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
MatchMode = Literal["position", "id"]
_REFERENCE_PANEL_BUILDS = ("GRCh37", "GRCh38")


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


QUALITY_DF_SCHEMA: dict[str, pl.DataType] = {
    "pgs_id": pl.Utf8,
    "status": pl.Utf8,
    "variants_total": pl.Int64,
    "variants_matched": pl.Int64,
    "match_rate": pl.Float64,
    "n_samples": pl.Int64,
    "score_mean": pl.Float64,
    "score_std": pl.Float64,
    "elapsed_sec": pl.Float64,
    "error": pl.Utf8,
}

DISTRIBUTION_ISSUE_SCHEMA: dict[str, pl.DataType] = {
    "pgs_id": pl.Utf8,
    "superpopulation": pl.Utf8,
    "severity": pl.Utf8,
    "issue": pl.Utf8,
    "recommended_action": pl.Utf8,
    "mean": pl.Float64,
    "std": pl.Float64,
    "n": pl.Int64,
    "median": pl.Float64,
    "p5": pl.Float64,
    "p25": pl.Float64,
    "p75": pl.Float64,
    "p95": pl.Float64,
}


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
    distribution_issues_df: pl.DataFrame

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


def _build_name_tokens(genome_build: str) -> tuple[str, ...]:
    """Return filename tokens used by PGS reference panels for a genome build."""
    normalized = genome_build.upper()
    if normalized in {"GRCH38", "HG38"}:
        return ("GRCh38", "hg38")
    if normalized in {"GRCH37", "HG19"}:
        return ("GRCh37", "hg19")
    raise ReferencePanelError(f"Unsupported reference panel build: {genome_build!r}")


def _find_reference_panel_file(
    ref_dir: Path,
    genome_build: str,
    extension: str,
    allow_single_fallback: bool = True,
) -> Path:
    """Find a build-specific reference panel file without silently crossing builds."""
    tokens = tuple(token.lower() for token in _build_name_tokens(genome_build))
    all_files = sorted(path for path in ref_dir.rglob(f"*{extension}") if path.is_file())
    matches = [path for path in all_files if any(token in path.name.lower() for token in tokens)]
    if matches:
        return matches[0]
    if allow_single_fallback and len(all_files) == 1:
        return all_files[0]

    available = ", ".join(path.name for path in all_files[:10]) or "none"
    raise ReferencePanelError(
        f"No {extension} file in {ref_dir} for build {genome_build!r} "
        f"(expected tokens: {', '.join(_build_name_tokens(genome_build))}; available: {available})"
    )


def _reference_panel_complete(ref_dir: Path) -> bool:
    """Check whether an extracted panel has the files needed for supported builds."""
    if not ref_dir.exists():
        return False
    if not any(path.is_file() for path in ref_dir.rglob("*.psam")):
        return False
    for build in _REFERENCE_PANEL_BUILDS:
        try:
            _find_reference_panel_file(ref_dir, build, ".pgen", allow_single_fallback=False)
            _find_reference_panel_file(ref_dir, build, ".pvar.zst", allow_single_fallback=False)
        except ReferencePanelError:
            return False
    return True


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

    if dest.exists() and not overwrite and _reference_panel_complete(dest):
        log_message(
            message_type="reference:panel_already_exists",
            path=str(dest),
            panel=panel,
        )
        return dest
    if dest.exists():
        log_message(
            message_type="reference:removing_incomplete_panel",
            path=str(dest),
            panel=panel,
        )
        shutil.rmtree(dest)

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

    tmp_dest = dest.with_name(f"{dest.name}.extracting")
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest)

    with start_action(action_type="reference:extract_panel", tarball=str(tarball)):
        import zstandard as zstd

        tmp_dest.mkdir(parents=True, exist_ok=True)
        try:
            with tarball.open("rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    with tarfile.open(fileobj=reader, mode="r|") as tar:
                        tar.extractall(tmp_dest)
            if not _reference_panel_complete(tmp_dest):
                raise ReferencePanelError(
                    f"Extracted reference panel is incomplete: {tmp_dest}"
                )
            tmp_dest.replace(dest)
        except Exception:
            shutil.rmtree(tmp_dest, ignore_errors=True)
            tarball.unlink(missing_ok=True)
            raise

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
            "--score", str(score_input), "1", "2", "3", "header", "no-mean-imputation", "cols=scoresums",
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
    _require_pgenlib()
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
        POS (i64), ID (str), REF (str), ALT (str). The variant_idx column corresponds
        to the 0-based row index in the .pgen file (needed by pgenlib).
    """
    parquet_cache = _pvar_parquet_cache_path(pvar_zst_path)

    if parquet_cache.exists():
        cached = pl.read_parquet(parquet_cache)
        if "ID" in cached.columns:
            return cached
        log_message(
            message_type="reference:pvar_parquet_cache_missing_id_rebuild",
            pvar_zst=str(pvar_zst_path),
            parquet_cache=str(parquet_cache),
        )
        parquet_cache.unlink()

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
            "ID": pl.Utf8,
            "REF": pl.Utf8,
            "ALT": pl.Utf8,
        },
        columns=["#CHROM", "POS", "ID", "REF", "ALT"],
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
    match_mode: MatchMode = "position",
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
    if match_mode == "id":
        if "ID" not in pvar_df.columns:
            raise ReferencePanelError("pvar DataFrame is missing ID column required for id-based matching")
        scoring_ids = _prepare_id_match_scoring_df(scoring_df)
        return pvar_df.join(
            scoring_ids,
            left_on="ID",
            right_on="variant_id",
            how="inner",
        )

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


def _prepare_id_match_scoring_df(scoring_df: pl.DataFrame) -> pl.DataFrame:
    """Build PLINK-parity synthetic variant IDs from normalized scoring rows.

    The reference panel .pvar uses ``CHROM:POS:REF:ALT`` IDs. To mirror the PLINK2
    test path, emit both allele orderings for each scoring row so exactly one ID
    matches the pvar when ``other_allele`` is available.
    """
    required = {"chr_name_norm", "chr_pos_norm", "effect_allele", "effect_weight", "other_allele"}
    missing = sorted(required - set(scoring_df.columns))
    if missing:
        raise ReferencePanelError(
            "id-based matching requires normalized scoring columns "
            f"{sorted(required)}; missing {missing}"
        )

    chrom_expr = pl.col("chr_name_norm").cast(pl.Utf8).str.replace("(?i)^chr", "")
    pos_expr = pl.col("chr_pos_norm").cast(pl.Int64).cast(pl.Utf8)
    effect_expr = pl.col("effect_allele").cast(pl.Utf8)
    other_expr = pl.col("other_allele").cast(pl.Utf8).str.split("/").list.first()

    filtered = scoring_df.filter(
        pl.col("effect_allele").is_not_null()
        & pl.col("effect_weight").is_not_null()
        & pl.col("chr_name_norm").is_not_null()
        & pl.col("chr_pos_norm").is_not_null()
        & pl.col("other_allele").is_not_null()
        & (pl.col("other_allele").cast(pl.Utf8).str.len_chars() > 0)
    )

    id_fwd = chrom_expr + pl.lit(":") + pos_expr + pl.lit(":") + other_expr + pl.lit(":") + effect_expr
    id_rev = chrom_expr + pl.lit(":") + pos_expr + pl.lit(":") + effect_expr + pl.lit(":") + other_expr

    fwd = filtered.with_columns(
        id_fwd.alias("variant_id"),
        pl.lit(True).alias("effect_is_alt"),
    ).select(["variant_id", "effect_weight", "effect_is_alt"])
    rev = filtered.with_columns(
        id_rev.alias("variant_id"),
        pl.lit(False).alias("effect_is_alt"),
    ).select(["variant_id", "effect_weight", "effect_is_alt"])
    return pl.concat([fwd, rev])


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
    _require_pgenlib()
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
        self.pvar_zst_path = _find_reference_panel_file(ref_dir, genome_build, ".pvar.zst")

        self.pvar_parquet_path = _pvar_parquet_cache_path(self.pvar_zst_path)
        if not self.pvar_parquet_path.exists():
            parse_pvar(self.pvar_zst_path)
        else:
            pvar_schema = pl.read_parquet_schema(self.pvar_parquet_path)
            if "ID" not in pvar_schema:
                parse_pvar(self.pvar_zst_path)

        self.pgen_path = _find_reference_panel_file(ref_dir, genome_build, ".pgen")

        psam_files = list(ref_dir.rglob("*.psam"))
        if not psam_files:
            raise ReferencePanelError(f"No .psam file found in {ref_dir}")
        self.psam_df = parse_psam(psam_files[0])

        _require_duckdb()
        import duckdb
        con = duckdb.connect(config={"memory_limit": _resolve_duckdb_memory_limit()})
        self.pvar_variant_ct = con.sql(
            f"SELECT count(*) FROM '{self.pvar_parquet_path}'"
        ).fetchone()[0]  # type: ignore[index]
        con.close()

        log_message(
            message_type="reference:panel_resolved",
            pvar_variants=self.pvar_variant_ct,
            n_samples=self.psam_df.height,
            pvar_parquet=str(self.pvar_parquet_path),
        )

    def match_scoring(self, scoring_df: pl.DataFrame, match_mode: MatchMode = "position") -> pl.DataFrame:
        """Join scoring file variants with pvar using DuckDB (memory-efficient).

        Scans the 434 MB pvar parquet with DuckDB instead of loading 75M rows
        into polars (which spikes to 6+ GB). Returns a small DataFrame with
        only the matched variants, including ``variant_idx`` and ``effect_is_alt``.
        """
        _require_duckdb()
        import duckdb

        _MAX_ALLELE_LEN = 1000
        con = duckdb.connect(config={"memory_limit": _resolve_duckdb_memory_limit()})
        con.execute("SET arrow_large_buffer_size = true")
        pvar = str(self.pvar_parquet_path)

        allele_filter = pl.col("effect_allele").str.len_bytes() <= _MAX_ALLELE_LEN
        if "other_allele" in scoring_df.columns:
            allele_filter = allele_filter & (
                pl.col("other_allele").is_null()
                | (pl.col("other_allele").str.len_bytes() <= _MAX_ALLELE_LEN)
            )
        n_before = scoring_df.height
        scoring_df = scoring_df.filter(allele_filter)
        n_dropped = n_before - scoring_df.height
        if n_dropped > 0:
            log_message(
                message_type="reference:match_scoring_allele_filter",
                dropped=n_dropped,
                max_allele_len=_MAX_ALLELE_LEN,
                remaining=scoring_df.height,
            )

        from just_prs.prs import DOSAGE_WEIGHT_COLUMNS, is_dosage_weight_format

        dosage_weight = is_dosage_weight_format(scoring_df.columns)
        weight_cols_sql = (
            ", ".join(f's."{c}"' for c in DOSAGE_WEIGHT_COLUMNS)
            if dosage_weight
            else "s.effect_weight"
        )

        if match_mode == "id":
            scoring_ids = _prepare_id_match_scoring_df(scoring_df)
            con.register("scoring_ids", scoring_ids.to_arrow())
            query = f"""
                SELECT
                    p.variant_idx,
                    p.chrom,
                    p."POS",
                    p."ID",
                    p."REF",
                    p."ALT",
                    {weight_cols_sql},
                    s.effect_is_alt
                FROM '{pvar}' p
                INNER JOIN scoring_ids s
                    ON p."ID" = s.variant_id
            """
        else:
            has_other = "other_allele" in scoring_df.columns
            con.register("scoring", scoring_df.to_arrow())
            if has_other:
                query = f"""
                    SELECT
                        p.variant_idx,
                        p.chrom,
                        p."POS",
                        p."REF",
                        p."ALT",
                        s.effect_allele,
                        {weight_cols_sql},
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
                        {weight_cols_sql},
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

        try:
            result = con.sql(query).pl()
        finally:
            con.close()
        return result


_DEFAULT_MEMORY_SAFETY_PERCENT = 10
_DEFAULT_MEMORY_SAFETY_MIN_MB = 512
_BYTES_PER_VARIANT_SAMPLE = 13
_DEFAULT_DUCKDB_MEMORY_PERCENT = 75


def _resolve_duckdb_memory_limit() -> str:
    """Compute DuckDB per-connection memory limit.

    This is a safety guardrail, not a tight budget.  DuckDB's actual usage
    for scoring-vs-pvar INNER JOINs is well under 1 GB (hash table on the
    small scoring side, streaming scan of the pvar parquet).  The limit
    exists only to prevent a runaway query from consuming all system RAM.

    Resolution order:
      1. ``PRS_DUCKDB_MEMORY_LIMIT`` env var (e.g. ``"8GB"``) — used as-is.
      2. ``PRS_DUCKDB_MEMORY_PERCENT`` env var — percentage of total RAM.
      3. Default: 75 % of total RAM (on a 90 GB machine → ~67 GB).

    Returns a string suitable for DuckDB's ``memory_limit`` config key.
    """
    import os

    import psutil

    explicit = os.environ.get("PRS_DUCKDB_MEMORY_LIMIT", "").strip()
    if explicit:
        return explicit

    total_bytes = psutil.virtual_memory().total
    pct_str = os.environ.get("PRS_DUCKDB_MEMORY_PERCENT", "").strip()
    pct = int(pct_str) if pct_str else _DEFAULT_DUCKDB_MEMORY_PERCENT
    limit_bytes = int(total_bytes * pct / 100)
    limit_gb = max(limit_bytes / (1024 ** 3), 1.0)
    return f"{limit_gb:.1f}GB"


def _memory_safety_floor_bytes() -> int:
    """Return the memory safety floor in bytes, respecting env overrides."""
    import os
    import psutil

    total = psutil.virtual_memory().total
    pct_str = os.environ.get("PRS_MEMORY_SAFETY_PERCENT", "").strip()
    pct = int(pct_str) if pct_str else _DEFAULT_MEMORY_SAFETY_PERCENT
    min_str = os.environ.get("PRS_MEMORY_SAFETY_MIN_MB", "").strip()
    min_mb = int(min_str) if min_str else _DEFAULT_MEMORY_SAFETY_MIN_MB
    return max(int(total * pct / 100), min_mb * 1024 * 1024)


def _resolve_geno_chunk_size(n_samples: int, variants_remaining: int) -> int:
    """Determine genotype chunk size based on env var or available memory.

    Per-chunk peak memory accounts for all arrays alive simultaneously:
      - geno_buf:       chunk × n_samples × 1 byte   (int8, freed after cast)
      - geno_float:     chunk × n_samples × 4 bytes  (float32)
      - missing_mask:   chunk × n_samples × 1 byte   (bool)
      - multiply temp:  chunk × n_samples × 4 bytes  (float32, broadcast result)
    Peak ≈ chunk × n_samples × 10 bytes (after geno_buf is freed, before
    multiply temp is reduced by .sum()).  We use 13 bytes as a conservative
    estimate to cover numpy internal temporaries.

    When ``PRS_GENO_CHUNK_SIZE`` is set, that value is used directly.
    Otherwise, we auto-size to use up to 50% of currently free RAM
    (always keeping a safety floor of ``_MEMORY_SAFETY_PERCENT`` % of
    total RAM, minimum ``_MEMORY_SAFETY_MIN_MB``), clamped to
    [10_000, variants_remaining].

    This function is called **before each chunk** so the budget adapts
    to live memory pressure from other processes.
    """
    import os

    env_val = os.environ.get("PRS_GENO_CHUNK_SIZE", "").strip()
    if env_val:
        return min(max(int(env_val), 1000), variants_remaining)

    import psutil

    available_bytes = psutil.virtual_memory().available
    safety_floor = _memory_safety_floor_bytes()
    usable_bytes = max(available_bytes - safety_floor, 0)
    budget_bytes = usable_bytes // 2
    bytes_per_variant = n_samples * _BYTES_PER_VARIANT_SAMPLE
    auto_chunk = max(budget_bytes // bytes_per_variant, 10_000)
    return min(auto_chunk, variants_remaining)


def _check_memory_pressure(pgs_id: str) -> None:
    """Raise ``MemoryError`` if available RAM drops below the safety floor.

    The safety floor is ``PRS_MEMORY_SAFETY_PERCENT`` % of total RAM
    (minimum ``PRS_MEMORY_SAFETY_MIN_MB``).  Called before each chunk so
    the process exits cleanly instead of letting the OOM killer strike.
    """
    import psutil

    floor_bytes = _memory_safety_floor_bytes()
    floor_mb = floor_bytes / (1024 * 1024)
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    if available_mb < floor_mb:
        raise MemoryError(
            f"Available RAM ({available_mb:.0f} MB) dropped below safety floor "
            f"({floor_mb:.0f} MB) while scoring {pgs_id}. "
            f"Aborting to avoid OOM-killing other processes."
        )


def compute_reference_prs_polars(
    pgs_id: str,
    scoring_file: Path,
    ref_dir: Path,
    out_dir: Path,
    genome_build: str = "GRCh38",
    match_mode: MatchMode = "position",
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
        match_mode: Variant matching strategy. Default ``"position"`` matches on
            chromosome + position + allele logic. ``"id"`` is an opt-in PLINK-parity
            mode which matches on synthetic ``CHROM:POS:REF:ALT`` IDs.
        _panel: Pre-resolved panel data. If None, resolved from ref_dir
            (loads pvar into memory — fine for single calls, wasteful for batches).

    Returns:
        DataFrame with columns: iid, superpop, population, score, pgs_id
    """
    import os

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
        match_mode=match_mode,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        scoring_lf = parse_scoring_file(scoring_file)
        scoring_norm = _normalize_scoring_columns(scoring_lf)
        scoring_df = scoring_norm.collect()
        variants_total = scoring_df.height
        t_scoring = time.monotonic() - t0
        log_message(
            message_type="reference:phase_parse_scoring",
            pgs_id=pgs_id,
            elapsed_sec=round(t_scoring, 3),
            variants_total=variants_total,
        )

        t0 = time.monotonic()
        matched = panel.match_scoring(scoring_df, match_mode=match_mode)
        del scoring_df
        variants_matched = matched.height
        t_pvar = time.monotonic() - t0
        log_message(
            message_type="reference:phase_duckdb_match",
            pgs_id=pgs_id,
            elapsed_sec=round(t_pvar, 3),
            variants_matched=variants_matched,
        )

        from just_prs.prs import DOSAGE_WEIGHT_COLUMNS, is_dosage_weight_format

        t0 = time.monotonic()
        variant_indices = matched["variant_idx"].cast(pl.UInt32).to_numpy()
        n_samples = panel.psam_df.height
        t_geno = 0.0

        t0 = time.monotonic()
        if variants_matched == 0:
            raise ReferencePanelError(f"[{pgs_id}] No variants matched between scoring file and reference panel.")

        dosage_weight = is_dosage_weight_format(matched.columns)
        if dosage_weight:
            w0 = matched["dosage_0_weight"].to_numpy().astype(np.float32)
            w1 = matched["dosage_1_weight"].to_numpy().astype(np.float32)
            w2 = matched["dosage_2_weight"].to_numpy().astype(np.float32)
        else:
            weights = matched["effect_weight"].to_numpy().astype(np.float32)
        is_alt = matched["effect_is_alt"].to_numpy()
        del matched

        _require_pgenlib()
        import pgenlib

        sort_order = np.argsort(variant_indices)
        sorted_all = variant_indices[sort_order]
        is_alt_sorted = is_alt[sort_order]
        if dosage_weight:
            w0_sorted = w0[sort_order]
            w1_sorted = w1[sort_order]
            w2_sorted = w2[sort_order]
            del w0, w1, w2
        else:
            weights_sorted = weights[sort_order]
            del weights
        del is_alt, variant_indices

        offsets_cache = _allele_offsets_cache_path(panel.pvar_zst_path)
        if not offsets_cache.exists():
            _build_allele_offsets_cache(panel.pvar_zst_path)
        allele_offsets = _load_allele_idx_offsets(offsets_cache, variant_ct=panel.pvar_variant_ct)

        prs_sum = np.zeros(n_samples, dtype=np.float64)

        geno_t0 = time.monotonic()
        with pgenlib.PgenReader(
            str(panel.pgen_path).encode("utf-8"),
            raw_sample_ct=n_samples,
            variant_ct=panel.pvar_variant_ct,
            allele_idx_offsets=allele_offsets,
        ) as greader:
            actual_samples = greader.get_raw_sample_ct()
            start = 0
            while start < variants_matched:
                _check_memory_pressure(pgs_id)
                chunk_size = _resolve_geno_chunk_size(n_samples, variants_matched - start)
                end = min(start + chunk_size, variants_matched)
                batch_indices = sorted_all[start:end]
                batch_is_alt = is_alt_sorted[start:end]

                geno_buf = np.empty((end - start, actual_samples), dtype=np.int8)
                greader.read_list(batch_indices, geno_buf)

                missing_mask = geno_buf == -9

                if dosage_weight:
                    batch_w0 = w0_sorted[start:end]
                    batch_w1 = w1_sorted[start:end]
                    batch_w2 = w2_sorted[start:end]

                    geno_int = geno_buf.copy()
                    ref_effect_mask = np.logical_not(batch_is_alt)
                    if ref_effect_mask.any():
                        geno_int[ref_effect_mask, :] = 2 - geno_int[ref_effect_mask, :]
                    geno_int[missing_mask] = -9

                    contrib = np.where(
                        geno_int == 0,
                        batch_w0[:, np.newaxis],
                        np.where(
                            geno_int == 1,
                            batch_w1[:, np.newaxis],
                            np.where(
                                geno_int == 2,
                                batch_w2[:, np.newaxis],
                                np.float32(0.0),
                            ),
                        ),
                    )
                    prs_sum += contrib.sum(axis=0)
                    del geno_int, contrib, batch_w0, batch_w1, batch_w2
                else:
                    batch_weights = weights_sorted[start:end]
                    geno_float = geno_buf.astype(np.float32)

                    ref_effect_mask = np.logical_not(batch_is_alt)
                    if ref_effect_mask.any():
                        geno_float[ref_effect_mask, :] = np.float32(2.0) - geno_float[ref_effect_mask, :]
                    geno_float[missing_mask] = 0.0

                    prs_sum += (geno_float * batch_weights[:, np.newaxis]).sum(axis=0)
                    del geno_float, batch_weights

                del geno_buf, missing_mask, batch_indices, batch_is_alt
                start = end

        t_geno = time.monotonic() - geno_t0

        if dosage_weight:
            del sorted_all, is_alt_sorted, w0_sorted, w1_sorted, w2_sorted, allele_offsets
        else:
            del sorted_all, weights_sorted, is_alt_sorted, allele_offsets
        prs_scores = prs_sum
        t_compute = time.monotonic() - t0
        log_message(
            message_type="reference:phase_pgen_read",
            pgs_id=pgs_id,
            elapsed_sec=round(t_geno, 3),
            n_samples=n_samples,
            n_variants_read=variants_matched,
            genotype_batch_size=chunk_size,
        )
        log_message(
            message_type="reference:phase_numpy_compute",
            pgs_id=pgs_id,
            elapsed_sec=round(t_compute, 3),
            variants_matched=variants_matched,
            n_samples=n_samples,
        )

        t0 = time.monotonic()
        result = (
            pl.DataFrame({"iid": panel.psam_df["iid"].to_list(), "score": prs_scores.tolist()})
            .join(panel.psam_df, on="iid", how="inner")
            .with_columns(
                pl.lit(pgs_id).alias("pgs_id"),
                pl.lit(variants_total).alias("variants_total"),
                pl.lit(variants_matched).alias("variants_matched"),
                pl.lit(variants_matched / variants_total if variants_total > 0 else None).alias("match_rate"),
            )
        )
        del prs_scores
        t_join = time.monotonic() - t0
        t_total = time.monotonic() - t_total_start

        log_message(
            message_type="reference:score_done",
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


class _CorruptParquet(Exception):
    """Raised when a cached parquet file is found to be corrupted."""


def _aggregate_single_pgs(parquet_path: Path, pgs_id: str) -> _SinglePgsAgg | None:
    """Read a cached per-PGS scores parquet, aggregate distributions, discard raw scores.

    Returns None if the parquet is empty or unreadable.
    Raises _CorruptParquet if the file exists but cannot be parsed at any stage
    (truncated write, invalid thrift header, corrupt row-group data, etc.).
    collect_schema() only reads the footer metadata — actual data corruption is
    only detected when collect() reads the row groups.  The entire read sequence
    is therefore wrapped so any polars read error triggers delete-and-recompute.
    The caller should delete the file and fall through to recompute.
    """
    try:
        lf = pl.scan_parquet(parquet_path)
        schema = lf.collect_schema()
        if "score" not in schema.names():
            return None
        # Run all aggregations lazily — group_by + quantiles are fully supported
        # in the lazy engine so we never materialise the full per-sample frame.
        agg_lf = (
            lf.group_by(["pgs_id", "superpop"])
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
        # Also need overall n / mean / std and, for current score parquets,
        # reference match metadata for ScoringOutcome. Older caches won't have
        # these columns, so keep nulls and let the audit flag them.
        stat_exprs: list[pl.Expr] = [
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.len().alias("n"),
        ]
        schema_names = set(schema.names())
        if {"variants_total", "variants_matched", "match_rate"}.issubset(schema_names):
            stat_exprs.extend([
                pl.col("variants_total").first().alias("variants_total"),
                pl.col("variants_matched").first().alias("variants_matched"),
                pl.col("match_rate").first().alias("match_rate"),
            ])
        else:
            stat_exprs.extend([
                pl.lit(None, dtype=pl.Int64).alias("variants_total"),
                pl.lit(None, dtype=pl.Int64).alias("variants_matched"),
                pl.lit(None, dtype=pl.Float64).alias("match_rate"),
            ])
        stats_lf = lf.select(stat_exprs)
        dist, stats = pl.collect_all([agg_lf, stats_lf])
    except _CorruptParquet:
        raise
    except Exception as exc:
        raise _CorruptParquet(str(exc)) from exc
    if stats["n"][0] == 0:
        return None
    score_std = float(stats["std"][0]) if stats["std"][0] is not None else None
    status = "zero_variance" if score_std is not None and score_std < 1e-10 else "ok"
    return _SinglePgsAgg(
        df=dist,
        outcome=ScoringOutcome(
            pgs_id=pgs_id,
            status=status,
            variants_total=int(stats["variants_total"][0]) if stats["variants_total"][0] is not None else None,
            variants_matched=int(stats["variants_matched"][0]) if stats["variants_matched"][0] is not None else None,
            match_rate=float(stats["match_rate"][0]) if stats["match_rate"][0] is not None else None,
            n_samples=int(stats["n"][0]),
            score_mean=float(stats["mean"][0]) if stats["mean"][0] is not None else None,
            score_std=score_std,
        ),
    )


def _write_match_metadata_to_score_parquet(
    parquet_path: Path,
    variants_total: int,
    variants_matched: int,
    match_rate: float,
) -> None:
    """Persist reference match metadata into an existing per-PGS scores parquet."""
    tmp_path = parquet_path.with_name(f"{parquet_path.stem}.match-metadata.tmp{parquet_path.suffix}")
    (
        pl.scan_parquet(parquet_path)
        .with_columns(
            pl.lit(variants_total).alias("variants_total"),
            pl.lit(variants_matched).alias("variants_matched"),
            pl.lit(match_rate).alias("match_rate"),
        )
        .sink_parquet(tmp_path)
    )
    tmp_path.replace(parquet_path)


def _compute_reference_match_metadata(
    pgs_id: str,
    scores_cache: Path,
    genome_build: str,
    resolved: _ResolvedRefPanel,
) -> tuple[int, int, float]:
    """Compute scoring-file/reference-panel variant match counts without reading pgen genotypes."""
    from just_prs.prs import _normalize_scoring_columns
    from just_prs.scoring import download_scoring_file, parse_scoring_file, scoring_parquet_path

    parquet_path = scoring_parquet_path(pgs_id, scores_cache, genome_build)
    if parquet_path.exists():
        scoring_file = parquet_path
    else:
        scoring_file = download_scoring_file(
            pgs_id=pgs_id,
            output_dir=scores_cache,
            genome_build=genome_build,
        )
    scoring_df = _normalize_scoring_columns(parse_scoring_file(scoring_file)).collect()
    variants_total = scoring_df.height
    variants_matched = resolved.match_scoring(scoring_df).height
    match_rate = variants_matched / variants_total if variants_total > 0 else 0.0
    return variants_total, variants_matched, match_rate


def backfill_reference_quality_metadata(
    pgs_ids: list[str],
    ref_dir: Path,
    cache_dir: Path,
    genome_build: str = "GRCh38",
    panel: str = DEFAULT_PANEL,
    match_rate_threshold: float = 0.1,
    output_subdir: str | None = None,
    rewrite_score_parquets: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BatchScoringResult:
    """Rebuild reference quality/audit parquets from cached scores without pgen scoring.

    This repairs quality reports produced by older buggy batch runs where
    per-PGS score parquets existed but `variants_total`, `variants_matched`,
    and `match_rate` were missing from `{panel}_quality.parquet`.
    """
    scores_cache = cache_dir / "scores"
    scores_cache.mkdir(parents=True, exist_ok=True)
    scores_dir = cache_dir / "reference_scores" / panel
    percentiles_dir = cache_dir / "percentiles"
    if output_subdir:
        percentiles_dir = percentiles_dir / output_subdir
    percentiles_dir.mkdir(parents=True, exist_ok=True)

    t_resolve_start = time.monotonic()
    resolved = _ResolvedRefPanel(ref_dir, genome_build)
    t_resolve = time.monotonic() - t_resolve_start
    if progress_callback is not None:
        progress_callback({
            "processed": 0,
            "total": len(pgs_ids),
            "ok": 0,
            "failed": 0,
            "problematic": 0,
            "cached": 0,
            "last_pgs_id": "panel_init",
            "last_status": "panel_resolved",
            "elapsed_sec": t_resolve,
            "is_complete": False,
        })

    outcomes: list[ScoringOutcome] = []
    dist_parts: list[pl.DataFrame] = []
    n_ok = 0
    n_failed = 0
    n_problematic = 0
    started_at = time.monotonic()

    for i, pgs_id in enumerate(pgs_ids):
        result_parquet = scores_dir / pgs_id / "scores.parquet"
        if not result_parquet.exists():
            outcomes.append(ScoringOutcome(
                pgs_id=pgs_id,
                status="failed",
                error=f"cached score parquet not found: {result_parquet}",
            ))
            n_failed += 1
            continue

        try:
            aggregated = _aggregate_single_pgs(result_parquet, pgs_id)
            if aggregated is None:
                outcomes.append(ScoringOutcome(
                    pgs_id=pgs_id,
                    status="failed",
                    error=f"cached score parquet is empty or missing score column: {result_parquet}",
                ))
                n_failed += 1
                continue

            outcome = aggregated.outcome
            if outcome.variants_total is None or outcome.variants_matched is None or outcome.match_rate is None:
                variants_total, variants_matched, match_rate = _compute_reference_match_metadata(
                    pgs_id,
                    scores_cache,
                    genome_build,
                    resolved,
                )
                outcome.variants_total = variants_total
                outcome.variants_matched = variants_matched
                outcome.match_rate = match_rate
                if rewrite_score_parquets:
                    _write_match_metadata_to_score_parquet(
                        result_parquet,
                        variants_total,
                        variants_matched,
                        match_rate,
                    )

            if outcome.status == "ok" and outcome.match_rate is not None and outcome.match_rate < match_rate_threshold:
                outcome.status = "low_match"
            if outcome.status in {"low_match", "zero_variance"}:
                n_problematic += 1

            dist_parts.append(aggregated.df)
            outcomes.append(outcome)
            n_ok += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            outcomes.append(ScoringOutcome(
                pgs_id=pgs_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ))
            n_failed += 1

        if progress_callback is not None:
            progress_callback({
                "processed": i + 1,
                "total": len(pgs_ids),
                "ok": n_ok,
                "failed": n_failed,
                "problematic": n_problematic,
                "cached": n_ok,
                "last_pgs_id": pgs_id,
                "last_status": outcomes[-1].status if outcomes else "unknown",
                "elapsed_sec": time.monotonic() - started_at,
                "is_complete": i + 1 >= len(pgs_ids),
            })

    quality_rows = [o.model_dump() for o in outcomes]
    quality_df = pl.DataFrame(quality_rows, schema=QUALITY_DF_SCHEMA) if quality_rows else pl.DataFrame(
        schema=QUALITY_DF_SCHEMA
    )
    distributions_df = (
        pl.concat(dist_parts, how="diagonal_relaxed")
        if dist_parts
        else pl.DataFrame(schema={
            "pgs_id": pl.Utf8, "superpopulation": pl.Utf8,
            "mean": pl.Float64, "std": pl.Float64, "n": pl.UInt32,
            "median": pl.Float64, "p5": pl.Float64, "p25": pl.Float64,
            "p75": pl.Float64, "p95": pl.Float64,
        })
    )
    distribution_issues_df = reference_distribution_audit_issues(
        distributions_df,
        quality_df,
        min_match_rate=match_rate_threshold,
    )
    quality_df.write_parquet(percentiles_dir / f"{panel}_quality.parquet")
    distribution_issues_df.write_parquet(percentiles_dir / f"{panel}_distribution_quality_issues.parquet")
    distributions_df.write_parquet(percentiles_dir / f"{panel}_distributions.parquet")

    return BatchScoringResult(
        panel=panel,
        distributions_df=distributions_df,
        outcomes=outcomes,
        quality_df=quality_df,
        distribution_issues_df=distribution_issues_df,
    )


def distribution_quality_issues(distributions_df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per distribution-level quality issue.

    The per-PGS quality report tracks scoring outcomes, but some failures only
    appear after aggregation, e.g. one ancestry group has zero variance or a
    stale cached distribution contains non-finite values.  This report is meant
    for manual triage and downstream exclusion decisions.
    """
    if distributions_df.height == 0:
        return pl.DataFrame(schema=DISTRIBUTION_ISSUE_SCHEMA)

    issues: list[dict[str, object]] = []
    numeric_cols = ("mean", "std", "n", "median", "p5", "p25", "p75", "p95")
    required_cols = {"pgs_id", "superpopulation", *numeric_cols}
    missing_cols = required_cols - set(distributions_df.columns)
    if missing_cols:
        return pl.DataFrame(
            [{
                "pgs_id": "",
                "superpopulation": "",
                "severity": "ERROR",
                "issue": "missing_distribution_columns",
                "recommended_action": f"fix_distribution_schema:{','.join(sorted(missing_cols))}",
                "mean": None,
                "std": None,
                "n": None,
                "median": None,
                "p5": None,
                "p25": None,
                "p75": None,
                "p95": None,
            }],
            schema=DISTRIBUTION_ISSUE_SCHEMA,
        )

    for row in distributions_df.iter_rows(named=True):
        pgs_id = str(row["pgs_id"])
        superpopulation = str(row["superpopulation"])
        stats = {col: row[col] for col in numeric_cols}

        def add_issue(severity: str, issue: str, recommended_action: str) -> None:
            issues.append({
                "pgs_id": pgs_id,
                "superpopulation": superpopulation,
                "severity": severity,
                "issue": issue,
                "recommended_action": recommended_action,
                **stats,
            })

        mean = row["mean"]
        std = row["std"]
        if mean is None:
            add_issue("ERROR", "mean_null", "exclude_pgs_until_recomputed")
        elif not math.isfinite(float(mean)):
            add_issue("ERROR", "mean_nonfinite", "exclude_pgs_until_recomputed")

        if std is None:
            add_issue("ERROR", "std_null", "exclude_pgs_until_recomputed")
            continue

        std_float = float(std)
        if not math.isfinite(std_float):
            add_issue("ERROR", "std_nonfinite", "exclude_pgs_until_recomputed")
        elif std_float == 0.0:
            add_issue("WARN", "std_zero", "review_or_exclude_superpopulation_percentile")

        median = row["median"]
        p5 = row["p5"]
        p95 = row["p95"]
        if (
            mean is not None
            and std is not None
            and median is not None
            and p5 is not None
            and p95 is not None
            and all(math.isfinite(float(v)) for v in (mean, std, median, p5, p95))
        ):
            robust_span = abs(float(p95) - float(p5))
            tolerance = max(robust_span * 100.0, 1e-6)
            absolute_scale = max(abs(float(mean)), std_float)
            if absolute_scale > 1e4 and (abs(float(mean) - float(median)) > tolerance or std_float > tolerance):
                add_issue("ERROR", "robust_outlier_suspected", "exclude_pgs_until_recomputed")

    if not issues:
        return pl.DataFrame(schema=DISTRIBUTION_ISSUE_SCHEMA)
    return pl.DataFrame(issues, schema=DISTRIBUTION_ISSUE_SCHEMA)


def reference_distribution_audit_issues(
    distributions_df: pl.DataFrame,
    quality_df: pl.DataFrame | None = None,
    min_match_rate: float = 0.50,
) -> pl.DataFrame:
    """Return distribution issues plus quality-report audit findings.

    ``distribution_quality_issues()`` catches mathematically invalid percentile
    rows. This helper adds provenance checks from the per-PGS quality report, so
    finite-looking distributions are still flagged when we cannot audit how many
    variants the reference panel actually matched, or when the reference scoring
    run itself reported low match / failed / stale metadata.
    """
    base_issues = distribution_quality_issues(distributions_df)
    if distributions_df.height == 0:
        return base_issues

    issues = base_issues.to_dicts() if base_issues.height > 0 else []
    numeric_cols = ("mean", "std", "n", "median", "p5", "p25", "p75", "p95")
    if not {"pgs_id", "superpopulation", *numeric_cols}.issubset(set(distributions_df.columns)):
        return base_issues

    def issue_from_row(
        row: dict[str, object],
        severity: str,
        issue: str,
        recommended_action: str,
    ) -> dict[str, object]:
        return {
            "pgs_id": str(row["pgs_id"]),
            "superpopulation": str(row["superpopulation"]),
            "severity": severity,
            "issue": issue,
            "recommended_action": recommended_action,
            **{col: row[col] for col in numeric_cols},
        }

    if quality_df is None:
        issues.append({
            "pgs_id": "",
            "superpopulation": "",
            "severity": "WARN",
            "issue": "quality_report_missing",
            "recommended_action": "download_or_recompute_quality_report_before_publication",
            **{col: None for col in numeric_cols},
        })
        return pl.DataFrame(issues, schema=DISTRIBUTION_ISSUE_SCHEMA)

    required_quality_cols = {
        "pgs_id", "status", "variants_total", "variants_matched",
        "match_rate", "n_samples", "score_mean", "score_std",
    }
    missing_quality_cols = required_quality_cols - set(quality_df.columns)
    if missing_quality_cols:
        issues.append({
            "pgs_id": "",
            "superpopulation": "",
            "severity": "ERROR",
            "issue": "missing_quality_columns",
            "recommended_action": f"fix_quality_schema:{','.join(sorted(missing_quality_cols))}",
            **{col: None for col in numeric_cols},
        })
        return pl.DataFrame(issues, schema=DISTRIBUTION_ISSUE_SCHEMA)

    quality_by_id: dict[str, dict[str, object]] = {
        str(row["pgs_id"]): row for row in quality_df.iter_rows(named=True)
    }

    for pgs_id, pgs_rows in distributions_df.group_by("pgs_id", maintain_order=True):
        pgs_id_str = str(pgs_id[0] if isinstance(pgs_id, tuple) else pgs_id)
        q = quality_by_id.get(pgs_id_str)
        row_dicts = list(pgs_rows.iter_rows(named=True))
        if q is None:
            for row in row_dicts:
                issues.append(issue_from_row(
                    row,
                    "WARN",
                    "quality_row_missing",
                    "recompute_reference_scores_or_pull_quality_sidecar",
                ))
            continue

        status = str(q.get("status") or "")
        variants_total = q.get("variants_total")
        variants_matched = q.get("variants_matched")
        match_rate = q.get("match_rate")
        n_samples = q.get("n_samples")
        score_mean = q.get("score_mean")
        score_std = q.get("score_std")

        q_issues: list[tuple[str, str, str]] = []
        if status not in {"ok", "cached"}:
            severity = "ERROR" if status == "failed" else "WARN"
            q_issues.append((severity, f"quality_status_{status or 'unknown'}", "review_reference_scoring_outcome"))
        if variants_total is None or variants_matched is None or match_rate is None:
            q_issues.append(("WARN", "quality_match_metadata_missing", "recompute_reference_scores_with_match_counts"))
        elif float(match_rate) < min_match_rate:
            q_issues.append(("ERROR", "quality_low_match_rate", "exclude_pgs_until_reference_scoring_improves"))
        if n_samples is None:
            q_issues.append(("WARN", "quality_sample_count_missing", "recompute_reference_scores_with_sample_count"))
        elif int(n_samples) != int(pgs_rows["n"].sum()):
            q_issues.append(("ERROR", "quality_sample_count_mismatch", "recompute_distributions_from_raw_scores"))

        if score_mean is not None and score_std is not None:
            total_n = int(pgs_rows["n"].sum())
            if total_n > 0:
                weighted_mean = float((pgs_rows["mean"] * pgs_rows["n"]).sum() / total_n)
                pooled_second = float((((pgs_rows["std"] ** 2) + (pgs_rows["mean"] ** 2)) * pgs_rows["n"]).sum() / total_n)
                pooled_var = max(pooled_second - weighted_mean**2, 0.0)
                pooled_std = math.sqrt(pooled_var)
                mean_tol = max(abs(float(score_mean)) * 0.01, 1e-6)
                std_tol = max(abs(float(score_std)) * 0.01, 1e-6)
                if abs(weighted_mean - float(score_mean)) > mean_tol:
                    q_issues.append(("ERROR", "quality_score_mean_mismatch", "reaggregate_distributions_from_raw_scores"))
                if abs(pooled_std - float(score_std)) > std_tol:
                    q_issues.append(("ERROR", "quality_score_std_mismatch", "reaggregate_distributions_from_raw_scores"))

        for severity, issue, action in q_issues:
            for row in row_dicts:
                issues.append(issue_from_row(row, severity, issue, action))

    if not issues:
        return pl.DataFrame(schema=DISTRIBUTION_ISSUE_SCHEMA)
    return pl.DataFrame(issues, schema=DISTRIBUTION_ISSUE_SCHEMA)


def compute_reference_prs_batch(
    pgs_ids: list[str],
    ref_dir: Path,
    cache_dir: Path,
    genome_build: str = "GRCh38",
    panel: str = DEFAULT_PANEL,
    skip_existing: bool = True,
    match_rate_threshold: float = 0.1,
    output_subdir: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 0,
) -> BatchScoringResult:
    """Score multiple PGS IDs against a reference panel in a single process.

    Iterates over *pgs_ids*, downloading scoring files and calling
    ``compute_reference_prs_polars`` for each.  Per-sample scores are
    aggregated into distribution summaries (5 rows per PGS ID) immediately
    and then discarded to avoid OOM.  Failures are logged and recorded
    in the returned ``outcomes`` list — the loop never aborts.

    Writes the per-PGS quality report, distribution-quality issue report, and
    distributions parquet to *cache_dir/percentiles/* (or
    *cache_dir/percentiles/{output_subdir}/*).

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
        progress_callback: Optional callback invoked with throttled progress payloads.
        progress_every: Emit callback every N processed IDs (0 disables periodic emits;
            final completion event is still emitted when callback is provided).

    Returns:
        ``BatchScoringResult`` with aggregated distributions, per-ID outcomes,
        and a quality DataFrame.  Raw per-sample scores are NOT held in
        memory — they are written to disk per PGS ID and discarded.
    """
    from just_prs.scoring import download_scoring_file, scoring_parquet_path

    scores_cache = cache_dir / "scores"
    scores_cache.mkdir(parents=True, exist_ok=True)

    t_resolve_start = time.monotonic()
    resolved = _ResolvedRefPanel(ref_dir, genome_build)
    t_resolve = time.monotonic() - t_resolve_start

    if progress_callback is not None:
        progress_callback({
            "processed": 0,
            "total": len(pgs_ids),
            "ok": 0, "failed": 0, "problematic": 0, "cached": 0,
            "last_pgs_id": "panel_init",
            "last_status": "panel_resolved",
            "elapsed_sec": t_resolve,
            "eta_sec": None,
            "rate_per_sec": 0.0,
            "is_complete": False,
            "recent_ids": [],
            "panel_resolve_sec": round(t_resolve, 2),
            "pvar_variants": resolved.pvar_variant_ct,
            "n_samples": resolved.psam_df.height,
        })

    outcomes: list[ScoringOutcome] = []
    dist_parts: list[pl.DataFrame] = []
    _DIST_FLUSH_EVERY = 200  # compact dist_parts every N IDs to release small frames
    total = len(pgs_ids)
    started_at = time.monotonic()
    n_processed = 0
    n_ok = 0
    n_failed = 0
    n_problematic = 0
    n_cached = 0
    recent_ids: list[str] = []

    def _emit_progress(last_pgs_id: str, last_status: str, force: bool = False) -> None:
        nonlocal n_processed, n_ok, n_failed, n_problematic, n_cached, recent_ids
        if progress_callback is None:
            return
        
        current_time = time.monotonic()
        # Hack to attach last_log_time to the function object to persist state across calls
        if not hasattr(_emit_progress, "last_log_time"):
            _emit_progress.last_log_time = started_at
            
        time_elapsed = current_time - _emit_progress.last_log_time
        
        should_log = force or (progress_every > 0 and n_processed % progress_every == 0) or (time_elapsed > 15.0)
        if not should_log:
            return
            
        _emit_progress.last_log_time = current_time

        elapsed_sec = current_time - started_at
        rate_per_sec = (n_processed / elapsed_sec) if elapsed_sec > 0 else 0.0
        remaining = max(total - n_processed, 0)
        eta_sec = (remaining / rate_per_sec) if rate_per_sec > 0 else None
        progress_callback({
            "processed": n_processed,
            "total": total,
            "ok": n_ok,
            "failed": n_failed,
            "problematic": n_problematic,
            "cached": n_cached,
            "last_pgs_id": last_pgs_id,
            "last_status": last_status,
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec,
            "rate_per_sec": rate_per_sec,
            "is_complete": n_processed >= total,
            "recent_ids": list(recent_ids),
        })
        recent_ids = []

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
                stale = False
                input_parquet = scoring_parquet_path(pgs_id, scores_cache, genome_build)
                if input_parquet.exists() and input_parquet.stat().st_mtime > result_parquet.stat().st_mtime:
                    stale = True
                    log_message(
                        message_type="reference:batch_cache_stale",
                        pgs_id=pgs_id,
                        reason="scoring input newer than cached score",
                        input_mtime=input_parquet.stat().st_mtime,
                        output_mtime=result_parquet.stat().st_mtime,
                    )

                if not stale:
                    try:
                        dist = _aggregate_single_pgs(result_parquet, pgs_id)
                    except _CorruptParquet as exc:
                        log_message(
                            message_type="reference:batch_corrupt_cache",
                            pgs_id=pgs_id,
                            path=str(result_parquet),
                            error=str(exc),
                        )
                        result_parquet.unlink(missing_ok=True)
                        dist = None
                    if dist is not None:
                        dist_parts.append(dist.df)
                        outcomes.append(dist.outcome)
                        n_processed += 1
                        n_ok += 1
                        n_cached += 1
                        recent_ids.append(pgs_id)
                        log_message(
                            message_type="reference:batch_skip_cached",
                            pgs_id=pgs_id,
                            progress=f"{i + 1}/{len(pgs_ids)}",
                        )
                        _emit_progress(last_pgs_id=pgs_id, last_status="cached")
                        if len(dist_parts) >= _DIST_FLUSH_EVERY:
                            dist_parts = [pl.concat(dist_parts, how="diagonal_relaxed")]
                            gc.collect()
                        continue

            t0 = time.monotonic()
            try:
                parquet_path = scoring_parquet_path(pgs_id, scores_cache, genome_build)
                parquet_valid = False
                if parquet_path.exists():
                    try:
                        pl.scan_parquet(parquet_path).collect_schema()
                        parquet_valid = True
                    except Exception as _pq_exc:
                        log_message(
                            message_type="reference:batch_scoring_cache_corrupt",
                            pgs_id=pgs_id,
                            path=str(parquet_path),
                            error=str(_pq_exc),
                        )
                        try:
                            parquet_path.unlink()
                        except OSError:
                            pass
                if parquet_valid:
                    scoring_file = parquet_path
                else:
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
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                elapsed = round(time.monotonic() - t0, 3)
                exc_type = type(exc).__name__
                error_msg = f"{exc_type}: {exc}"
                log_message(
                    message_type="reference:batch_score_failed",
                    pgs_id=pgs_id,
                    error=error_msg,
                    elapsed_sec=elapsed,
                    progress=f"{i + 1}/{len(pgs_ids)}",
                )
                outcomes.append(ScoringOutcome(
                    pgs_id=pgs_id,
                    status="failed",
                    elapsed_sec=elapsed,
                    error=error_msg,
                ))
                n_processed += 1
                n_failed += 1
                recent_ids.append(pgs_id)
                _emit_progress(last_pgs_id=pgs_id, last_status="failed")
                gc.collect()
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
                variants_total=int(df["variants_total"][0]) if "variants_total" in df.columns and df.height > 0 else None,
                variants_matched=int(df["variants_matched"][0]) if "variants_matched" in df.columns and df.height > 0 else None,
                match_rate=float(df["match_rate"][0]) if "match_rate" in df.columns and df.height > 0 else None,
                n_samples=n_samples,
                score_mean=mean_val,
                score_std=std_val,
                elapsed_sec=elapsed,
            ))
            n_processed += 1
            recent_ids.append(pgs_id)
            if status in ("low_match", "zero_variance"):
                n_problematic += 1
                n_ok += 1
            else:
                n_ok += 1

            log_message(
                message_type="reference:batch_score_done",
                pgs_id=pgs_id,
                status=status,
                n_samples=n_samples,
                elapsed_sec=elapsed,
                progress=f"{i + 1}/{len(pgs_ids)}",
            )
            _emit_progress(last_pgs_id=pgs_id, last_status=status)

            # Periodically compact dist_parts to release per-ID DataFrame memory.
            # Without this, thousands of small DataFrames accumulate in the list
            # and RSS grows monotonically until OOM.
            if len(dist_parts) >= _DIST_FLUSH_EVERY:
                dist_parts = [pl.concat(dist_parts, how="diagonal_relaxed")]
                gc.collect()

        quality_rows = [o.model_dump() for o in outcomes]
        # Avoid Polars schema inference here: reruns can produce hundreds of
        # cached rows with only null elapsed/error fields before a late
        # recomputed or failed row introduces a float/string value.
        quality_df = pl.DataFrame(quality_rows, schema=QUALITY_DF_SCHEMA) if quality_rows else pl.DataFrame(
            schema=QUALITY_DF_SCHEMA
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

        distribution_issues_df = reference_distribution_audit_issues(
            distributions_df,
            quality_df,
            min_match_rate=match_rate_threshold,
        )

        percentiles_dir = cache_dir / "percentiles"
        if output_subdir:
            percentiles_dir = percentiles_dir / output_subdir
        percentiles_dir.mkdir(parents=True, exist_ok=True)
        quality_df.write_parquet(percentiles_dir / f"{panel}_quality.parquet")
        distribution_issues_df.write_parquet(
            percentiles_dir / f"{panel}_distribution_quality_issues.parquet"
        )
        distributions_df.write_parquet(percentiles_dir / f"{panel}_distributions.parquet")

        n_ok = sum(1 for o in outcomes if o.status == "ok")
        n_failed = sum(1 for o in outcomes if o.status == "failed")
        n_low = sum(1 for o in outcomes if o.status == "low_match")
        n_zero = sum(1 for o in outcomes if o.status == "zero_variance")
        n_distribution_errors = distribution_issues_df.filter(pl.col("severity") == "ERROR").height
        n_distribution_warnings = distribution_issues_df.filter(pl.col("severity") == "WARN").height
        _emit_progress(last_pgs_id="batch", last_status="complete", force=True)
        log_message(
            message_type="reference:batch_complete",
            panel=panel,
            n_total=len(pgs_ids),
            n_ok=n_ok,
            n_failed=n_failed,
            n_low_match=n_low,
            n_zero_variance=n_zero,
            n_distributions=distributions_df.height,
            n_distribution_errors=n_distribution_errors,
            n_distribution_warnings=n_distribution_warnings,
        )

        return BatchScoringResult(
            panel=panel,
            distributions_df=distributions_df,
            outcomes=outcomes,
            quality_df=quality_df,
            distribution_issues_df=distribution_issues_df,
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
    if std <= 0 or math.isnan(std) or math.isnan(mean):
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

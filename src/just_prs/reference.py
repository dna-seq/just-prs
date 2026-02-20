"""Reference panel utilities for ancestry-aware PRS percentile estimation.

Provides pure library functions (no Dagster dependency) for working with the
PGS Catalog 1000 Genomes reference panel:
  - Downloading and extracting the reference tarball
  - Running PLINK2 --score on all 2,504 reference individuals
  - Aggregating per-superpopulation distribution statistics
  - Ancestry-matched percentile estimation

These functions are called by Dagster assets in prs-pipeline and can also be
used directly from Python or the CLI.
"""

import math
import subprocess
import tarfile
from pathlib import Path

import httpx
import polars as pl
from eliot import log_message, start_action

from just_prs.scoring import resolve_cache_dir

REFERENCE_PANEL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_1000G_v1.tar.zst"
)

SUPERPOPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def reference_panel_dir(cache_dir: Path | None = None) -> Path:
    """Return the local directory where the reference panel is (or will be) extracted."""
    base = cache_dir or resolve_cache_dir()
    return base / "reference_panel" / "pgsc_1000G_v1"


def download_reference_panel(
    cache_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download and extract pgsc_1000G_v1.tar.zst from the PGS Catalog FTP.

    The tarball (~7 GB) is downloaded into the cache directory and extracted.
    Extraction uses the ``zstandard`` package (via the tarfile module with zstd
    support available in Python 3.14+, or the ``zstandard`` package for earlier
    versions).

    Args:
        cache_dir: Root cache directory. Defaults to resolve_cache_dir().
        overwrite: Re-download even if already extracted.

    Returns:
        Path to the extracted reference panel directory.
    """
    base = cache_dir or resolve_cache_dir()
    dest = reference_panel_dir(cache_dir)

    if dest.exists() and not overwrite:
        log_message(
            message_type="reference:panel_already_exists",
            path=str(dest),
        )
        return dest

    tarball = base / "reference_panel" / "pgsc_1000G_v1.tar.zst"
    tarball.parent.mkdir(parents=True, exist_ok=True)

    with start_action(
        action_type="reference:download_panel",
        url=REFERENCE_PANEL_URL,
        tarball=str(tarball),
    ):
        with httpx.stream("GET", REFERENCE_PANEL_URL, follow_redirects=True, timeout=None) as r:
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

    log_message(message_type="reference:panel_extracted", dest=str(dest))
    return dest


def parse_psam(psam_path: Path) -> pl.DataFrame:
    """Parse a PLINK2 .psam file and return a DataFrame with sample population labels.

    Expected columns: #IID (or IID), SuperPop (or SUP), Population (or POP).
    The '#' prefix on the first column header is stripped automatically.

    Returns:
        DataFrame with columns: iid, superpop, population
    """
    with start_action(action_type="reference:parse_psam", path=str(psam_path)):
        df = pl.read_csv(psam_path, separator="\t", comment_prefix="##")
        # Strip leading '#' from column names (PLINK2 convention)
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


def compute_reference_prs_plink2(
    pgs_id: str,
    scoring_file: Path,
    ref_dir: Path,
    out_dir: Path,
    plink2_bin: Path,
    genome_build: str = "GRCh38",
) -> pl.DataFrame | None:
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

    Returns:
        DataFrame with columns: iid, superpop, population, score, pgs_id
        Returns None if PLINK2 fails or match rate is 0.
    """
    with start_action(
        action_type="reference:compute_plink2_score",
        pgs_id=pgs_id,
        scoring_file=str(scoring_file),
        genome_build=genome_build,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find the .pgen file for the correct build
        build_suffix = "hg38" if genome_build in ("GRCh38", "hg38") else "hg19"
        pgen_files = list(ref_dir.rglob(f"*{build_suffix}*.pgen")) + list(
            ref_dir.rglob("*.pgen")
        )
        if not pgen_files:
            log_message(
                message_type="reference:no_pgen_found",
                ref_dir=str(ref_dir),
                build_suffix=build_suffix,
            )
            return None
        pgen = pgen_files[0]
        pfile_prefix = str(pgen).removesuffix(".pgen")

        # Parse scoring file → PLINK2 score input (ID ALT weight)
        score_input = _prepare_plink2_score_input(scoring_file, pgs_id, out_dir)
        if score_input is None:
            return None

        out_prefix = out_dir / pgs_id
        cmd = [
            str(plink2_bin),
            "--pfile", pfile_prefix, "vzs",
            "--score", str(score_input), "1", "2", "3", "header", "no-mean-imputation",
            "--out", str(out_prefix),
            "--memory", "4096",
            "--threads", "4",
        ]
        log_message(message_type="reference:plink2_cmd", cmd=" ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log_message(
                message_type="reference:plink2_failed",
                pgs_id=pgs_id,
                stderr=result.stderr[-2000:],
            )
            return None

        sscore_file = Path(str(out_prefix) + ".sscore")
        if not sscore_file.exists():
            log_message(message_type="reference:sscore_missing", path=str(sscore_file))
            return None

        scores_df = pl.read_csv(sscore_file, separator="\t")
        # Normalize PLINK2 output column names (vary by version)
        scores_df = scores_df.rename({c: c.lstrip("#") for c in scores_df.columns})
        iid_col = next((c for c in scores_df.columns if c.upper() in ("IID", "#IID")), None)
        score_col = next(
            (c for c in scores_df.columns if "SCORE" in c.upper() and "SUM" in c.upper()),
            None,
        )
        if iid_col is None or score_col is None:
            log_message(
                message_type="reference:sscore_bad_columns",
                columns=scores_df.columns,
            )
            return None

        scores_df = scores_df.rename({iid_col: "iid", score_col: "score"}).select(
            ["iid", "score"]
        )

        psam_files = list(ref_dir.rglob("*.psam"))
        if not psam_files:
            log_message(message_type="reference:no_psam_found", ref_dir=str(ref_dir))
            return None
        psam_df = parse_psam(psam_files[0])

        joined = scores_df.join(psam_df, on="iid", how="inner").with_columns(
            pl.lit(pgs_id).alias("pgs_id")
        )
        log_message(
            message_type="reference:plink2_score_done",
            pgs_id=pgs_id,
            n_samples=joined.height,
        )
        return joined


def _prepare_plink2_score_input(
    scoring_file: Path,
    pgs_id: str,
    out_dir: Path,
) -> Path | None:
    """Convert a PGS Catalog harmonized scoring file to the 3-column PLINK2 --score format.

    PLINK2 --score expects: variant_id, effect_allele, effect_weight (tab-separated).
    Variant IDs are constructed as chr:pos:ref:alt (or chr_pos if no ref/alt).

    Returns path to the prepared score input file, or None if parsing fails.
    """
    import gzip

    try:
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
            return None

        import io
        raw = "".join(data_lines)
        df = pl.read_csv(io.StringIO(raw), separator="\t", infer_schema_length=1000)
        df = df.rename({c: c.lstrip("#") for c in df.columns})

        # Identify columns (harmonized takes priority)
        chr_col = "hm_chr" if "hm_chr" in df.columns else "chr_name"
        pos_col = "hm_pos" if "hm_pos" in df.columns else "chr_position"

        if chr_col not in df.columns or pos_col not in df.columns:
            log_message(
                message_type="reference:score_missing_coords",
                pgs_id=pgs_id,
                columns=df.columns,
            )
            return None

        # Build variant ID: chr:pos (PLINK2 will match by chr:pos when no rsid)
        df = df.with_columns(
            (
                pl.col(chr_col).cast(pl.Utf8).str.replace("(?i)^chr", "")
                + pl.lit(":")
                + pl.col(pos_col).cast(pl.Utf8)
            ).alias("variant_id")
        ).filter(
            pl.col("effect_allele").is_not_null()
            & pl.col("effect_weight").is_not_null()
            & pl.col("variant_id").is_not_null()
        )

        score_input = out_dir / f"{pgs_id}_score_input.txt"
        df.select(["variant_id", "effect_allele", "effect_weight"]).write_csv(
            score_input, separator="\t", include_header=True
        )
        return score_input

    except Exception as exc:
        log_message(
            message_type="reference:score_input_prep_failed",
            pgs_id=pgs_id,
            error=str(exc),
        )
        return None


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

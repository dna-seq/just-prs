"""Consumer-genotyping-chip coverage of PGS scoring files.

Computes, per PGS scoring file and per consumer chip, how many of the score's
variants are *directly typed* on the chip (no imputation). This answers the
practical question for a citizen scientist uploading raw 23andMe / AncestryDNA
data: "which PRS models can I trust on my array, and which need imputation?".

Coverage is a position-set intersection in a single genome build. The Illumina
Global Screening Array (GSA) manifest provides typed-marker coordinates in
GRCh38 (the ``A2`` manifest), which intersects directly against the GRCh38
harmonized PGS scoring files (``hm_chr`` / ``hm_pos``) without any liftover.

Note on chip granularity: 23andMe v5 and AncestryDNA v2 are both built on the
Illumina GSA platform, so the GSA manifest is used as their shared marker core.
It does not include each vendor's custom add-on content (~tens of thousands of
extra markers), so reported coverage is a slight *under*-estimate of the real
chip. Older arrays (23andMe v3/v4 OmniExpress, the defunct deCODEme Omni) are a
different platform and are not represented by the GSA manifest.
"""

import io
import zipfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import polars as pl
from eliot import start_action


class Chip(StrEnum):
    """Consumer genotyping-chip identifiers.

    ``GSA_V3`` is the manifest-backed platform the current consumer market
    converged on (23andMe v5, AncestryDNA v2, etc.). ``OMNIEXPRESS`` covers the
    older v3/v4 kits that ``arrays.detect_chip_generation`` recognizes but for
    which we ship no manifest yet (so it has no typed-position set).
    """

    GSA_V3 = "gsa_v3"
    OMNIEXPRESS = "omniexpress"

# ---------------------------------------------------------------------------
# Chip definitions
# ---------------------------------------------------------------------------

_GSA_24V3_MANIFEST_BASE = (
    "https://support.illumina.com/content/dam/illumina-support/documents/"
    "downloads/productfiles/global-screening-array-24/v3-0/"
)

CHIPS: list[dict[str, object]] = [
    {
        "chip": "gsa_v3",
        "platform": "Illumina Global Screening Array v3.0",
        # The current consumer-array market has converged on the GSA core: all of
        # these add custom content on top of the same GSA backbone, so this single
        # manifest approximates them all. (Older v1/v2 kits used OmniExpress.)
        "consumer_products": "23andMe v5, AncestryDNA v2, MyHeritage (2019+), FamilyTreeDNA v2, LivingDNA",
        # Primary/coverage build — the build chip-coverage labels are computed in
        # (GRCh38 harmonized scoring files intersect the A2 manifest with no liftover).
        "build": "GRCh38",
        # Per-build CSV manifests (~67 MB zip, ~654K markers each):
        #   A2 = GRCh38 coordinates, A1 = GRCh37 coordinates.
        "manifests": {
            "GRCh38": f"{_GSA_24V3_MANIFEST_BASE}GSA-24v3-0-A2-manifest-file-csv.zip",
            "GRCh37": f"{_GSA_24V3_MANIFEST_BASE}GSA-24v3-0-A1-manifest-file-csv.zip",
        },
    },
]

# Keyed by the Chip enum. StrEnum members hash equal to their value, so legacy
# string lookups (e.g. CHIPS_BY_ID["gsa_v3"]) still resolve.
CHIPS_BY_ID: dict[Chip, dict[str, object]] = {Chip(chip["chip"]): chip for chip in CHIPS}

# A PRS is considered "array-ready" (usable on raw consumer-array data without
# imputation) when at least this fraction of its variants are directly typed on
# the chip. 0.90 is strict on purpose: below it, enough variants are missing that
# the score and its percentile drift from the full-coverage value.
ARRAY_READY_THRESHOLD = 0.90

_MIN_MANIFEST_BYTES = 1_000_000  # GSA zip is ~70 MB; reject truncated downloads.


def chip_manifest_dir(cache_dir: Path) -> Path:
    """Return (and create) the directory holding cached chip manifests."""
    path = cache_dir / "chip_manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_chr_expr(col: str) -> pl.Expr:
    """Normalize a chromosome column: strip ``chr`` prefix, uppercase."""
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.replace("(?i)^chr", "")
        .str.to_uppercase()
        .alias("chr_norm")
    )


def download_chip_manifest(chip: Chip, cache_dir: Path, *, build: str = "GRCh38") -> Path:
    """Download a chip manifest zip into the cache, skipping if already present.

    Args:
        chip: Chip identifier (key of ``CHIPS_BY_ID``).
        cache_dir: Root just-prs cache directory.
        build: Genome build whose manifest to fetch (``GRCh38`` → A2, ``GRCh37``
            → A1 for GSA). Must be a key of the chip's ``manifests`` map.

    Returns:
        Path to the downloaded manifest zip.
    """
    import fsspec

    spec = CHIPS_BY_ID[Chip(chip)]
    manifests: dict[str, str] = spec["manifests"]  # type: ignore[assignment]
    if build not in manifests:
        raise ValueError(
            f"chip {Chip(chip)} has no manifest for build {build!r} "
            f"(available: {sorted(manifests)})."
        )
    url = manifests[build]
    # The dest filename embeds the manifest name (e.g. ...A1... / ...A2...), so
    # builds never collide on disk.
    dest = chip_manifest_dir(cache_dir) / f"{Chip(chip).value}_{Path(url).name}"

    if dest.exists() and dest.stat().st_size >= _MIN_MANIFEST_BYTES:
        return dest

    with start_action(action_type="chip_coverage:download_manifest", chip=chip, url=url):
        with fsspec.open(url, "rb") as src:
            data = src.read()
        if len(data) < _MIN_MANIFEST_BYTES:
            raise ValueError(
                f"Downloaded manifest for {chip} is only {len(data)} bytes "
                f"(expected >= {_MIN_MANIFEST_BYTES}); refusing truncated/corrupt file."
            )
        dest.write_bytes(data)
    return dest


def parse_gsa_manifest(zip_path: Path) -> pl.DataFrame:
    """Parse an Illumina GSA CSV manifest zip into typed-marker positions.

    The Illumina manifest is a CSV with a ``[Heading]`` block, an ``[Assay]``
    block (the markers), and a trailing ``[Controls]`` block. Only the assay
    block is parsed; ``[Controls]`` is skipped because its rows have a different
    column shape.

    Args:
        zip_path: Path to the manifest zip (contains a single ``.csv``).

    Returns:
        DataFrame with columns ``name`` (rsID/marker name), ``chr_norm``
        (normalized chromosome), ``pos`` (GRCh38 position, Int64), filtered to
        markers with a valid mapped position.
    """
    with start_action(action_type="chip_coverage:parse_gsa_manifest", path=str(zip_path)):
        with zipfile.ZipFile(zip_path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No .csv member in manifest zip {zip_path}")
            text = zf.read(csv_names[0]).decode("utf-8", errors="replace")

        lines = text.splitlines()
        assay_idx: int | None = None
        controls_idx: int | None = None
        for i, line in enumerate(lines):
            if line.startswith("[Assay]"):
                assay_idx = i
            elif line.startswith("[Controls]"):
                controls_idx = i
                break
        if assay_idx is None:
            raise ValueError(f"No [Assay] section found in manifest {zip_path}")

        end = controls_idx if controls_idx is not None else len(lines)
        # Header is the line immediately after [Assay]; data rows follow until [Controls].
        block = "\n".join(lines[assay_idx + 1 : end])
        df = pl.read_csv(
            io.StringIO(block),
            columns=["Name", "Chr", "MapInfo"],
            schema_overrides={"Name": pl.Utf8, "Chr": pl.Utf8, "MapInfo": pl.Int64},
            ignore_errors=True,
        )

    parsed = (
        df.lazy()
        .select(
            pl.col("Name").alias("name"),
            _normalize_chr_expr("Chr"),
            pl.col("MapInfo").alias("pos"),
        )
        .filter(
            pl.col("pos").is_not_null()
            & (pl.col("pos") > 0)
            & pl.col("chr_norm").is_in(["0", ""]).not_()
        )
        .collect()
    )
    return parsed


def chip_typed_positions(
    chip: Chip,
    cache_dir: Path,
    *,
    build: str = "GRCh38",
    force: bool = False,
) -> pl.DataFrame:
    """Return unique typed (chr_norm, pos) positions for a chip, with parquet caching.

    Args:
        chip: Chip identifier.
        cache_dir: Root just-prs cache directory.
        build: Genome build the typed positions are needed in. GSA ships both the
            A2 (GRCh38) and A1 (GRCh37) manifests, so both builds are supported;
            requesting a build the chip has no manifest for raises ``ValueError``.
        force: If True, re-download and re-parse even when a cache exists.

    Returns:
        DataFrame with unique ``chr_norm``, ``pos`` columns.
    """
    spec = CHIPS_BY_ID[Chip(chip)]
    manifests: dict[str, str] = spec["manifests"]  # type: ignore[assignment]
    if build not in manifests:
        raise ValueError(
            f"chip {Chip(chip)} has no manifest for build {build!r} "
            f"(available: {sorted(manifests)})."
        )

    cache_path = chip_manifest_dir(cache_dir) / f"{Chip(chip).value}_{build}_positions.parquet"
    if cache_path.exists() and not force:
        try:
            return pl.read_parquet(cache_path)
        except (pl.exceptions.ComputeError, OSError):
            cache_path.unlink(missing_ok=True)

    zip_path = download_chip_manifest(Chip(chip), cache_dir, build=build)
    positions = parse_gsa_manifest(zip_path).select("chr_norm", "pos").unique()
    positions.write_parquet(cache_path)
    return positions


def _scoring_positions_lf(parquet_path: Path) -> pl.LazyFrame:
    """Lazily read normalized (chr_norm, pos) from a scoring parquet.

    Prefers harmonized ``hm_chr`` / ``hm_pos`` (used by ``compute_prs``), falling
    back to original ``chr_name`` / ``chr_position`` when harmonized columns are
    absent.
    """
    schema = pl.scan_parquet(parquet_path).collect_schema().names()
    if "hm_chr" in schema and "hm_pos" in schema:
        chr_col, pos_col = "hm_chr", "hm_pos"
    elif "chr_name" in schema and "chr_position" in schema:
        chr_col, pos_col = "chr_name", "chr_position"
    else:
        raise ValueError(
            f"Scoring parquet {parquet_path} lacks (hm_chr, hm_pos) and "
            f"(chr_name, chr_position); columns: {schema}"
        )
    return (
        pl.scan_parquet(parquet_path)
        .select(
            _normalize_chr_expr(chr_col),
            pl.col(pos_col).cast(pl.Int64).alias("pos"),
        )
        .filter(pl.col("pos").is_not_null() & (pl.col("pos") > 0))
    )


def compute_chip_coverage(
    scores_dir: Path,
    cache_dir: Path,
    chips: list[dict[str, object]] | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> pl.DataFrame:
    """Compute per-PGS, per-chip direct-typing coverage of scoring-file variants.

    For each scoring parquet in ``scores_dir`` and each chip, counts how many of
    the score's variants fall on a position the chip directly types. Variants not
    typed would require imputation.

    Args:
        scores_dir: Directory of ``*_hmPOS_GRCh38.parquet`` scoring caches.
        cache_dir: Root just-prs cache directory (for chip manifest caching).
        chips: Chip definitions to evaluate (defaults to module ``CHIPS``).
        progress_callback: Optional callback receiving
            ``{"completed", "total", "chip_index", "n_chips"}``.

    Returns:
        DataFrame with one row per (pgs_id, chip): ``pgs_id``, ``chip``,
        ``platform``, ``consumer_products``, ``build``, ``n_typed``,
        ``n_total``, ``coverage_ratio``, ``array_ready`` (bool —
        ``coverage_ratio >= ARRAY_READY_THRESHOLD`` and the score has mapped
        coordinates; usable on raw array data without imputation).
    """
    chips = chips or CHIPS
    score_files = sorted(scores_dir.glob("*_hmPOS_GRCh38.parquet"))
    total = len(score_files)
    rows: list[dict[str, object]] = []

    with start_action(
        action_type="chip_coverage:compute",
        n_scores=total,
        n_chips=len(chips),
    ):
        for chip_index, spec in enumerate(chips):
            chip = spec["chip"]
            typed = chip_typed_positions(Chip(chip), cache_dir, build=spec["build"]).with_columns(
                pl.lit(True).alias("typed")
            )
            for i, parquet_path in enumerate(score_files):
                pgs_id = parquet_path.name.split("_hmPOS_")[0]
                scored = _scoring_positions_lf(parquet_path)
                joined = scored.join(
                    typed.lazy(), on=["chr_norm", "pos"], how="left"
                )
                agg = joined.select(
                    pl.len().alias("n_total"),
                    pl.col("typed").fill_null(False).sum().alias("n_typed"),
                ).collect()
                n_total = int(agg["n_total"][0])
                n_typed = int(agg["n_typed"][0])
                ratio = (n_typed / n_total) if n_total > 0 else 0.0
                rows.append({
                    "pgs_id": pgs_id,
                    "chip": chip,
                    "platform": spec["platform"],
                    "consumer_products": spec["consumer_products"],
                    "build": spec["build"],
                    "n_typed": n_typed,
                    "n_total": n_total,
                    "coverage_ratio": ratio,
                    # Usable on raw array data (no imputation) when nearly all
                    # variants are directly typed. n_total==0 scores (no mapped
                    # coordinates) are never array-ready.
                    "array_ready": bool(n_total > 0 and ratio >= ARRAY_READY_THRESHOLD),
                })
                if progress_callback and ((i + 1) % 200 == 0 or (i + 1) == total):
                    progress_callback({
                        "completed": i + 1,
                        "total": total,
                        "chip_index": chip_index,
                        "n_chips": len(chips),
                    })

    return pl.DataFrame(rows).sort("chip", "pgs_id")

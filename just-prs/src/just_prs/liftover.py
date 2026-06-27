"""GRCh37 <-> GRCh38 coordinate liftover.

Wraps the pure-Python ``pyliftover`` package so any just-prs capability that
needs to translate coordinates between human genome builds can do so without
reimplementing UCSC chain-file parsing. This module is the canonical entry
point — callers should not import ``pyliftover`` directly.

The design is borrowed from the genomi project's ``runtime/liftover.py`` and
adapted to just-prs conventions (``eliot`` logging, ``resolve_cache_dir``,
lazy-import ``_require_*`` pattern, ``httpx`` streamed downloads, and reuse of
``cleanup.BUILD_NORMALIZATION``).

Inputs and outputs are **1-based VCF-style** coordinates; ``pyliftover`` works
in 0-based BED coordinates under the hood, and this module converts at the
boundary. It is deliberately **chip-agnostic** — a genotyping chip never gets
special liftover logic (see the chip-extension contract in CLAUDE.md).

``pyliftover`` is pure-Python (no C extension), so unlike ``pgenlib``/``pysam``
it installs on Windows; it is declared in the optional ``[liftover]`` extra and
imported lazily so the core install does not require it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx
from eliot import log_message, start_action

from just_prs.cleanup import BUILD_NORMALIZATION
from just_prs.scoring import resolve_cache_dir

# Short aliases layered on top of the canonical metadata mapping so that
# ``"37"`` / ``"b38"`` etc. also resolve. ``BUILD_NORMALIZATION`` already covers
# hg19/hg37/GRCh37 and hg38/GRCh38.
_EXTRA_BUILD_ALIASES: dict[str, str] = {
    "37": "GRCh37",
    "b37": "GRCh37",
    "grch37": "GRCh37",
    "38": "GRCh38",
    "b38": "GRCh38",
    "grch38": "GRCh38",
}

_UCSC_BUILD = {"GRCh37": "hg19", "GRCh38": "hg38"}

CHAIN_FILES: dict[tuple[str, str], str] = {
    ("GRCh38", "GRCh37"): "hg38ToHg19.over.chain.gz",
    ("GRCh37", "GRCh38"): "hg19ToHg38.over.chain.gz",
}

CHAIN_URLS: dict[tuple[str, str], str] = {
    ("GRCh37", "GRCh38"): (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
    ),
    ("GRCh38", "GRCh37"): (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz"
    ),
}

PYLIFTOVER_REQUIREMENT = "pyliftover>=0.4"


class LiftoverConfigurationError(RuntimeError):
    """Raised when a required chain file is missing or pyliftover is unusable."""


def _require_pyliftover() -> None:
    """Raise ImportError with a helpful message if pyliftover is not installed."""
    try:
        import pyliftover as _pyliftover  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyliftover is required for coordinate liftover but is not installed. "
            "Install it with: pip install just-prs[liftover] (or uv sync --extra liftover). "
            "pyliftover is pure-Python and installs on all platforms including Windows."
        ) from None


def normalize_build(value: str) -> str:
    """Map a build label to canonical ``GRCh37`` / ``GRCh38``.

    Reuses :data:`just_prs.cleanup.BUILD_NORMALIZATION` and layers short aliases
    (``37``/``b38``/...) on top. Raises ``ValueError`` for anything that is not a
    liftover-supported human build.
    """

    raw = (value or "").strip()
    key = raw.lower()
    canonical = _EXTRA_BUILD_ALIASES.get(key) or BUILD_NORMALIZATION.get(raw)
    if canonical not in ("GRCh37", "GRCh38"):
        raise ValueError(
            f"unsupported genome build for liftover: {value!r} "
            f"(expected one of GRCh37/hg19, GRCh38/hg38)"
        )
    return canonical


def liftover_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the local directory where UCSC chain files are cached."""
    base = cache_dir or resolve_cache_dir()
    return base / "liftover"


def chain_file_path(
    source_build: str,
    target_build: str,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Return the cached chain-file path for a build pair (may not exist yet)."""
    src, tgt = normalize_build(source_build), normalize_build(target_build)
    try:
        filename = CHAIN_FILES[(src, tgt)]
    except KeyError as exc:
        raise LiftoverConfigurationError(
            f"no chain file registered for {src} -> {tgt}"
        ) from exc
    return liftover_cache_dir(cache_dir) / filename


def download_chain_file(
    source_build: str,
    target_build: str,
    *,
    cache_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download the UCSC chain file for a build pair if not already cached.

    Returns the local path. Rejects a zero-byte download (matching the
    corrupt-file policy used elsewhere in the package).
    """
    src, tgt = normalize_build(source_build), normalize_build(target_build)
    dest = chain_file_path(src, tgt, cache_dir=cache_dir)
    if dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        return dest

    url = CHAIN_URLS[(src, tgt)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.downloading")
    with start_action(
        action_type="liftover:download_chain",
        url=url,
        dest=str(dest),
        source_build=src,
        target_build=tgt,
    ):
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise LiftoverConfigurationError(
                f"downloaded chain file is empty: {url}"
            )
        tmp.replace(dest)
    log_message(message_type="liftover:chain_ready", dest=str(dest))
    return dest


def _pyliftover_installed() -> bool:
    try:
        module = importlib.import_module("pyliftover")
        getattr(module, "LiftOver")
        return True
    except Exception:
        return False


def liftover_preflight(
    source_build: str,
    target_build: str,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Report liftover readiness without downloading anything.

    Liftover needs two independent resources: the ``pyliftover`` package and the
    UCSC chain file on disk. Returns a structured status dict with
    ``status`` in ``{not_required, available, requires_chain_download,
    requires_package_install}`` and ``tool_will_work``.
    """
    src, tgt = normalize_build(source_build), normalize_build(target_build)
    if src == tgt:
        return {
            "status": "not_required",
            "tool_will_work": True,
            "source_build": src,
            "target_build": tgt,
            "reason": "same_build",
        }

    chain_path = chain_file_path(src, tgt, cache_dir=cache_dir)
    chain_exists = chain_path.is_file() and chain_path.stat().st_size > 0
    package_installed = _pyliftover_installed()
    setup = {
        "source_build": src,
        "target_build": tgt,
        "chain_file": {"path": str(chain_path), "exists": chain_exists},
        "chain_url": CHAIN_URLS[(src, tgt)],
        "pyliftover_installed": package_installed,
        "requirement": PYLIFTOVER_REQUIREMENT,
    }
    if not package_installed:
        return {**setup, "status": "requires_package_install", "tool_will_work": False}
    if not chain_exists:
        # Recoverable: the chain auto-downloads on first use.
        return {**setup, "status": "requires_chain_download", "tool_will_work": True}
    return {**setup, "status": "available", "tool_will_work": True}


@dataclass(frozen=True)
class LiftRecordResult:
    lifted: list[dict[str, Any]]
    dropped: list[dict[str, Any]]


class LiftOver:
    """Translate coordinates from ``source_build`` to ``target_build``.

    Instances are cheap to keep around — chain parsing happens once at
    construction (a few hundred ms) and lookups are then in-memory. The chain
    file is auto-downloaded on first construction if absent.
    """

    def __init__(
        self,
        source_build: str,
        target_build: str,
        *,
        cache_dir: Path | None = None,
        auto_download: bool = True,
    ) -> None:
        self.source_build = normalize_build(source_build)
        self.target_build = normalize_build(target_build)
        if self.source_build == self.target_build:
            raise ValueError(
                "source_build and target_build are identical; no liftover needed"
            )
        _require_pyliftover()
        if auto_download:
            self._chain_path = download_chain_file(
                self.source_build, self.target_build, cache_dir=cache_dir
            )
        else:
            self._chain_path = chain_file_path(
                self.source_build, self.target_build, cache_dir=cache_dir
            )
            if not (self._chain_path.is_file() and self._chain_path.stat().st_size > 0):
                raise LiftoverConfigurationError(
                    f"chain file missing: {self._chain_path} "
                    f"(download from {CHAIN_URLS[(self.source_build, self.target_build)]})"
                )
        from pyliftover import LiftOver as _PyLiftOver

        self._lifter = _PyLiftOver(
            str(self._chain_path),
            use_web=False,
            write_cache=False,
        )

    @property
    def chain_path(self) -> Path:
        return self._chain_path

    def lift_position_full(
        self, chrom: str, pos: int
    ) -> tuple[str, int, str] | None:
        """Lift a single 1-based position; return ``(chrom, pos, strand)`` or None.

        ``None`` means the position falls in a chain gap (unmapped). Callers that
        care about strand should inspect the returned strand; :meth:`lift_position`
        is the strand-``+``-only convenience.
        """
        if pos < 1:
            return None
        # pyliftover takes a 0-based BED-style position; convert at the boundary.
        results = self._lifter.convert_coordinate(_ucsc_chrom(chrom), pos - 1)
        if not results:
            return None
        target_chrom, target_pos_zero, strand, _score = results[0]
        return _strip_chr_prefix_like(chrom, target_chrom), target_pos_zero + 1, strand

    def lift_position(self, chrom: str, pos: int) -> tuple[str, int] | None:
        """Lift a single 1-based position, returning ``None`` on a strand flip.

        SNP-only callers should treat strand-flipped hits as unmappable.
        """
        full = self.lift_position_full(chrom, pos)
        if full is None or full[2] != "+":
            return None
        return full[0], full[1]

    def lift_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        chrom_field: str = "chrom",
        pos_field: str = "pos",
    ) -> LiftRecordResult:
        """Lift a stream of row-shaped records.

        Each lifted record is shallow-copied with ``chrom_field``/``pos_field``
        replaced. Records that fail to lift are returned in ``dropped`` with a
        ``"liftover_reason"`` field (``missing_coordinates``, ``invalid_position``,
        ``unmapped``, or ``strand_flipped``).
        """
        lifted: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for record in records:
            chrom = record.get(chrom_field)
            pos_raw = record.get(pos_field)
            if chrom is None or pos_raw is None:
                dropped.append({**dict(record), "liftover_reason": "missing_coordinates"})
                continue
            try:
                pos = int(pos_raw)
            except (TypeError, ValueError):
                dropped.append({**dict(record), "liftover_reason": "invalid_position"})
                continue
            full = self.lift_position_full(str(chrom), pos)
            if full is None:
                dropped.append({**dict(record), "liftover_reason": "unmapped"})
                continue
            target_chrom, target_pos, strand = full
            if strand != "+":
                dropped.append({**dict(record), "liftover_reason": "strand_flipped"})
                continue
            new_record = dict(record)
            new_record[chrom_field] = target_chrom
            new_record[pos_field] = target_pos
            lifted.append(new_record)
        return LiftRecordResult(lifted=lifted, dropped=dropped)


@lru_cache(maxsize=8)
def get_liftover(
    source_build: str,
    target_build: str,
    *,
    cache_dir: Path | None = None,
) -> LiftOver:
    """Cached accessor; reuses the parsed chain file across callers."""
    return LiftOver(source_build, target_build, cache_dir=cache_dir)


def lift_frame(
    df: "Any",  # pl.DataFrame
    source_build: str,
    target_build: str,
    *,
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    cache_dir: Path | None = None,
) -> "tuple[Any, Any]":  # (lifted_df, dropped_df)
    """Lift a polars DataFrame's ``(chrom, pos)`` from one build to another.

    Distinct positions are lifted once (deduplicated) and joined back, so a
    600K-row array lifts in a couple of seconds. Returns ``(lifted, dropped)``:

    - ``lifted``: rows that mapped on the ``+`` strand, with ``chrom_col``/
      ``pos_col`` replaced by the target coordinates (all other columns and row
      order preserved). ``pos_col`` becomes ``Int64`` and ``chrom_col`` ``Utf8``.
    - ``dropped``: rows that failed to lift, with all original columns plus a
      ``liftover_reason`` column (``missing_coordinates``, ``invalid_position``,
      ``unmapped``, or ``strand_flipped``).

    Used to lift both array genotypes and chip-typed position sets; it is
    chip-agnostic.
    """
    import polars as pl

    lifter = get_liftover(source_build, target_build, cache_dir=cache_dir)
    keyed = df.with_columns(
        pl.col(chrom_col).cast(pl.Utf8).alias("__lo_c"),
        pl.col(pos_col).cast(pl.Int64, strict=False).alias("__lo_p"),
    )
    uniq = keyed.select("__lo_c", "__lo_p").unique()

    mapping_rows: list[tuple[Any, Any, str | None, int | None, str | None]] = []
    for c, p in uniq.iter_rows():
        if c is None or p is None:
            mapping_rows.append((c, p, None, None, "missing_coordinates"))
            continue
        full = lifter.lift_position_full(c, int(p))
        if full is None:
            mapping_rows.append((c, p, None, None, "unmapped"))
            continue
        tc, tp, strand = full
        if strand != "+":
            mapping_rows.append((c, p, None, None, "strand_flipped"))
            continue
        mapping_rows.append((c, p, tc, tp, None))

    mapping = pl.DataFrame(
        mapping_rows,
        schema={
            "__lo_c": pl.Utf8,
            "__lo_p": pl.Int64,
            "__lo_tc": pl.Utf8,
            "__lo_tp": pl.Int64,
            "liftover_reason": pl.Utf8,
        },
        orient="row",
    )

    joined = keyed.join(mapping, on=["__lo_c", "__lo_p"], how="left")
    lifted = (
        joined.filter(pl.col("liftover_reason").is_null())
        .with_columns(
            pl.col("__lo_tc").alias(chrom_col),
            pl.col("__lo_tp").alias(pos_col),
        )
        .drop("__lo_c", "__lo_p", "__lo_tc", "__lo_tp", "liftover_reason")
    )
    dropped = joined.filter(pl.col("liftover_reason").is_not_null()).drop(
        "__lo_c", "__lo_p", "__lo_tc", "__lo_tp"
    )
    return lifted, dropped


def _ucsc_chrom(chrom: str) -> str:
    """pyliftover expects UCSC-style ``chrN`` contig names."""
    chrom = str(chrom).strip()
    if not chrom:
        return chrom
    if chrom.startswith("chr"):
        return chrom
    if chrom in {"MT", "mt"}:
        return "chrM"
    return f"chr{chrom}"


def _strip_chr_prefix_like(input_chrom: str, output_chrom: str) -> str:
    """Match the caller's contig-naming convention on the way out.

    If the caller passed ``chr1`` we return ``chr1``; if they passed ``1`` we
    strip the ``chr`` prefix so records round-trip cleanly.
    """
    if input_chrom.startswith("chr"):
        return output_chrom
    if output_chrom.startswith("chr"):
        stripped = output_chrom[3:]
        if stripped == "M":
            return "MT" if input_chrom in {"MT", "mt"} else "M"
        return stripped
    return output_chrom

"""Read-only audit of the just-prs cache directory.

Answers one operational question: **is this cache holding only what this host
needs, and if not, what is the excess and where did it come from?**

The cache mixes artifacts with very different lifetimes — a ~200 MB runtime core
that every host needs, a scoring-file catalog that grows to ~54 GB per genome
build, multi-gigabyte reference panels that only a rebuild host needs, and pure
leaks (a ``.txt.gz`` kept beside its parquet, a panel tarball kept after
extraction, an HF download left duplicated under a nested ``data/`` directory).
Left unclassified they are indistinguishable from each other until the disk
fills, which is how a 301 GB cache went unnoticed until a server crashed.

Classification is driven by :data:`CACHE_MANIFEST`, so a directory nobody
recognises is a **signal** rather than a silent pass — the point is to notice
artifacts that were never supposed to reach a production host.

Nothing here mutates the filesystem.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Anchors — measured 2026-07-28 against the full PGS Catalog
# ---------------------------------------------------------------------------

#: Scoring parquets for one genome build, whole catalog.
PROJECTED_WARM_BYTES_PER_BUILD = 54 * 1024**3

#: Scores in the catalog at the time of measurement (both builds carry all of
#: them: every score is published in both harmonized position files).
PROJECTED_CATALOG_SCORES = 5385

DEFAULT_PRIMARY_BUILD = "GRCh38"


class ArtifactClass(StrEnum):
    """What a cached path is for, and therefore who needs it."""

    RUNTIME = "runtime"
    """Small artifacts every host needs to compute and interpret a PRS."""

    WARM_CATALOG = "warm_catalog"
    """Scoring parquets for the primary build — the bulk of a healthy cache."""

    SECONDARY_BUILD = "secondary_build"
    """Scoring parquets for a build this host does not serve."""

    LEAK = "leak"
    """Redundant by construction: a duplicate or a spent download artifact."""

    REBUILD = "rebuild"
    """Inputs only a pipeline/rebuild host needs (panels, FASTA, intermediates)."""

    DEV = "dev"
    """Test fixtures, sample genomes, user uploads, local results."""

    UNKNOWN = "unknown"
    """Not in the manifest — flagged so new artifacts cannot creep in unnoticed."""


class FlagSeverity(StrEnum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class CacheProfile(StrEnum):
    """Which classes are legitimate for this host."""

    WARM = "warm"
    """Compute-only host: runtime + primary-build scoring parquets."""

    REBUILD = "rebuild"
    """Build host: additionally expects panels, FASTA and intermediates."""


_EXPECTED_BY_PROFILE: dict[CacheProfile, frozenset[ArtifactClass]] = {
    CacheProfile.WARM: frozenset({ArtifactClass.RUNTIME, ArtifactClass.WARM_CATALOG}),
    CacheProfile.REBUILD: frozenset(
        {
            ArtifactClass.RUNTIME,
            ArtifactClass.WARM_CATALOG,
            ArtifactClass.SECONDARY_BUILD,
            ArtifactClass.REBUILD,
        }
    ),
}


class ManifestEntry(BaseModel):
    """A known top-level cache entry."""

    artifact_class: ArtifactClass
    note: str


#: Every top-level name the codebase is known to create under the cache root.
#: ``scores`` and ``reference_panel`` are classified per-file (see ``_classify_*``)
#: because they mix classes; the entry here documents their default.
CACHE_MANIFEST: dict[str, ManifestEntry] = {
    "metadata": ManifestEntry(
        artifact_class=ArtifactClass.RUNTIME,
        note="Cleaned PGS Catalog metadata parquets",
    ),
    "percentiles": ManifestEntry(
        artifact_class=ArtifactClass.RUNTIME,
        note="Reference distributions, quality + audit sidecars, chip coverage",
    ),
    "reference": ManifestEntry(
        artifact_class=ArtifactClass.RUNTIME,
        note="Reference-allele universes pulled from HF",
    ),
    "ancestry": ManifestEntry(
        artifact_class=ArtifactClass.RUNTIME,
        note="Ancestry models (SVD loadings, reference PCs)",
    ),
    "chip_manifests": ManifestEntry(
        artifact_class=ArtifactClass.RUNTIME,
        note="Chip typed-position parquets (manifest zips are rebuild-only)",
    ),
    "liftover": ManifestEntry(
        artifact_class=ArtifactClass.RUNTIME,
        note="UCSC chain files",
    ),
    "scores": ManifestEntry(
        artifact_class=ArtifactClass.WARM_CATALOG,
        note="Scoring files — classified per file by build and extension",
    ),
    "reference_panel": ManifestEntry(
        artifact_class=ArtifactClass.REBUILD,
        note="Extracted PLINK2 panels — tarballs are spent download artifacts",
    ),
    "reference_fasta": ManifestEntry(
        artifact_class=ArtifactClass.REBUILD,
        note="Ensembl primary assemblies — precompute-only input",
    ),
    "reference_scores": ManifestEntry(
        artifact_class=ArtifactClass.DEV,
        note="Per-individual reference PRS (pipeline output, not published)",
    ),
    "normalized": ManifestEntry(
        artifact_class=ArtifactClass.DEV,
        note="Normalized genotype parquets from uploads",
    ),
    "genomes": ManifestEntry(
        artifact_class=ArtifactClass.DEV,
        note="Alias VCFs auto-downloaded from Zenodo",
    ),
    "test-data": ManifestEntry(
        artifact_class=ArtifactClass.DEV,
        note="Test fixtures",
    ),
    "results": ManifestEntry(
        artifact_class=ArtifactClass.DEV,
        note="PRS result cache",
    ),
    "plink2": ManifestEntry(
        artifact_class=ArtifactClass.REBUILD,
        note="Auto-downloaded PLINK2 binary (LD-pruning at build time)",
    ),
    "scoring": ManifestEntry(
        artifact_class=ArtifactClass.LEAK,
        note="Legacy write-only UI layout — nothing reads it",
    ),
}


class ArtifactEntry(BaseModel):
    """One classified path in the cache."""

    path: str
    artifact_class: ArtifactClass
    bytes: int
    n_files: int
    note: str = ""


class CacheFlag(BaseModel):
    """A condition worth surfacing to an operator."""

    code: str
    severity: FlagSeverity
    message: str
    bytes: int = 0
    n_files: int = 0
    paths: list[str] = Field(default_factory=list)
    reclaim_hint: str = ""


class CatalogCoverage(BaseModel):
    """How complete the warm scoring catalog is for the primary build."""

    primary_build: str
    n_primary_parquet: int
    n_projected: int = PROJECTED_CATALOG_SCORES
    warm_bytes: int = 0
    projected_full_bytes: int = PROJECTED_WARM_BYTES_PER_BUILD

    @property
    def ratio(self) -> float:
        return self.n_primary_parquet / self.n_projected if self.n_projected else 0.0


class CacheAuditReport(BaseModel):
    """Complete read-only assessment of a cache directory."""

    cache_dir: str
    profile: CacheProfile
    primary_build: str
    total_bytes: int
    n_files: int
    entries: list[ArtifactEntry]
    by_class: dict[ArtifactClass, int]
    expected_bytes: int
    delta_bytes: int
    flags: list[CacheFlag]
    catalog: CatalogCoverage

    @property
    def has_errors(self) -> bool:
        return any(f.severity is FlagSeverity.ERROR for f in self.flags)


# ---------------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------------


class _Walk(BaseModel):
    """Accumulated size of a subtree."""

    bytes: int = 0
    n_files: int = 0


def _walk(path: Path, seen: set[tuple[int, int]]) -> _Walk:
    """Sum apparent file sizes under ``path``, counting hardlinked inodes once.

    Uses ``os.scandir`` rather than ``Path.rglob``: the scores directory alone
    holds >20k entries and rglob's per-entry ``Path`` construction dominates.
    """
    total = _Walk()
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    key = (st.st_dev, st.st_ino)
                    if st.st_nlink > 1:
                        if key in seen:
                            continue
                        seen.add(key)
                    total.bytes += st.st_size
                    total.n_files += 1
        except OSError:
            continue
    return total


def _file_size(path: Path, seen: set[tuple[int, int]]) -> _Walk:
    try:
        st = path.stat()
    except OSError:
        return _Walk()
    key = (st.st_dev, st.st_ino)
    if st.st_nlink > 1:
        if key in seen:
            return _Walk()
        seen.add(key)
    return _Walk(bytes=st.st_size, n_files=1)


# ---------------------------------------------------------------------------
# Per-directory classification
# ---------------------------------------------------------------------------


def _classify_scores_dir(
    scores_dir: Path, primary_build: str, seen: set[tuple[int, int]]
) -> list[ArtifactEntry]:
    """Split the scores cache by build and extension.

    A ``.txt.gz`` here is always a leak: the managed cache is parquet-only, and
    the gz is discarded as soon as its parquet is verified readable.
    """
    buckets: dict[tuple[ArtifactClass, str], _Walk] = {}

    def add(cls: ArtifactClass, label: str, size: _Walk) -> None:
        cur = buckets.setdefault((cls, label), _Walk())
        cur.bytes += size.bytes
        cur.n_files += size.n_files

    try:
        entries = list(os.scandir(scores_dir))
    except OSError:
        return []

    for entry in entries:
        p = Path(entry.path)
        if entry.is_dir(follow_symlinks=False):
            if p.name == "data":
                add(ArtifactClass.LEAK, "nested HF data/ mirror", _walk(p, seen))
            else:
                add(ArtifactClass.RUNTIME, f"{p.name}/", _walk(p, seen))
            continue

        size = _file_size(p, seen)
        name = p.name
        if name.endswith(".txt.gz"):
            add(ArtifactClass.LEAK, ".txt.gz beside parquet", size)
        elif name.endswith(".parquet") and "_hmPOS_" in name:
            build = name.rsplit("_hmPOS_", 1)[1].removesuffix(".parquet")
            if build == primary_build:
                add(ArtifactClass.WARM_CATALOG, f"{build} scoring parquets", size)
            else:
                add(ArtifactClass.SECONDARY_BUILD, f"{build} scoring parquets", size)
        else:
            add(ArtifactClass.RUNTIME, name, size)

    return [
        ArtifactEntry(
            path=f"scores/ [{label}]",
            artifact_class=cls,
            bytes=walk.bytes,
            n_files=walk.n_files,
        )
        for (cls, label), walk in sorted(buckets.items(), key=lambda kv: -kv[1].bytes)
    ]


def _classify_reference_panel_dir(
    panel_dir: Path, seen: set[tuple[int, int]]
) -> list[ArtifactEntry]:
    """Separate spent panel tarballs from the extracted panels."""
    out: list[ArtifactEntry] = []
    try:
        entries = list(os.scandir(panel_dir))
    except OSError:
        return out

    for entry in entries:
        p = Path(entry.path)
        if entry.is_dir(follow_symlinks=False):
            walk = _walk(p, seen)
            out.append(
                ArtifactEntry(
                    path=f"reference_panel/{p.name}",
                    artifact_class=ArtifactClass.REBUILD,
                    bytes=walk.bytes,
                    n_files=walk.n_files,
                    note="extracted panel",
                )
            )
        elif p.name.endswith(".tar.zst"):
            size = _file_size(p, seen)
            out.append(
                ArtifactEntry(
                    path=f"reference_panel/{p.name}",
                    artifact_class=ArtifactClass.LEAK,
                    bytes=size.bytes,
                    n_files=size.n_files,
                    note="spent tarball — panel already extracted",
                )
            )
        else:
            size = _file_size(p, seen)
            out.append(
                ArtifactEntry(
                    path=f"reference_panel/{p.name}",
                    artifact_class=ArtifactClass.REBUILD,
                    bytes=size.bytes,
                    n_files=size.n_files,
                )
            )
    return out


def _classify_generic_dir(
    directory: Path, entry: ManifestEntry, seen: set[tuple[int, int]]
) -> list[ArtifactEntry]:
    """Classify a manifest directory, splitting out nested HF ``data/`` mirrors."""
    out: list[ArtifactEntry] = []
    nested = directory / "data"
    if nested.is_dir():
        walk = _walk(nested, seen)
        out.append(
            ArtifactEntry(
                path=f"{directory.name}/data",
                artifact_class=ArtifactClass.LEAK,
                bytes=walk.bytes,
                n_files=walk.n_files,
                note="HF download duplicate — should have been moved, not copied",
            )
        )

    rest = _Walk()
    try:
        for child in os.scandir(directory):
            p = Path(child.path)
            if p.name == "data" and child.is_dir(follow_symlinks=False):
                continue
            sub = _walk(p, seen) if child.is_dir(follow_symlinks=False) else _file_size(p, seen)
            rest.bytes += sub.bytes
            rest.n_files += sub.n_files
    except OSError:
        pass

    out.append(
        ArtifactEntry(
            path=f"{directory.name}/",
            artifact_class=entry.artifact_class,
            bytes=rest.bytes,
            n_files=rest.n_files,
            note=entry.note,
        )
    )
    return out


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def _bytes_of(entries: list[ArtifactEntry], cls: ArtifactClass) -> tuple[int, int, list[str]]:
    hits = [e for e in entries if e.artifact_class is cls]
    return (
        sum(e.bytes for e in hits),
        sum(e.n_files for e in hits),
        [e.path for e in hits],
    )


def _build_flags(
    cache_dir: Path,
    entries: list[ArtifactEntry],
    profile: CacheProfile,
    primary_build: str,
    catalog: CatalogCoverage,
) -> list[CacheFlag]:
    flags: list[CacheFlag] = []
    by_path = {e.path: e for e in entries}

    def entry_for(predicate) -> list[ArtifactEntry]:
        return [e for e in entries if predicate(e)]

    gz = [e for e in entries if ".txt.gz" in e.path]
    if gz:
        flags.append(
            CacheFlag(
                code="scoring_gz_present",
                severity=FlagSeverity.ERROR,
                message=(
                    "Scoring .txt.gz found beside parquet — the parquet-only cache "
                    "invariant is broken (a stale cache, or a regression)"
                ),
                bytes=sum(e.bytes for e in gz),
                n_files=sum(e.n_files for e in gz),
                paths=[e.path for e in gz],
                reclaim_hint=f"find {cache_dir}/scores -name '*.txt.gz' -delete",
            )
        )

    nested = [e for e in entries if e.path.endswith("/data")]
    if nested:
        flags.append(
            CacheFlag(
                code="hf_nested_data_dir",
                severity=FlagSeverity.ERROR,
                message=(
                    "Nested HF data/ mirror present — every pulled artifact is "
                    "stored twice (hf_hub_download replicates the repo layout)"
                ),
                bytes=sum(e.bytes for e in nested),
                n_files=sum(e.n_files for e in nested),
                paths=[e.path for e in nested],
                reclaim_hint=(
                    f"find {cache_dir} -type d -name data -prune -exec rm -rf {{}} +"
                ),
            )
        )

    tarballs = entry_for(lambda e: e.path.endswith(".tar.zst"))
    if tarballs:
        flags.append(
            CacheFlag(
                code="panel_tarball_present",
                severity=FlagSeverity.ERROR,
                message="Reference panel tarball retained after extraction",
                bytes=sum(e.bytes for e in tarballs),
                n_files=sum(e.n_files for e in tarballs),
                paths=[e.path for e in tarballs],
                reclaim_hint=f"rm {cache_dir}/reference_panel/*.tar.zst",
            )
        )

    legacy = by_path.get("scoring/")
    if legacy is not None:
        flags.append(
            CacheFlag(
                code="legacy_scoring_dir",
                severity=FlagSeverity.ERROR,
                message=(
                    "Legacy <cache>/scoring/ layout present — written by the old UI "
                    "download path and read by nothing"
                ),
                bytes=legacy.bytes,
                n_files=legacy.n_files,
                paths=[legacy.path],
                reclaim_hint=f"rm -rf {cache_dir}/scoring",
            )
        )

    strays = entry_for(lambda e: e.path.startswith("<root>/") and "_hmPOS_" in e.path)
    if strays:
        flags.append(
            CacheFlag(
                code="stray_scores_at_cache_root",
                severity=FlagSeverity.WARN,
                message="Scoring files sitting at the cache root instead of scores/",
                bytes=sum(e.bytes for e in strays),
                n_files=sum(e.n_files for e in strays),
                paths=[e.path for e in strays],
                reclaim_hint=f"rm {cache_dir}/*_hmPOS_*",
            )
        )

    expected = _EXPECTED_BY_PROFILE[profile]

    if ArtifactClass.SECONDARY_BUILD not in expected:
        size, n, paths = _bytes_of(entries, ArtifactClass.SECONDARY_BUILD)
        if size:
            flags.append(
                CacheFlag(
                    code="secondary_build_scores",
                    severity=FlagSeverity.WARN,
                    message=(
                        f"Scoring parquets for a build other than {primary_build}. "
                        "Every score is published in both harmonized builds, so the "
                        f"{primary_build} set already covers the whole catalog unless "
                        "this host scores samples in their native build."
                    ),
                    bytes=size,
                    n_files=n,
                    paths=paths,
                )
            )

    if ArtifactClass.REBUILD not in expected:
        size, n, paths = _bytes_of(entries, ArtifactClass.REBUILD)
        if size:
            flags.append(
                CacheFlag(
                    code="rebuild_artifacts_present",
                    severity=FlagSeverity.WARN,
                    message=(
                        "Pipeline/rebuild inputs on a compute-only host "
                        "(reference panels, genome FASTA, build intermediates)"
                    ),
                    bytes=size,
                    n_files=n,
                    paths=paths,
                )
            )

    size, n, paths = _bytes_of(entries, ArtifactClass.DEV)
    if size:
        flags.append(
            CacheFlag(
                code="dev_artifacts_present",
                severity=FlagSeverity.WARN,
                message="Test fixtures, sample genomes, uploads or local results",
                bytes=size,
                n_files=n,
                paths=paths,
            )
        )

    size, n, paths = _bytes_of(entries, ArtifactClass.UNKNOWN)
    if size or paths:
        flags.append(
            CacheFlag(
                code="unknown_path",
                severity=FlagSeverity.WARN,
                message=(
                    "Path not present in the reference dev/test environment — "
                    "an artifact this tool has never seen before"
                ),
                bytes=size,
                n_files=n,
                paths=paths,
            )
        )

    if 0 < catalog.n_primary_parquet < catalog.n_projected:
        flags.append(
            CacheFlag(
                code="catalog_incomplete",
                severity=FlagSeverity.INFO,
                message=(
                    f"Warm catalog is {catalog.ratio:.1%} populated "
                    f"({catalog.n_primary_parquet:,}/{catalog.n_projected:,} scores)"
                ),
            )
        )

    return flags


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def scan_cache(
    cache_dir: Path,
    primary_build: str = DEFAULT_PRIMARY_BUILD,
    profile: CacheProfile = CacheProfile.WARM,
) -> CacheAuditReport:
    """Classify every path under ``cache_dir`` and report excess against ``profile``.

    Args:
        cache_dir: Cache root (``resolve_cache_dir()`` by convention).
        primary_build: The build this host serves; scoring parquets for any other
            build are counted as excess.
        profile: Which artifact classes are legitimate here.

    Returns:
        A :class:`CacheAuditReport`. The filesystem is never modified.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    seen: set[tuple[int, int]] = set()
    entries: list[ArtifactEntry] = []

    for child in sorted(os.scandir(cache_dir), key=lambda e: e.name):
        path = Path(child.path)
        name = path.name
        manifest = CACHE_MANIFEST.get(name)

        if child.is_dir(follow_symlinks=False):
            if name == "scores":
                entries.extend(_classify_scores_dir(path, primary_build, seen))
            elif name == "reference_panel":
                entries.extend(_classify_reference_panel_dir(path, seen))
            elif name == "data":
                walk = _walk(path, seen)
                entries.append(
                    ArtifactEntry(
                        path="data/",
                        artifact_class=ArtifactClass.LEAK,
                        bytes=walk.bytes,
                        n_files=walk.n_files,
                        note="HF download duplicate at cache root",
                    )
                )
            elif manifest is not None:
                entries.extend(_classify_generic_dir(path, manifest, seen))
            else:
                walk = _walk(path, seen)
                entries.append(
                    ArtifactEntry(
                        path=f"{name}/",
                        artifact_class=ArtifactClass.UNKNOWN,
                        bytes=walk.bytes,
                        n_files=walk.n_files,
                    )
                )
            continue

        size = _file_size(path, seen)
        if "_hmPOS_" in name:
            entries.append(
                ArtifactEntry(
                    path=f"<root>/{name}",
                    artifact_class=ArtifactClass.LEAK,
                    bytes=size.bytes,
                    n_files=size.n_files,
                    note="scoring file outside scores/",
                )
            )
        elif manifest is not None:
            entries.append(
                ArtifactEntry(
                    path=name,
                    artifact_class=manifest.artifact_class,
                    bytes=size.bytes,
                    n_files=size.n_files,
                    note=manifest.note,
                )
            )
        else:
            entries.append(
                ArtifactEntry(
                    path=name,
                    artifact_class=ArtifactClass.UNKNOWN,
                    bytes=size.bytes,
                    n_files=size.n_files,
                )
            )

    by_class: dict[ArtifactClass, int] = {cls: 0 for cls in ArtifactClass}
    for entry in entries:
        by_class[entry.artifact_class] += entry.bytes

    total_bytes = sum(e.bytes for e in entries)
    n_files = sum(e.n_files for e in entries)

    warm_entries = [e for e in entries if e.artifact_class is ArtifactClass.WARM_CATALOG]
    catalog = CatalogCoverage(
        primary_build=primary_build,
        n_primary_parquet=sum(e.n_files for e in warm_entries),
        warm_bytes=sum(e.bytes for e in warm_entries),
    )

    expected_classes = _EXPECTED_BY_PROFILE[profile]
    expected_bytes = sum(by_class[cls] for cls in expected_classes)

    return CacheAuditReport(
        cache_dir=str(cache_dir),
        profile=profile,
        primary_build=primary_build,
        total_bytes=total_bytes,
        n_files=n_files,
        entries=sorted(entries, key=lambda e: -e.bytes),
        by_class=by_class,
        expected_bytes=expected_bytes,
        delta_bytes=total_bytes - expected_bytes,
        flags=_build_flags(cache_dir, entries, profile, primary_build, catalog),
        catalog=catalog,
    )


def format_bytes(n: int) -> str:
    """Human-readable size using binary units."""
    size = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if abs(size) < 1024 or unit == "T":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} T"

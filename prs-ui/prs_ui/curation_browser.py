"""Display frames for the Curated tab.

Compiles the reviewable YAML under ``curation/`` with ``just_prs.curation.build_tables``
and projects the published columns into grid-ready frames. The tab reads this module;
By PRS and By Trait do not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from just_prs.curation import CurationTables, build_tables

CURATION_VIEWS: tuple[str, ...] = ("scores", "review", "clusters")
CURATION_STATUSES: tuple[str, ...] = ("all", "human_verified", "agent_proposed")

_REVIEW_STATUS: dict[str, str] = {
    "human_verified": "Human verified",
    "agent_proposed": "Unreviewed",
}
_VALENCE: dict[str, str] = {
    "desirable": "Desirable",
    "undesirable": "Undesirable",
    "neutral": "Neutral",
    "context_dependent": "Context-dependent",
}
_DIRECTION: dict[str, str] = {
    "increases": "Increases",
    "decreases": "Decreases",
}
_CONFIDENCE: dict[str, str] = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}
_LEAD: dict[str, str] = {
    "agree": "Agrees with published associations",
    "disagree": "Disagrees with published associations",
    "mixed": "Mixed against published associations",
    "not_run": "Not checked",
}

SCORE_COLUMNS: tuple[str, ...] = (
    "pgs_id",
    "review_status",
    "higher_score_means",
    "valence",
    "concept",
    "phenotype",
    "direction",
    "confidence",
    "paper",
    "review_reason",
    "alignment",
    "axis",
    "target",
    "coding",
    "cluster",
    "lead_variants",
    "corrections",
    "unannotated_reason",
    "notes",
    "evidence",
    "sources",
)
CLUSTER_COLUMNS: tuple[str, ...] = (
    "domain",
    "concept",
    "axis",
    "valence",
    "phenotype",
    "versus_axis",
    "n_scores",
    "review_status",
    "valence_basis",
)

SCORE_DETAIL_FIELDS: list[str] = [
    "evidence",
    "sources",
    "alignment",
    "axis",
    "target",
    "coding",
    "cluster",
    "lead_variants",
    "corrections",
    "unannotated_reason",
    "notes",
]
SCORE_DETAIL_LABELS: dict[str, str] = {
    "evidence": "Evidence",
    "sources": "Sources",
    "alignment": "Alignment with the phenotype",
    "axis": "Concept axis",
    "target": "Target",
    "coding": "Coding",
    "cluster": "Cluster",
    "lead_variants": "Lead variants",
    "corrections": "Corrections",
    "unannotated_reason": "Why there is no direction",
    "notes": "Notes",
}
CLUSTER_DETAIL_FIELDS: list[str] = ["valence_basis"]
CLUSTER_DETAIL_LABELS: dict[str, str] = {
    "valence_basis": "Why this valence",
}


@dataclass(frozen=True)
class CurationBrowserData:
    """Compiled curation tables plus the directory they were read from."""

    tables: CurationTables
    source: Path

    @property
    def n_verified(self) -> int:
        return _status_count(self.tables.annotations, "human_verified")

    @property
    def n_unreviewed(self) -> int:
        return _status_count(self.tables.annotations, "agent_proposed")

    @property
    def n_review(self) -> int:
        return self.tables.review.height


def find_curation_root(start: Path) -> Path | None:
    """Return the ``curation/`` directory that contains ``clusters.yaml``."""
    start = start.resolve()
    if (start / "clusters.yaml").is_file():
        return start
    for parent in (start, *start.parents):
        candidate = parent / "curation"
        if (candidate / "clusters.yaml").is_file():
            return candidate
    return None


def compile_curation_browser(root: Path) -> CurationBrowserData:
    """Compile ``root`` (the ``curation/`` directory) into browser tables.

    Raises:
        CurationBuildError: The YAML is inconsistent, so nothing is shown as curated.
    """
    return CurationBrowserData(tables=build_tables(root), source=root)


def prepare_score_display(frame: pl.DataFrame, status: str) -> pl.DataFrame:
    """Project a score annotation or review frame into grid columns.

    ``status`` is ``all``, ``human_verified``, or ``agent_proposed``.
    """
    filtered = frame
    if status in ("human_verified", "agent_proposed"):
        filtered = frame.filter(pl.col("annotation_status") == status)
    labeled = filtered.with_columns(
        _label("annotation_status", _REVIEW_STATUS).alias("review_status"),
        pl.col("higher_percentile_means").fill_null("").alias("higher_score_means"),
        _label("valence", _VALENCE).alias("valence"),
        pl.col("concept_label").fill_null("").alias("concept"),
        pl.col("phenotype_label").fill_null("").alias("phenotype"),
        _label("direction", _DIRECTION).alias("direction"),
        _label("confidence", _CONFIDENCE).alias("confidence"),
        pl.col("pgp_id").alias("paper"),
        pl.col("review_reason").fill_null("").alias("review_reason"),
        pl.when(pl.col("polarity") == 1)
        .then(pl.lit("Same direction as the phenotype"))
        .when(pl.col("polarity") == -1)
        .then(pl.lit("Opposite of the phenotype"))
        .otherwise(pl.lit(""))
        .alias("alignment"),
        pl.col("axis").fill_null(""),
        pl.col("target").fill_null(""),
        pl.col("coding").fill_null(""),
        pl.col("cluster").fill_null(""),
        _lead_text().alias("lead_variants"),
        pl.col("correction_kinds").fill_null("None").alias("corrections"),
        pl.col("unannotated_reason").fill_null(""),
        pl.col("notes").fill_null(""),
        pl.col("provenance_json")
        .map_elements(_evidence_text, return_dtype=pl.Utf8)
        .alias("evidence"),
        pl.struct(
            "pgs_catalog_url",
            "publication_url",
            "pubmed_url",
            "europepmc_url",
            "doi_url",
        )
        .map_elements(_source_links_json, return_dtype=pl.Utf8)
        .alias("sources"),
    )
    return labeled.select(SCORE_COLUMNS)


def prepare_cluster_display(frame: pl.DataFrame) -> pl.DataFrame:
    """Project trait clusters into grid columns, used concepts first."""
    return (
        frame.with_columns(
            pl.col("domain_label").fill_null("").alias("domain"),
            pl.col("concept_label").fill_null("").alias("concept"),
            pl.col("axis").fill_null(""),
            _label("valence", _VALENCE).alias("valence"),
            pl.col("phenotype_label").fill_null("").alias("phenotype"),
            pl.when(pl.col("phenotype_sign") == 1)
            .then(pl.lit("Same as the axis"))
            .when(pl.col("phenotype_sign") == -1)
            .then(pl.lit("Opposite of the axis"))
            .otherwise(pl.lit(""))
            .alias("versus_axis"),
            _label("concept_status", _REVIEW_STATUS).alias("review_status"),
            pl.col("valence_basis").fill_null(""),
        )
        .select(CLUSTER_COLUMNS)
        .sort(["n_scores", "domain", "concept", "phenotype"], descending=[True, False, False, False])
    )


def score_column_overrides(view: str) -> dict[str, dict[str, object]]:
    """Column widths, badges, and links for a score or review grid."""
    overrides: dict[str, dict[str, object]] = {
        "pgs_id": {
            "width": 130,
            "headerName": "PGS",
            "cellRendererType": "url",
            "cellRendererConfig": {
                "baseUrl": "https://www.pgscatalog.org/score/",
                "suffixUrl": "/",
                "color": "#1565c0",
            },
        },
        "review_status": {
            "width": 150,
            "headerName": "Review",
            **_badge(
                {
                    "Human verified": ("#2e7d32", "#e8f5e9"),
                    "Unreviewed": ("#e65100", "#fff3e0"),
                }
            ),
        },
        "higher_score_means": {
            "minWidth": 280,
            "flex": 2,
            "headerName": "Higher score means",
        },
        "valence": {
            "width": 170,
            "headerName": "Valence",
            **_badge(
                {
                    "Desirable": ("#2e7d32", "#e8f5e9"),
                    "Undesirable": ("#c62828", "#ffebee"),
                    "Neutral": ("#455a64", "#eceff1"),
                    "Context-dependent": ("#1565c0", "#e3f2fd"),
                }
            ),
        },
        "concept": {"minWidth": 160, "flex": 1, "headerName": "Concept"},
        "phenotype": {"minWidth": 160, "flex": 1, "headerName": "Phenotype"},
        "direction": {"width": 120, "headerName": "Direction"},
        "confidence": {"width": 120, "headerName": "Confidence"},
        "paper": {
            "width": 130,
            "headerName": "Paper",
            "cellRendererType": "url",
            "cellRendererConfig": {
                "baseUrl": "https://www.pgscatalog.org/publication/",
                "suffixUrl": "/",
                "color": "#1565c0",
            },
        },
        "review_reason": {"hide": True},
        "alignment": {"hide": True},
        "axis": {"hide": True},
        "target": {"hide": True},
        "coding": {"hide": True},
        "cluster": {"hide": True},
        "lead_variants": {"hide": True},
        "corrections": {"hide": True},
        "unannotated_reason": {"hide": True},
        "notes": {"hide": True},
        "evidence": {"hide": True},
        "sources": {"hide": True},
    }
    if view == "review":
        overrides["review_reason"] = {
            "minWidth": 240,
            "flex": 1,
            "headerName": "Why it needs review",
        }
    return overrides


def cluster_column_overrides() -> dict[str, dict[str, object]]:
    """Column widths and badges for the trait-cluster grid."""
    return {
        "domain": {"width": 140, "headerName": "Domain"},
        "concept": {"minWidth": 180, "flex": 1, "headerName": "Concept"},
        "axis": {"minWidth": 180, "flex": 1, "headerName": "Axis"},
        "valence": {
            "width": 170,
            "headerName": "Valence",
            **_badge(
                {
                    "Desirable": ("#2e7d32", "#e8f5e9"),
                    "Undesirable": ("#c62828", "#ffebee"),
                    "Neutral": ("#455a64", "#eceff1"),
                    "Context-dependent": ("#1565c0", "#e3f2fd"),
                }
            ),
        },
        "phenotype": {"minWidth": 180, "flex": 1, "headerName": "Phenotype"},
        "versus_axis": {"width": 180, "headerName": "Versus axis"},
        "n_scores": {"width": 100, "headerName": "Scores"},
        "review_status": {
            "width": 150,
            "headerName": "Concept review",
            **_badge(
                {
                    "Human verified": ("#2e7d32", "#e8f5e9"),
                    "Unreviewed": ("#e65100", "#fff3e0"),
                }
            ),
        },
        "valence_basis": {"hide": True},
    }


def _status_count(frame: pl.DataFrame, status: str) -> int:
    if frame.is_empty():
        return 0
    return frame.filter(pl.col("annotation_status") == status).height


def _label(column: str, mapping: dict[str, str]) -> pl.Expr:
    return pl.col(column).replace_strict(mapping, default="").fill_null("")


def _lead_text() -> pl.Expr:
    label = _label("lead_variant_status", _LEAD)
    note = pl.col("lead_variant_note").fill_null("")
    return pl.when(note == "").then(label).otherwise(label + pl.lit(" — ") + note)


def _badge(colors: dict[str, tuple[str, str]]) -> dict[str, object]:
    return {
        "cellRendererType": "badge",
        "cellRendererConfig": {
            "colorMap": {label: fg for label, (fg, _bg) in colors.items()},
            "bgColorMap": {label: bg for label, (_fg, bg) in colors.items()},
        },
    }


def _evidence_text(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(items, list):
        return raw
    blocks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        locator = str(item.get("locator") or "").strip()
        source = str(item.get("source") or "").replace("_", " ")
        confidence = str(item.get("confidence") or "")
        header = ", ".join(part for part in (source, confidence, locator) if part)
        if header and quote:
            blocks.append(f"{header}\n{quote}")
        elif quote or header:
            blocks.append(quote or header)
    return "\n\n".join(blocks)


def _source_links_json(row: dict[str, object]) -> str:
    items: list[dict[str, str]] = []
    for label, key in (
        ("PGS Catalog", "pgs_catalog_url"),
        ("Publication", "publication_url"),
        ("PubMed", "pubmed_url"),
        ("Europe PMC", "europepmc_url"),
        ("DOI", "doi_url"),
    ):
        url = row.get(key)
        if isinstance(url, str) and url:
            items.append({"label": label, "url": url})
    return json.dumps(items)


__all__ = [
    "CLUSTER_COLUMNS",
    "CLUSTER_DETAIL_FIELDS",
    "CLUSTER_DETAIL_LABELS",
    "CURATION_STATUSES",
    "CURATION_VIEWS",
    "CurationBrowserData",
    "SCORE_COLUMNS",
    "SCORE_DETAIL_FIELDS",
    "SCORE_DETAIL_LABELS",
    "cluster_column_overrides",
    "compile_curation_browser",
    "find_curation_root",
    "prepare_cluster_display",
    "prepare_score_display",
    "score_column_overrides",
]

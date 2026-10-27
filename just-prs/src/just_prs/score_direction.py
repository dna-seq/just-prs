"""Curated score direction: what a higher score means, read on its concept's axis.

The curation (``curation/`` → ``prs curation push`` → ``just-dna-seq/pgs-catalog``
``data/metadata/``) records, per PGS ID, which way the score points and which concept it
belongs to. This module is the single home of the rules for *consuming* it (AGENTS.md,
"Consuming curated directions"): the per-score columns ``PRSCatalog.scores()`` joins, the
axis alignment ``trait_summary`` aggregates on, and the valence tone the UI colours with.
Nothing here reads YAML or the network; ``PRSCatalog`` loads the published tables.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

import polars as pl


class CurationStatus(StrEnum):
    """Where a score stands in the curation."""

    #: A published row of ``score_annotations.parquet``: the direction is used.
    RESOLVED = "resolved"
    #: In ``curation_review.parquet`` with a proposed direction: shown, never counted or coloured.
    IN_REVIEW = "in_review"
    #: In ``curation_review.parquet`` without a direction (no verbatim evidence found).
    NO_DIRECTION = "no_direction"


#: Columns joined onto catalog scores and carried on every result row.
SCORE_DIRECTION_COLUMNS: tuple[str, ...] = (
    "curation_status",
    "higher_percentile_means",
    "direction",
    "target",
    "cluster",
    "concept_label",
    "axis",
    "valence",
    "phenotype",
    "phenotype_label",
    "phenotype_sign",
    "polarity",
    "annotation_status",
    "curation_review_reason",
    "correction_kinds",
)

SCORE_DIRECTION_SCHEMA: dict[str, pl.DataType] = {
    "pgs_id": pl.Utf8(),
    "curation_status": pl.Utf8(),
    "higher_percentile_means": pl.Utf8(),
    "direction": pl.Utf8(),
    "target": pl.Utf8(),
    "cluster": pl.Utf8(),
    "concept_label": pl.Utf8(),
    "axis": pl.Utf8(),
    "valence": pl.Utf8(),
    "phenotype": pl.Utf8(),
    "phenotype_label": pl.Utf8(),
    "phenotype_sign": pl.Int8(),
    "polarity": pl.Int8(),
    "annotation_status": pl.Utf8(),
    "curation_review_reason": pl.Utf8(),
    "correction_kinds": pl.Utf8(),
}

DEPOSIT_INVERTED = "deposit_inverted"

VALENCE_LABELS: dict[str, str] = {
    "desirable": "Desirable",
    "undesirable": "Undesirable",
    "neutral": "Neutral",
    "context_dependent": "Context-dependent",
}
_COLOURED_VALENCES = frozenset({"desirable", "undesirable"})


def score_direction_table(
    annotations: pl.DataFrame | None,
    review: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per PGS ID with :data:`SCORE_DIRECTION_COLUMNS`.

    ``annotations`` is ``score_annotations.parquet`` (resolved rows), ``review`` is
    ``curation_review.parquet``. A review row with a direction is ``in_review``, one
    without is ``no_direction``. When an ID appears in both (a mixed cache), the
    resolved row wins.
    """
    parts: list[pl.DataFrame] = []
    for frame, resolved in ((annotations, True), (review, False)):
        if frame is None or frame.height == 0:
            continue
        status = (
            pl.lit(CurationStatus.RESOLVED.value)
            if resolved
            else pl.when(pl.col("direction").is_not_null())
            .then(pl.lit(CurationStatus.IN_REVIEW.value))
            .otherwise(pl.lit(CurationStatus.NO_DIRECTION.value))
        )
        review_reason = (
            pl.col("review_reason") if "review_reason" in frame.columns else pl.lit(None, dtype=pl.Utf8)
        )
        frame = frame.with_columns(
            status.alias("curation_status"),
            review_reason.alias("curation_review_reason"),
            pl.lit(0 if resolved else 1).alias("_rank"),
        )
        parts.append(
            frame.select(
                *[
                    (pl.col(name) if name in frame.columns else pl.lit(None)).cast(dtype).alias(name)
                    for name, dtype in SCORE_DIRECTION_SCHEMA.items()
                ],
                "_rank",
            )
        )
    if not parts:
        return pl.DataFrame(schema=SCORE_DIRECTION_SCHEMA)
    return (
        pl.concat(parts)
        .sort("pgs_id", "_rank")
        .unique(subset=["pgs_id"], keep="first", maintain_order=True)
        .drop("_rank")
    )


# ---------------------------------------------------------------------------
# Row helpers (rows are plain dicts: UI rows, EnrichedPRSResult dumps, CLI rows)
# ---------------------------------------------------------------------------


def curation_status(row: dict[str, Any]) -> str | None:
    """``resolved`` / ``in_review`` / ``no_direction``, or None when the score is not curated."""
    value = row.get("curation_status")
    return str(value) if value else None


def is_resolved(row: dict[str, Any]) -> bool:
    """True when the row's direction is published (rule 4/5: the only rows coloured and counted)."""
    return curation_status(row) == CurationStatus.RESOLVED.value and row_polarity(row) is not None


def row_polarity(row: dict[str, Any]) -> int | None:
    """+1 when a higher score moves up the concept axis, -1 when down, None when unknown."""
    value = row.get("polarity")
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number in (1, -1) else None


def correction_kinds(row: dict[str, Any]) -> list[str]:
    """Correction kinds recorded for the score (``deposit_inverted``, ``broken_weights`` …)."""
    raw = row.get("correction_kinds")
    return [part for part in str(raw or "").split(";") if part]


def has_inverted_deposit(row: dict[str, Any]) -> bool:
    """The deposited file is recorded as the negated paper score (direction is as distributed)."""
    return DEPOSIT_INVERTED in correction_kinds(row)


def axis_percentile(percentile: float | None, polarity: int | None) -> float | None:
    """Percentile on the concept axis: ``p`` for polarity +1, ``100 - p`` for -1."""
    if percentile is None or polarity is None:
        return None
    if isinstance(percentile, float) and not math.isfinite(percentile):
        return None
    return float(percentile) if polarity > 0 else 100.0 - float(percentile)


def unfavourable_percentile(axis_pct: float | None, valence: str | None) -> float | None:
    """How far toward the unwanted end of the axis (0-100); None for uncoloured valences."""
    if axis_pct is None or valence not in _COLOURED_VALENCES:
        return None
    return axis_pct if valence == "undesirable" else 100.0 - axis_pct


def tone_from_unfavourable(unfavourable: float | None) -> str:
    """Red only for alarming, green for favourable (the legacy percentile thresholds)."""
    if unfavourable is None:
        return "neutral"
    if unfavourable >= 90:
        return "danger"
    if unfavourable >= 75:
        return "warning"
    if unfavourable < 25:
        return "good"
    return "neutral"


def legacy_percentile_tone(percentile: float | None) -> str:
    """Tone for an uncurated score: assumes a higher percentile means more of the reported trait."""
    return tone_from_unfavourable(percentile)


def direction_tone(percentile: float | None, row: dict[str, Any]) -> str:
    """Valence tone for one score's (raw) percentile under AGENTS.md rule 4."""
    status = curation_status(row)
    if status is None:
        return legacy_percentile_tone(percentile)
    if not is_resolved(row):
        return "neutral"
    axis_pct = axis_percentile(percentile, row_polarity(row))
    return tone_from_unfavourable(unfavourable_percentile(axis_pct, _valence(row)))


def axis_tone(axis_pct: float | None, valence: str | None) -> str:
    """Valence tone for a percentile already on the concept axis (trait-summary medians)."""
    return tone_from_unfavourable(unfavourable_percentile(axis_pct, valence))


def higher_score_means(row: dict[str, Any]) -> str:
    """Plain sentence for "what does a higher score mean" (always a sentence, never empty)."""
    status = curation_status(row)
    means = _sentence(row.get("higher_percentile_means"))
    if status == CurationStatus.RESOLVED.value and means:
        return means
    if status == CurationStatus.IN_REVIEW.value and means:
        return f"{means} (proposed; in review)"
    if status == CurationStatus.NO_DIRECTION.value:
        return "Direction not determined yet (in review)"
    return "Direction not curated yet"


def curation_badge(row: dict[str, Any]) -> tuple[str, str]:
    """Short (label, tone) badge for the curation state of a score."""
    status = curation_status(row)
    if status == CurationStatus.RESOLVED.value:
        if has_inverted_deposit(row):
            return "Curated · inverted deposit", "warning"
        verified = str(row.get("annotation_status") or "") == "human_verified"
        return ("Curated · verified" if verified else "Curated"), "info"
    if status == CurationStatus.IN_REVIEW.value:
        return "In review", "warning"
    if status == CurationStatus.NO_DIRECTION.value:
        return "No direction", "warning"
    return "Not curated", "neutral"


def direction_detail(row: dict[str, Any]) -> str:
    """Longer explanation for detail panels: concept, axis, valence, review reason, corrections."""
    status = curation_status(row)
    if status is None:
        return (
            "This score's direction has not been curated yet, so colours assume a higher "
            "percentile means more of the reported trait."
        )
    parts: list[str] = []
    concept = str(row.get("concept_label") or "")
    axis = str(row.get("axis") or "")
    if concept:
        parts.append(f"Concept: {concept}" + (f" (axis: {axis})" if axis else "") + ".")
    phenotype = str(row.get("phenotype_label") or "")
    polarity = row_polarity(row)
    if phenotype and polarity is not None:
        along = "up" if polarity > 0 else "down"
        parts.append(f"Phenotype: {phenotype}; a higher score moves {along} the axis.")
    valence = _valence(row)
    if valence:
        parts.append(f"Valence: {VALENCE_LABELS.get(valence, valence)}.")
    if has_inverted_deposit(row):
        parts.append(
            "The deposited scoring file runs opposite to its paper (inverted deposit); "
            "the direction shown describes the file you scored."
        )
    reason = str(row.get("curation_review_reason") or "")
    if status != CurationStatus.RESOLVED.value:
        parts.append(
            "In the curation review queue, so it is not coloured or counted in trait summaries"
            + (f": {reason}" if reason else ".")
        )
    elif str(row.get("annotation_status") or "") != "human_verified":
        parts.append("Agent-curated with a verbatim quote; not yet verified by a person.")
    return " ".join(parts)


def _valence(row: dict[str, Any]) -> str | None:
    value = row.get("valence")
    return str(value) if value else None


def _sentence(value: Any) -> str:
    text = str(value or "").strip()
    return text[:1].upper() + text[1:] if text else ""


__all__ = [
    "DEPOSIT_INVERTED",
    "SCORE_DIRECTION_COLUMNS",
    "SCORE_DIRECTION_SCHEMA",
    "VALENCE_LABELS",
    "CurationStatus",
    "axis_percentile",
    "axis_tone",
    "correction_kinds",
    "curation_badge",
    "curation_status",
    "direction_detail",
    "direction_tone",
    "has_inverted_deposit",
    "higher_score_means",
    "is_resolved",
    "legacy_percentile_tone",
    "row_polarity",
    "score_direction_table",
    "tone_from_unfavourable",
    "unfavourable_percentile",
]

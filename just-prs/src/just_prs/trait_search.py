"""Synonym-aware trait search over PGS Catalog scores.

Grouping scores into curated concepts must never make a score harder to find, so every name a
score is known by stays searchable: its catalog names (ontology label and ID, reported trait,
score name) plus the labels and aliases of the curated concept it belongs to
(``concept_search_terms.parquet`` from ``prs curation build``).

Matching is token-AND over one lowercase text per score: every word of the query must occur
somewhere in that score's names, so "night owl", "owl night" and "creatinine" all work.
"""

import re

import polars as pl

#: Catalog columns whose values become search terms. Composite ontology labels (``a|b``) are split.
CATALOG_TERM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("trait_efo", "catalog_trait"),
    ("trait_efo_id", "ontology_id"),
    ("trait_reported", "reported_trait"),
    ("name", "score_name"),
)
#: Curated term kinds shown to people as "also known as" (catalog names are shown elsewhere).
DISPLAY_TERM_KINDS: tuple[str, ...] = ("concept_label", "alias", "phenotype_label", "reported_trait", "catalog_trait")

_TOKEN_RE = re.compile(r"[\w'-]+")


def score_search_terms(
    scores: pl.DataFrame,
    concept_terms: pl.DataFrame | None = None,
    score_concepts: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Every searchable term of every score. Grain: ``(pgs_id, term)``.

    ``scores`` needs ``pgs_id`` plus any of ``CATALOG_TERM_COLUMNS``. ``score_concepts``
    (``pgs_id``, ``cluster``) maps curated scores to concepts; each such score then inherits all
    ``concept_terms`` (``cluster``, ``term``, ``term_kind``) of its concept, so an alias written
    once on the concept finds every member score.
    """
    parts = [
        scores.select(
            "pgs_id",
            pl.col(column).cast(pl.Utf8).str.split("|").alias("term"),
            pl.lit(kind).alias("term_kind"),
        ).explode("term")
        for column, kind in CATALOG_TERM_COLUMNS
        if column in scores.columns
    ]
    if concept_terms is not None and score_concepts is not None and concept_terms.height and score_concepts.height:
        parts.append(
            score_concepts.select("pgs_id", "cluster")
            .unique()
            .join(concept_terms.select("cluster", "term", "term_kind"), on="cluster", how="inner")
            .select("pgs_id", "term", "term_kind")
        )
    if not parts:
        return pl.DataFrame(schema={"pgs_id": pl.Utf8, "term": pl.Utf8, "term_kind": pl.Utf8})
    return (
        pl.concat(parts)
        .with_columns(pl.col("term").str.strip_chars())
        .filter(pl.col("term").is_not_null() & (pl.col("term") != ""))
        .unique(subset=["pgs_id", "term"], maintain_order=True)
    )


def query_tokens(query: str) -> list[str]:
    """Lowercase word tokens of a search query."""
    return _TOKEN_RE.findall(query.lower())


def matching_pgs_ids(terms: pl.DataFrame, query: str) -> set[str]:
    """PGS IDs whose combined names contain every word of ``query`` (empty query → none)."""
    tokens = query_tokens(query)
    if not tokens:
        return set()
    text = terms.group_by("pgs_id").agg(pl.col("term").str.join(" ").str.to_lowercase().alias("text"))
    condition = pl.all_horizontal([pl.col("text").str.contains(token, literal=True) for token in tokens])
    return set(text.filter(condition)["pgs_id"])


def group_aliases(terms: pl.DataFrame, groups: pl.DataFrame, max_terms: int = 8) -> pl.DataFrame:
    """Short "also known as" text per trait group. Grain: ``trait``.

    ``groups`` maps ``pgs_id`` → ``trait`` (the displayed group label). Curated labels and aliases
    come first, then reported and ontology names; the group's own label is left out.
    """
    rank = {kind: i for i, kind in enumerate(DISPLAY_TERM_KINDS)}
    shown = (
        groups.select("pgs_id", "trait")
        .join(terms.filter(pl.col("term_kind").is_in(DISPLAY_TERM_KINDS)), on="pgs_id", how="inner")
        .with_columns(
            pl.col("term_kind").replace_strict(rank, return_dtype=pl.Int32).alias("_rank"),
            pl.col("term").str.to_lowercase().alias("_key"),
        )
        .filter(pl.col("_key") != pl.col("trait").str.to_lowercase())
        .sort("trait", "_rank", "term")
        .unique(subset=["trait", "_key"], keep="first", maintain_order=True)
    )
    return shown.group_by("trait", maintain_order=True).agg(
        pl.col("term").head(max_terms).str.join(" · ").alias("also_known_as")
    )

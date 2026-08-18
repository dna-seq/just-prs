"""Build ``papers`` and ``score_paper_links`` from publications + full performance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import polars as pl
from eliot import start_action

from just_prs.sample_scores.evidence.models import (
    AbstractStatus,
    PaperRecord,
    PaperRelationship,
    ScorePaperLink,
)

_EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_PMID_RE = re.compile(r"^\d{5,9}$")


def paper_id_from(*, pmid: object = None, doi: object = None, pgp_id: object = None) -> str | None:
    """Prefer PMID, then DOI, then PGP."""
    if pmid is not None and str(pmid).strip() and str(pmid).strip().lower() not in {"none", "nan"}:
        digits = re.sub(r"\D", "", str(pmid))
        if digits:
            return digits
    if doi is not None and str(doi).strip() and str(doi).strip().lower() not in {"none", "nan"}:
        return f"DOI:{str(doi).strip()}"
    if pgp_id is not None and str(pgp_id).strip():
        return str(pgp_id).strip()
    return None


def abstract_is_redistributable(license_text: str | None) -> bool:
    """True only for public-domain / CC0 / CC-BY (no NC/SA/ND)."""
    if not license_text:
        return False
    cleaned = re.sub(r"[\s_]+", "-", license_text.strip().lower())
    cleaned = cleaned.replace("cc-by-4.0", "cc-by").replace("cc-by-4", "cc-by")
    if cleaned in {"cc0", "public-domain", "pd", "cc-by"}:
        return True
    return False


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    return text


def _seed_paper(
    *,
    paper_id: str,
    pmid: str | None,
    doi: str | None,
    pgp_id: str | None,
    title: str | None = None,
    authors: str | None = None,
    first_author: str | None = None,
    journal: str | None = None,
    date_publication: str | None = None,
) -> PaperRecord:
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
    epmc_url = f"https://europepmc.org/article/MED/{pmid}" if pmid else None
    doi_url = f"https://doi.org/{doi}" if doi else None
    return PaperRecord(
        paper_id=paper_id,
        pmid=pmid,
        doi=doi,
        pgp_id=pgp_id,
        title=title,
        authors=authors,
        first_author=first_author,
        journal=journal,
        date_publication=date_publication,
        pubmed_url=pubmed_url,
        europepmc_url=epmc_url,
        doi_url=doi_url,
        abstract_status=AbstractStatus.LINK_ONLY.value,
        resolution_status="catalog",
    )


def collect_score_paper_links(
    scores_df: pl.DataFrame,
    publications_df: pl.DataFrame | None,
    performance_df: pl.DataFrame | None,
) -> tuple[dict[str, PaperRecord], list[ScorePaperLink]]:
    """Collect development + evaluation paper links. Does not call Europe PMC."""
    papers: dict[str, PaperRecord] = {}
    links: list[ScorePaperLink] = []
    pubs_by_pgp: dict[str, dict[str, object]] = {}
    if publications_df is not None and publications_df.height:
        for row in publications_df.iter_rows(named=True):
            pgp = _clean_optional(row.get("pgp_id"))
            if pgp:
                pubs_by_pgp[pgp] = row

    def _add_link(
        pgs_id: str,
        relationship: str,
        *,
        pmid: object = None,
        doi: object = None,
        pgp_id: object = None,
        ppm_id: object = None,
        title: object = None,
        authors: object = None,
        first_author: object = None,
        journal: object = None,
        date_publication: object = None,
    ) -> None:
        paper_id = paper_id_from(pmid=pmid, doi=doi, pgp_id=pgp_id)
        if paper_id is None:
            return
        pmid_text = _clean_optional(pmid)
        if pmid_text:
            pmid_text = re.sub(r"\D", "", pmid_text) or pmid_text
        doi_text = _clean_optional(doi)
        pgp_text = _clean_optional(pgp_id)
        if paper_id not in papers:
            papers[paper_id] = _seed_paper(
                paper_id=paper_id,
                pmid=pmid_text,
                doi=doi_text,
                pgp_id=pgp_text,
                title=_clean_optional(title),
                authors=_clean_optional(authors),
                first_author=_clean_optional(first_author),
                journal=_clean_optional(journal),
                date_publication=_clean_optional(date_publication),
            )
        else:
            existing = papers[paper_id]
            if pgp_text and not existing.pgp_id:
                existing.pgp_id = pgp_text
            if doi_text and not existing.doi:
                existing.doi = doi_text
                existing.doi_url = f"https://doi.org/{doi_text}"
        links.append(
            ScorePaperLink(
                pgs_id=pgs_id,
                paper_id=paper_id,
                relationship_type=relationship,
                pgp_id=pgp_text,
                ppm_id=_clean_optional(ppm_id),
            )
        )

    for row in scores_df.iter_rows(named=True):
        pgs_id = _clean_optional(row.get("pgs_id"))
        if pgs_id is None:
            continue
        pgp = _clean_optional(row.get("pgp_id"))
        pub = pubs_by_pgp.get(pgp or "", {})
        _add_link(
            pgs_id,
            PaperRelationship.DEVELOPMENT.value,
            pmid=row.get("pmid") or pub.get("pmid"),
            doi=pub.get("doi"),
            pgp_id=pgp,
            title=pub.get("title"),
            authors=pub.get("authors"),
            first_author=pub.get("first_author"),
            journal=pub.get("journal"),
            date_publication=pub.get("date_publication"),
        )

    if performance_df is not None and performance_df.height:
        for row in performance_df.iter_rows(named=True):
            pgs_id = _clean_optional(row.get("pgs_id"))
            if pgs_id is None:
                continue
            pgp = _clean_optional(row.get("pgp_id"))
            pub = pubs_by_pgp.get(pgp or "", {})
            _add_link(
                pgs_id,
                PaperRelationship.EVALUATION.value,
                pmid=row.get("pmid") or pub.get("pmid"),
                doi=row.get("doi") or pub.get("doi"),
                pgp_id=pgp,
                ppm_id=row.get("ppm_id"),
                title=pub.get("title"),
                authors=pub.get("authors"),
                first_author=pub.get("first_author"),
                journal=pub.get("journal"),
                date_publication=pub.get("date_publication"),
            )

    unique_links = {
        (link.pgs_id, link.paper_id, link.relationship_type): link for link in links
    }
    return papers, list(unique_links.values())


def _epmc_cache_path(cache_dir: Path, pmid: str) -> Path:
    return cache_dir / f"{pmid}.json"


def fetch_europepmc_batch(
    pmids: list[str],
    cache_dir: Path | None = None,
    *,
    allow_network: bool = True,
    batch_size: int = 40,
) -> dict[str, dict[str, object]]:
    """Resolve citation metadata from Europe PMC. Cached per PMID."""
    wanted = [pmid for pmid in pmids if _PMID_RE.match(pmid)]
    found: dict[str, dict[str, object]] = {}
    pending: list[str] = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for pmid in wanted:
            path = _epmc_cache_path(cache_dir, pmid)
            if path.exists():
                found[pmid] = json.loads(path.read_text(encoding="utf-8"))
            else:
                pending.append(pmid)
    else:
        pending = list(wanted)
    if not allow_network or not pending:
        return found

    with start_action(action_type="sample_scores:evidence:europepmc", n=len(pending)):
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            query = " OR ".join(f"EXT_ID:{pmid}" for pmid in chunk)
            response = httpx.get(
                _EUROPEPMC_SEARCH,
                params={
                    "query": query,
                    "resultType": "core",
                    "format": "json",
                    "pageSize": str(len(chunk)),
                },
                timeout=30.0,
                follow_redirects=True,
            )
            if response.status_code != 200:
                continue
            payload = response.json()
            results = payload.get("resultList", {}).get("result", []) or []
            for item in results:
                pmid = str(item.get("pmid") or item.get("id") or "")
                if not pmid:
                    continue
                found[pmid] = item
                if cache_dir is not None:
                    _epmc_cache_path(cache_dir, pmid).write_text(
                        json.dumps(item), encoding="utf-8"
                    )
    return found


def apply_europepmc(
    papers: dict[str, PaperRecord],
    epmc_by_pmid: dict[str, dict[str, object]],
) -> None:
    """Mutate paper records with Europe PMC metadata and license-gated abstracts."""
    for paper in papers.values():
        if not paper.pmid or paper.pmid not in epmc_by_pmid:
            continue
        item = epmc_by_pmid[paper.pmid]
        paper.pmcid = _clean_optional(item.get("pmcid")) or paper.pmcid
        paper.doi = _clean_optional(item.get("doi")) or paper.doi
        paper.title = _clean_optional(item.get("title")) or paper.title
        paper.authors = _clean_optional(item.get("authorString")) or paper.authors
        paper.first_author = paper.first_author or (
            paper.authors.split(",")[0].strip() if paper.authors else None
        )
        paper.journal = _clean_optional(item.get("journalTitle")) or paper.journal
        paper.date_publication = _clean_optional(item.get("firstPublicationDate")) or paper.date_publication
        paper.is_open_access = str(item.get("isOpenAccess") or "").lower() in {"y", "yes", "true"}
        license_text = _clean_optional(item.get("license"))
        paper.content_license = license_text
        paper.citation_text = _clean_optional(item.get("authorString"))
        if paper.doi and not paper.doi_url:
            paper.doi_url = f"https://doi.org/{paper.doi}"
        paper.resolution_status = "europepmc"
        abstract = _clean_optional(item.get("abstractText"))
        if abstract and abstract_is_redistributable(license_text):
            paper.abstract_text = abstract
            paper.abstract_status = AbstractStatus.INCLUDED.value
        elif abstract:
            paper.abstract_text = None
            paper.abstract_status = AbstractStatus.LINK_ONLY.value
        else:
            paper.abstract_status = AbstractStatus.UNAVAILABLE.value


def build_paper_tables(
    scores_df: pl.DataFrame,
    publications_df: pl.DataFrame | None,
    performance_df: pl.DataFrame | None,
    *,
    epmc_cache_dir: Path | None = None,
    allow_network: bool = True,
) -> tuple[list[PaperRecord], list[ScorePaperLink]]:
    papers, links = collect_score_paper_links(scores_df, publications_df, performance_df)
    pmids = [paper.pmid for paper in papers.values() if paper.pmid]
    epmc = fetch_europepmc_batch(pmids, epmc_cache_dir, allow_network=allow_network)
    apply_europepmc(papers, epmc)
    return list(papers.values()), links

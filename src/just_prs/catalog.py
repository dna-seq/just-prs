"""PGS Catalog REST API client for querying scores, traits, and publications."""

from collections.abc import Iterator
from typing import cast

import httpx
from eliot import start_action

from just_prs.models import EffectSizeInfo, PerformanceInfo, ScoreInfo, TraitInfo

PGS_CATALOG_BASE_URL = "https://www.pgscatalog.org/rest"
DEFAULT_TIMEOUT = 30.0


class PGSCatalogClient:
    """Synchronous client for the PGS Catalog REST API."""

    def __init__(
        self,
        base_url: str = PGS_CATALOG_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> "PGSCatalogClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_json(self, endpoint: str, params: dict[str, str | int] | None = None) -> dict:
        """Make a GET request and return JSON response."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.client.get(url, params={**(params or {}), "format": "json"})
        response.raise_for_status()
        return response.json()

    def get_score(self, pgs_id: str) -> ScoreInfo:
        """Fetch metadata for a single PGS score by ID.

        Args:
            pgs_id: PGS Catalog score ID (e.g. "PGS000001")

        Returns:
            ScoreInfo with full metadata
        """
        with start_action(action_type="pgs_catalog:get_score", pgs_id=pgs_id):
            data = self._get_json(f"/score/{pgs_id}")
            return ScoreInfo.model_validate(data)

    def search_scores(self, term: str, limit: int = 25) -> list[ScoreInfo]:
        """Search PGS scores by term (trait name, PGS ID, etc.).

        Args:
            term: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching ScoreInfo objects
        """
        with start_action(action_type="pgs_catalog:search_scores", term=term, limit=limit):
            data = self._get_json("/score/search", params={"term": term, "limit": limit})
            results = data.get("results", [])
            return [ScoreInfo.model_validate(r) for r in results]

    def list_scores_page(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[ScoreInfo], int]:
        """Fetch a single page of all PGS scores from the catalog.

        Args:
            limit: Number of results per page
            offset: Starting offset for pagination

        Returns:
            Tuple of (list of ScoreInfo, total count in catalog)
        """
        with start_action(
            action_type="pgs_catalog:list_scores_page",
            limit=limit,
            offset=offset,
        ):
            data = self._get_json(
                "/score/all", params={"limit": limit, "offset": offset}
            )
            results = data.get("results", [])
            total = cast(int, data.get("count", 0))
            return [ScoreInfo.model_validate(r) for r in results], total

    def iter_all_scores(
        self, page_size: int = 50, max_results: int | None = None
    ) -> Iterator[ScoreInfo]:
        """Iterate over all PGS scores in the catalog, page by page.

        Args:
            page_size: Number of results per API request
            max_results: Optional cap on total results (None = fetch all)

        Yields:
            ScoreInfo for each score in the catalog
        """
        offset = 0
        yielded = 0
        with start_action(
            action_type="pgs_catalog:iter_all_scores",
            page_size=page_size,
        ):
            while True:
                page, total = self.list_scores_page(limit=page_size, offset=offset)
                if not page:
                    break
                for s in page:
                    yield s
                    yielded += 1
                    if max_results is not None and yielded >= max_results:
                        return
                offset += len(page)
                if offset >= total or (
                    max_results is not None and yielded >= max_results
                ):
                    break

    def get_trait(self, efo_id: str) -> TraitInfo:
        """Fetch metadata for a trait by EFO ID.

        Args:
            efo_id: EFO trait ID (e.g. "EFO_0001645")

        Returns:
            TraitInfo with full metadata
        """
        with start_action(action_type="pgs_catalog:get_trait", efo_id=efo_id):
            data = self._get_json(f"/trait/{efo_id}")
            return TraitInfo.model_validate(data)

    def search_traits(self, term: str, limit: int = 25) -> list[TraitInfo]:
        """Search traits by term.

        Args:
            term: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching TraitInfo objects
        """
        with start_action(action_type="pgs_catalog:search_traits", term=term, limit=limit):
            data = self._get_json("/trait/search", params={"term": term, "limit": limit})
            results = data.get("results", [])
            return [TraitInfo.model_validate(r) for r in results]

    def list_scores_for_trait(self, efo_id: str) -> list[ScoreInfo]:
        """List all PGS scores associated with a given trait.

        Args:
            efo_id: EFO trait ID

        Returns:
            List of ScoreInfo objects for scores associated with the trait
        """
        with start_action(action_type="pgs_catalog:list_scores_for_trait", efo_id=efo_id):
            trait = self.get_trait(efo_id)
            scores: list[ScoreInfo] = []
            for pgs_id in trait.associated_pgs_ids:
                score = self.get_score(pgs_id)
                scores.append(score)
            return scores

    def get_performance_metrics(self, pgs_id: str) -> PerformanceInfo | None:
        """Fetch performance metrics for a PGS score and return the best one.

        Selects the metric with the largest evaluation sample size,
        preferring European-ancestry cohorts when available.

        Args:
            pgs_id: PGS Catalog score ID (e.g. "PGS000001")

        Returns:
            PerformanceInfo for the best available metric, or None if none exist
        """
        with start_action(action_type="pgs_catalog:get_performance_metrics", pgs_id=pgs_id):
            data = self._get_json("/performance/search", params={"pgs_id": pgs_id})
            results = data.get("results", [])
            if not results:
                return None

            best: dict | None = None
            best_score = -1

            for entry in results:
                sampleset = entry.get("sampleset", {})
                samples = sampleset.get("samples", [])
                sample_number = sum(s.get("sample_number", 0) for s in samples)
                ancestry = samples[0].get("ancestry_broad", "") if samples else ""
                is_european = "european" in ancestry.lower() if ancestry else False

                score = sample_number + (1_000_000 if is_european else 0)
                if score > best_score:
                    best_score = score
                    best = entry

            if best is None:
                return None

            perf = best.get("performance_metrics", {})
            sampleset = best.get("sampleset", {})
            samples = sampleset.get("samples", [])
            sample_number = sum(s.get("sample_number", 0) for s in samples)
            ancestry = samples[0].get("ancestry_broad") if samples else None

            def _parse_metrics(items: list[dict]) -> list[EffectSizeInfo]:
                parsed: list[EffectSizeInfo] = []
                for m in items:
                    parsed.append(EffectSizeInfo(
                        name_short=m.get("name_short", ""),
                        name_long=m.get("name_long"),
                        estimate=m["estimate"],
                        ci_lower=m.get("ci_lower"),
                        ci_upper=m.get("ci_upper"),
                        se=m.get("se"),
                    ))
                return parsed

            return PerformanceInfo(
                ppm_id=best.get("id", ""),
                effect_sizes=_parse_metrics(perf.get("effect_sizes", [])),
                class_acc=_parse_metrics(perf.get("class_acc", [])),
                sample_number=sample_number if sample_number > 0 else None,
                ancestry_broad=ancestry,
                phenotyping_reported=best.get("phenotyping_reported"),
                covariates=best.get("covariates"),
            )

    def get_score_download_url(
        self, pgs_id: str, genome_build: str = "GRCh38"
    ) -> str:
        """Get the download URL for a harmonized scoring file.

        Args:
            pgs_id: PGS Catalog score ID
            genome_build: Genome build (GRCh37 or GRCh38)

        Returns:
            HTTPS URL for the scoring file

        Raises:
            ValueError: If no download URL is available for the given build
        """
        with start_action(
            action_type="pgs_catalog:get_score_download_url",
            pgs_id=pgs_id,
            genome_build=genome_build,
        ):
            score = self.get_score(pgs_id)
            url = score.get_download_url(genome_build)
            if url is None:
                raise ValueError(
                    f"No download URL available for {pgs_id} build {genome_build}"
                )
            return url

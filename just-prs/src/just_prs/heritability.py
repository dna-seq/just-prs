"""Heritability data sourcing, caching, and consolidation.

Downloads SNP heritability (h²) estimates from external bulk databases and maps
them to EFO trait IDs used by the PGS Catalog.

Two tiers:
  1. Pan-UK Biobank (highest quality, multi-ancestry, ~7,200 traits × 6 ancestries)
  2. GWAS Atlas (complementary, multi-source GWAS, ~4,700 GWAS)

The EFO mapping relies on the EBISPOT/EFO-UKB-mappings master file for
UKB field / ICD-10 code → EFO translation.
"""

import gzip
import io
import logging
from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.hf import (
    DEFAULT_HF_CATALOG_REPO,
    HF_DATA_PREFIX,
    _configure_hf_timeouts,
    _resolve_token,
)

logger = logging.getLogger(__name__)

PAN_UKBB_PHENOTYPE_MANIFEST_URL = (
    "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/phenotype_manifest.tsv.bgz"
)

EBISPOT_EFO_UKB_MASTER_URL = (
    "https://raw.githubusercontent.com/EBISPOT/EFO-UKB-mappings/master/UK_Biobank_master_file.tsv"
)

# Primary path moved off /public/ on the CTG server; keep fallbacks for older mirrors.
GWAS_ATLAS_DATABASE_URLS: tuple[str, ...] = (
    "https://atlas.ctglab.nl/gwasATLAS_v20191115.txt.gz",
    "https://atlas.ctglab.nl/public/gwasATLAS_v20191115.txt.gz",
)

_HERITABILITY_SCHEMA = {
    "efo_id": pl.Utf8,
    "trait_label": pl.Utf8,
    "h2_observed": pl.Float64,
    "h2_observed_se": pl.Float64,
    "h2_liability": pl.Float64,
    "h2_liability_se": pl.Float64,
    "h2_z": pl.Float64,
    "ancestry": pl.Utf8,
    "method": pl.Utf8,
    "source": pl.Utf8,
    "source_detail": pl.Utf8,
    "confidence": pl.Utf8,
    "n_samples": pl.Int64,
    "trait_type": pl.Utf8,
}

_ANCESTRY_MAP_PAN_UKBB = {
    "AFR": "AFR",
    "AMR": "AMR",
    "CSA": "SAS",
    "EAS": "EAS",
    "EUR": "EUR",
    "MID": "MID",
}


def download_efo_ukb_mappings(
    output_path: Path,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download the EBISPOT EFO-UKB master mapping file.

    Maps UK Biobank ICD-10 codes and field codes to EFO term IDs.

    Args:
        output_path: Destination .parquet file path.
        overwrite: If True, re-download even if parquet exists.

    Returns:
        DataFrame with columns: query, efo_label, efo_id, mapping_type, ukb_code.
    """
    import httpx

    with start_action(action_type="heritability:download_efo_ukb_mappings"):
        if output_path.exists() and not overwrite:
            return pl.read_parquet(output_path)

        resp = httpx.get(EBISPOT_EFO_UKB_MASTER_URL, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()

        df = pl.read_csv(
            io.BytesIO(resp.content),
            separator="\t",
            infer_schema_length=5000,
            null_values=["", "NA", "None"],
            truncate_ragged_lines=True,
            quote_char=None,
        )

        rename_map: dict[str, str] = {}
        for raw, clean in [
            ("ZOOMA QUERY", "query"),
            ("MAPPED_TERM_LABEL", "efo_label"),
            ("MAPPED_TERM_URI", "efo_uri"),
            ("MAPPING_TYPE", "mapping_type"),
            ("ICD10_CODE/SELF_REPORTED_TRAIT_FIELD_CODE", "ukb_code"),
        ]:
            if raw in df.columns:
                rename_map[raw] = clean
        if rename_map:
            df = df.rename(rename_map)

        if "efo_uri" in df.columns:
            df = df.with_columns(
                pl.col("efo_uri")
                .str.replace_all(r",\s*", "|")
                .str.extract(r"([A-Za-z_]+\d+)")
                .alias("efo_id")
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        return df


def _build_icd10_to_efo_map(mappings_df: pl.DataFrame) -> dict[str, str]:
    """Build a dict mapping ICD-10 codes to EFO IDs from the EBISPOT master file."""
    result: dict[str, str] = {}
    if "ukb_code" not in mappings_df.columns or "efo_id" not in mappings_df.columns:
        return result

    for row in mappings_df.filter(
        pl.col("ukb_code").is_not_null() & pl.col("efo_id").is_not_null()
    ).select("ukb_code", "efo_id").iter_rows(named=True):
        code = str(row["ukb_code"]).strip()
        efo = str(row["efo_id"]).strip()
        if code and efo:
            result[code] = efo
    return result


def download_pan_ukbb_heritability(
    output_path: Path,
    mappings_df: pl.DataFrame,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download Pan-UKBB phenotype manifest and extract heritability estimates.

    Uses the topline S-LDSC (EUR) and RHEmc (non-EUR) final h2 estimates
    from the phenotype manifest TSV.

    Args:
        output_path: Destination .parquet file path.
        mappings_df: EFO-UKB mappings DataFrame for trait ID resolution.
        overwrite: If True, re-download even if parquet exists.

    Returns:
        DataFrame with heritability schema columns.
    """
    import httpx

    with start_action(action_type="heritability:download_pan_ukbb"):
        if output_path.exists() and not overwrite:
            return pl.read_parquet(output_path)

        logger.info("Downloading Pan-UKBB phenotype manifest (~15 MB)...")
        resp = httpx.get(PAN_UKBB_PHENOTYPE_MANIFEST_URL, follow_redirects=True, timeout=300.0)
        resp.raise_for_status()

        decompressed = gzip.decompress(resp.content)
        df = pl.read_csv(
            io.BytesIO(decompressed),
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NA", "None", "nan"],
            truncate_ragged_lines=True,
            quote_char=None,
        )

        icd10_to_efo = _build_icd10_to_efo_map(mappings_df)

        pop_codes = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]
        rows: list[dict] = []

        for row in df.iter_rows(named=True):
            trait_type = str(row.get("trait_type", "") or "")
            phenocode = str(row.get("phenocode", "") or "")
            description = str(row.get("description", "") or "")
            pheno_sex = str(row.get("pheno_sex", "") or "")

            efo_id = icd10_to_efo.get(phenocode)
            if efo_id is None and "." in phenocode:
                efo_id = icd10_to_efo.get(phenocode.split(".")[0])
            if efo_id is None:
                continue

            for pop in pop_codes:
                h2_obs_key = f"sldsc_25bin_h2_observed_{pop}"
                h2_lia_key = f"sldsc_25bin_h2_liability_{pop}"
                h2_z_key = f"sldsc_25bin_h2_z_{pop}"
                n_key = f"n_cases_{pop}"

                if pop != "EUR":
                    for prefix in ("rhemc_25bin_50rv", "rhemc_25bin"):
                        alt_obs = f"{prefix}_h2_observed_{pop}"
                        alt_lia = f"{prefix}_h2_liability_{pop}"
                        alt_z = f"{prefix}_h2_z_{pop}"
                        if alt_obs in row and row.get(alt_obs) is not None:
                            h2_obs_key = alt_obs
                            h2_lia_key = alt_lia
                            h2_z_key = alt_z
                            break

                h2_obs = row.get(h2_obs_key)
                if h2_obs is None:
                    continue

                h2_obs_val: float | None = None
                h2_lia_val: float | None = None
                h2_z_val: float | None = None
                n_val: int | None = None

                try:
                    h2_obs_val = float(h2_obs)
                except (TypeError, ValueError):
                    continue

                h2_lia_raw = row.get(h2_lia_key)
                if h2_lia_raw is not None:
                    try:
                        h2_lia_val = float(h2_lia_raw)
                    except (TypeError, ValueError):
                        pass

                h2_z_raw = row.get(h2_z_key)
                if h2_z_raw is not None:
                    try:
                        h2_z_val = float(h2_z_raw)
                    except (TypeError, ValueError):
                        pass

                n_raw = row.get(n_key)
                if n_raw is not None:
                    try:
                        n_val = int(float(n_raw))
                    except (TypeError, ValueError):
                        pass

                method = "S-LDSC" if "sldsc" in h2_obs_key else "RHEmc"
                mapped_ancestry = _ANCESTRY_MAP_PAN_UKBB.get(pop, pop)

                conf = "moderate"
                if h2_z_val is not None and h2_z_val > 4:
                    conf = "high"
                elif h2_z_val is not None and h2_z_val < 2:
                    conf = "low"

                sex_suffix = f", {pheno_sex}" if pheno_sex and pheno_sex != "both_sexes" else ""

                rows.append({
                    "efo_id": efo_id,
                    "trait_label": description or phenocode,
                    "h2_observed": h2_obs_val,
                    "h2_observed_se": None,
                    "h2_liability": h2_lia_val,
                    "h2_liability_se": None,
                    "h2_z": h2_z_val,
                    "ancestry": mapped_ancestry,
                    "method": method,
                    "source": "pan_ukbb",
                    "source_detail": f"Pan-UKBB {pop} {method}{sex_suffix}",
                    "confidence": conf,
                    "n_samples": n_val,
                    "trait_type": trait_type,
                })

        result = pl.DataFrame(rows, schema=_HERITABILITY_SCHEMA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        logger.info("Pan-UKBB heritability: %d rows for %d EFO traits.", result.height, result["efo_id"].n_unique())
        return result


def download_gwas_atlas_heritability(
    output_path: Path,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download GWAS Atlas database and extract SNP heritability estimates.

    Maps traits to EFO IDs via PMID → GWAS Catalog lookup (when available)
    or ICD chapter matching.

    Args:
        output_path: Destination .parquet file path.
        overwrite: If True, re-download even if parquet exists.

    Returns:
        DataFrame with heritability schema columns.
    """
    import httpx

    with start_action(action_type="heritability:download_gwas_atlas"):
        if output_path.exists() and not overwrite:
            return pl.read_parquet(output_path)

        logger.info("Downloading GWAS Atlas database (~25 MB)...")
        resp = None
        for atlas_url in GWAS_ATLAS_DATABASE_URLS:
            r = httpx.get(atlas_url, follow_redirects=True, timeout=300.0)
            if r.status_code >= 400:
                logger.warning(
                    "GWAS Atlas URL returned HTTP %s: %s",
                    r.status_code,
                    atlas_url,
                )
                continue
            r.raise_for_status()
            resp = r
            break

        if resp is None:
            logger.warning(
                "GWAS Atlas: no working download URL — continuing with Pan-UKBB tier only."
            )
            result = pl.DataFrame(schema=_HERITABILITY_SCHEMA)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.write_parquet(output_path)
            return result

        decompressed = gzip.decompress(resp.content)
        df = pl.read_csv(
            io.BytesIO(decompressed),
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NA", "None", "nan", "-"],
            truncate_ragged_lines=True,
            quote_char=None,
        )

        h2_col = None
        for candidate in ("SNPh2", "snph2", "h2"):
            if candidate in df.columns:
                h2_col = candidate
                break

        if h2_col is None:
            logger.warning("GWAS Atlas: no SNP heritability column found.")
            result = pl.DataFrame(schema=_HERITABILITY_SCHEMA)
            result.write_parquet(output_path)
            return result

        filtered = df.filter(pl.col(h2_col).is_not_null())
        if filtered.height == 0:
            result = pl.DataFrame(schema=_HERITABILITY_SCHEMA)
            result.write_parquet(output_path)
            return result

        pop_col = "Population" if "Population" in filtered.columns else None
        trait_col = "uniqTrait" if "uniqTrait" in filtered.columns else "Trait"
        se_col = "SNPh2_se" if "SNPh2_se" in filtered.columns else None
        z_col = "SNPh2_z" if "SNPh2_z" in filtered.columns else None
        n_col = "N" if "N" in filtered.columns else None
        pmid_col = "PMID" if "PMID" in filtered.columns else None

        rows: list[dict] = []
        for row in filtered.iter_rows(named=True):
            h2_val = row.get(h2_col)
            if h2_val is None:
                continue
            try:
                h2_float = float(h2_val)
            except (TypeError, ValueError):
                continue

            trait_name = str(row.get(trait_col, "") or "")
            pop = str(row.get(pop_col, "EUR") or "EUR") if pop_col else "EUR"
            if "EUR" in pop or "UKB" in pop:
                ancestry = "EUR"
            elif "AFR" in pop:
                ancestry = "AFR"
            elif "EAS" in pop:
                ancestry = "EAS"
            elif "AMR" in pop:
                ancestry = "AMR"
            elif "SAS" in pop:
                ancestry = "SAS"
            else:
                ancestry = "EUR"

            se_val: float | None = None
            if se_col and row.get(se_col) is not None:
                try:
                    se_val = float(row[se_col])
                except (TypeError, ValueError):
                    pass

            z_val: float | None = None
            if z_col and row.get(z_col) is not None:
                try:
                    z_val = float(row[z_col])
                except (TypeError, ValueError):
                    pass

            n_val: int | None = None
            if n_col and row.get(n_col) is not None:
                try:
                    n_val = int(float(row[n_col]))
                except (TypeError, ValueError):
                    pass

            pmid = str(row.get(pmid_col, "")) if pmid_col else ""

            conf = "moderate"
            if z_val is not None and z_val > 4:
                conf = "high"
            elif z_val is not None and z_val < 2:
                conf = "low"

            atlas_id = row.get("id", "")
            source_detail = f"GWAS Atlas id={atlas_id}"
            if pmid:
                source_detail += f", PMID:{pmid}"

            rows.append({
                "efo_id": None,
                "trait_label": trait_name,
                "h2_observed": h2_float,
                "h2_observed_se": se_val,
                "h2_liability": None,
                "h2_liability_se": None,
                "h2_z": z_val,
                "ancestry": ancestry,
                "method": "LDSC",
                "source": "gwas_atlas",
                "source_detail": source_detail,
                "confidence": conf,
                "n_samples": n_val,
                "trait_type": None,
            })

        result = pl.DataFrame(rows, schema=_HERITABILITY_SCHEMA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        logger.info("GWAS Atlas heritability: %d rows.", result.height)
        return result


def map_gwas_atlas_to_efo(
    atlas_df: pl.DataFrame,
    gwas_trait_mappings_path: Path | None = None,
) -> pl.DataFrame:
    """Map GWAS Atlas traits to EFO IDs via PMID → GWAS Catalog lookup.

    If a gwas_trait_mappings parquet is available (from just_prs.gwas), uses
    the trait name → EFO mapping from the GWAS Catalog. Falls back to
    returning unmapped rows (efo_id=None) for traits that cannot be resolved.

    Args:
        atlas_df: GWAS Atlas heritability DataFrame.
        gwas_trait_mappings_path: Path to GWAS Catalog trait mappings parquet.

    Returns:
        DataFrame with efo_id populated where mapping succeeded.
    """
    if gwas_trait_mappings_path is None or not gwas_trait_mappings_path.exists():
        return atlas_df

    mappings = pl.read_parquet(gwas_trait_mappings_path)
    if "trait" not in mappings.columns or "efo_id" not in mappings.columns:
        return atlas_df

    trait_to_efo: dict[str, str] = {}
    for row in mappings.filter(
        pl.col("efo_id").is_not_null()
    ).select("trait", "efo_id").unique().iter_rows(named=True):
        t = str(row["trait"]).strip().lower()
        if t:
            trait_to_efo[t] = str(row["efo_id"])

    efo_ids: list[str | None] = []
    for row in atlas_df.iter_rows(named=True):
        existing_efo = row.get("efo_id")
        if existing_efo:
            efo_ids.append(existing_efo)
            continue
        trait = str(row.get("trait_label", "") or "").strip().lower()
        efo_ids.append(trait_to_efo.get(trait))

    return atlas_df.with_columns(
        pl.Series("efo_id", efo_ids, dtype=pl.Utf8)
    )


def build_heritability_table(
    pan_ukbb_df: pl.DataFrame | None = None,
    gwas_atlas_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Merge heritability data from all tiers into a single table.

    Unlike prevalence (one row per EFO ID), heritability retains ALL estimates:
    multiple sources, multiple ancestries, multiple methods per trait. The
    downstream consumer picks the relevant rows based on ancestry and method.

    Args:
        pan_ukbb_df: Pan-UKBB heritability DataFrame (Tier 1).
        gwas_atlas_df: GWAS Atlas heritability DataFrame (Tier 2).

    Returns:
        Combined heritability DataFrame with all estimates.
    """
    with start_action(action_type="heritability:build_table"):
        tiers: list[pl.DataFrame] = []

        if pan_ukbb_df is not None and pan_ukbb_df.height > 0:
            tiers.append(pan_ukbb_df.select(list(_HERITABILITY_SCHEMA.keys())))

        if gwas_atlas_df is not None and gwas_atlas_df.height > 0:
            mapped = gwas_atlas_df.filter(pl.col("efo_id").is_not_null())
            if mapped.height > 0:
                tiers.append(mapped.select(list(_HERITABILITY_SCHEMA.keys())))

        if not tiers:
            return pl.DataFrame(schema=_HERITABILITY_SCHEMA)

        combined = pl.concat(tiers, how="vertical_relaxed")
        logger.info(
            "Combined heritability table: %d rows for %d EFO traits.",
            combined.height,
            combined["efo_id"].n_unique(),
        )
        return combined


def pull_heritability_from_hf(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_CATALOG_REPO,
    token: str | None = None,
) -> Path | None:
    """Download trait_heritability.parquet from the HF catalog repo.

    Args:
        local_dir: Directory to save the downloaded file.
        repo_id: HuggingFace dataset repository ID.
        token: HF API token.

    Returns:
        Path to the downloaded file, or None if not available.
    """
    import shutil

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    resolved_token = _resolve_token(token)
    hf_path = f"{HF_DATA_PREFIX}/metadata/trait_heritability.parquet"

    with start_action(action_type="heritability:pull_from_hf", repo_id=repo_id):
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                repo_type="dataset",
                local_dir=local_dir,
                token=resolved_token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            logger.debug("trait_heritability.parquet not found on HF (%s)", repo_id)
            return None

        target = local_dir / "trait_heritability.parquet"
        hf_cached = Path(path)
        if hf_cached != target:
            shutil.copy2(hf_cached, target)
        return target


def push_heritability_to_hf(
    parquet_path: Path,
    repo_id: str = DEFAULT_HF_CATALOG_REPO,
    token: str | None = None,
) -> None:
    """Upload trait_heritability.parquet to the HF catalog repo.

    Args:
        parquet_path: Local path to the heritability parquet file.
        repo_id: HuggingFace dataset repository ID.
        token: HF API token.
    """
    from huggingface_hub import HfApi

    resolved_token = _resolve_token(token)
    with start_action(action_type="heritability:push_to_hf", repo_id=repo_id):
        _configure_hf_timeouts()
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"{HF_DATA_PREFIX}/metadata/trait_heritability.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )

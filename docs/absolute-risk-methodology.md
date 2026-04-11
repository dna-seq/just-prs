# Absolute Risk Estimation from Polygenic Risk Scores

This document describes how `just-prs` converts a PRS percentile (e.g. "70th percentile for Type 2 Diabetes") into an estimated absolute disease risk (e.g. "~15% lifetime probability").

## Overview

A Polygenic Risk Score tells you where someone falls in the population distribution of genetic risk, but it does **not** directly say what fraction of people at that position actually develop the disease. Translating a relative position into an absolute probability requires two additional pieces of information:

1. **Population prevalence** — what fraction of the general population develops the disease.
2. **Effect size** — how much the PRS discriminates between cases and controls (OR per SD, or AUROC).

The absolute risk estimation pipeline operates in three stages:

```
PRS z-score  +  Prevalence  +  Effect size  →  Absolute risk
    ↑                ↑              ↑
 From PRS       3-tier data    PGS Catalog
 computation     sourcing      performance
                               metrics
```

## Mathematical Models

Two models are implemented, and the system automatically selects the best available one for each score.

### Method 1: OR-per-SD Model

**Used when:** The PGS Catalog provides an Odds Ratio per standard deviation of PRS for the score.

**Priority:** Preferred over the AUC method because OR per SD directly describes how odds change with PRS position.

**The math:**

Given:
- \( z \) — PRS z-score (how many standard deviations above/below the population mean)
- \( \text{OR}_{sd} \) — Odds ratio per standard deviation of PRS
- \( K \) — Population prevalence

The model computes:

\[
\text{baseline\_odds} = \frac{K}{1 - K}
\]

\[
\text{user\_odds} = \text{baseline\_odds} \times \text{OR}_{sd}^{z}
\]

\[
P(\text{disease} \mid z) = \frac{\text{user\_odds}}{1 + \text{user\_odds}}
\]

**Intuition:** At the population mean (\( z = 0 \)), the user's odds equal the baseline odds, and their risk equals the population prevalence. Each standard deviation shift multiplies the odds by the OR. For example, with \( \text{OR}_{sd} = 1.5 \) and \( z = 1 \), the user's odds are 1.5× the baseline.

**Example:** For Type 2 Diabetes with prevalence 11%, OR per SD = 1.5, and a person at the 70th percentile (z ≈ 0.524):
```
baseline_odds = 0.11 / 0.89 = 0.1236
user_odds = 0.1236 × 1.5^0.524 = 0.1236 × 1.226 = 0.1515
P(disease) = 0.1515 / 1.1515 = 13.2%
risk_ratio = 13.2% / 11% = 1.20×
```

### Method 2: AUC-Bivariate Normal Model

**Used when:** The PGS Catalog provides an AUROC for the score but no OR per SD.

**Based on:** The bivariate normal model used by [GenoPred](https://github.com/opain/GenoPred) and described in the PRS literature.

**The math:**

Given:
- \( z \) — PRS z-score
- AUC — Area Under the Receiver Operating Characteristic curve
- \( K \) — Population prevalence

Step 1 — Derive Cohen's d (separation between case and control distributions):

\[
d = \sqrt{2} \cdot \Phi^{-1}(\text{AUC})
\]

where \( \Phi^{-1} \) is the inverse standard normal CDF.

Step 2 — Compute the means of case and control distributions in the combined population:

\[
\mu_{\text{case}} = d \cdot (1 - K)
\]

\[
\mu_{\text{control}} = -d \cdot K
\]

Step 3 — Apply Bayes' theorem to compute P(case | z):

\[
P(\text{case} \mid z) = \frac{K \cdot \phi(z; \mu_{\text{case}}, 1)}{K \cdot \phi(z; \mu_{\text{case}}, 1) + (1-K) \cdot \phi(z; \mu_{\text{control}}, 1)}
\]

where \( \phi(z; \mu, \sigma) \) is the normal probability density function.

**Intuition:** The model assumes PRS follows a normal distribution in both cases and controls, with the same variance but different means. The AUROC tells us how well-separated the two distributions are. For a person at a given z-score, we compute how likely it is that they "came from" the case distribution vs. the control distribution, weighted by the prevalence.

**Example:** For Coronary Artery Disease with prevalence 6%, AUROC = 0.63, and a person at z = 1.0:
```
d = √2 × Φ⁻¹(0.63) = √2 × 0.332 = 0.469
μ_case = 0.469 × 0.94 = 0.441
μ_control = -0.469 × 0.06 = -0.028
P(case | z=1.0) ≈ 8.3%
risk_ratio = 8.3% / 6% = 1.38×
```

### Method Selection Logic

The facade function `estimate_absolute_risk()` selects the method automatically:

1. If OR per SD is available and > 0 → use OR-per-SD method
2. Else if AUROC is available and in (0.5, 1.0) → use AUC-bivariate method
3. Else → return None (insufficient data for estimation)

## Prevalence Data Sourcing

Accurate prevalence data is the key bottleneck for absolute risk estimation. There is no single API that provides population prevalence for all traits in the PGS Catalog. We use a 3-tier strategy with confidence-based prioritization:

### Tier 1: Hand-Curated Seed Data (confidence: high)

A manually curated CSV (`data/trait_prevalence_seed.csv`) with ~50 common traits, sourced from WHO, CDC, and peer-reviewed epidemiological literature.

| Column | Description |
|--------|-------------|
| `efo_id` | Experimental Factor Ontology identifier (e.g. `EFO_0001645`) |
| `trait_label` | Human-readable trait name |
| `prevalence` | Prevalence as a fraction (0-1) |
| `prevalence_type` | `lifetime`, `point`, or `period` |
| `sex` | Sex-specific prevalence (if applicable) |
| `source` | Source identifier (e.g. `WHO`, `CDC`, `PMID:12345678`) |
| `source_detail` | Full citation or URL |

**Why this tier exists:** Epidemiological prevalence from population-based studies is fundamentally different from case-control study fractions. No automated source reliably provides true population prevalence. For the most impactful traits (Type 2 Diabetes, Coronary Artery Disease, Breast Cancer, etc.), hand-curated values from authoritative sources are the gold standard.

### Tier 2: GWAS Catalog Cohort Fractions (confidence: low)

Automated extraction from the [GWAS Catalog](https://www.ebi.ac.uk/gwas/) bulk studies download.

**Process:**
1. Download the full studies TSV from EBI (`studies_new` endpoint)
2. Download the trait mappings TSV (EFO ID ↔ study accession)
3. Parse case and control counts from the free-text `INITIAL SAMPLE SIZE` field using regex (e.g. "1,019 cases, 1,710 controls")
4. Compute case fraction: `n_cases / (n_cases + n_controls)`
5. For each EFO trait, take the study with the largest total sample size

**Caveats:** GWAS case-control ratios do NOT reflect population prevalence — they are designed for statistical power and are typically enriched for cases (~50/50). These fractions are used only when no better data is available and are flagged as `confidence: low`.

### Tier 3: PGS Catalog Evaluation Cohorts (confidence: low)

Last-resort extraction from the PGS Catalog evaluation sample sets.

**Process:**
1. Use `n_cases` and `n_controls` from the best_performance evaluation records
2. Join with scores to map PGS IDs to EFO trait IDs
3. Compute case fraction per EFO trait

**Caveats:** Same ascertainment bias as Tier 2. Evaluation cohorts are not population-representative.

### Tier Merge and Deduplication

For each EFO trait ID, only one prevalence row is kept, selected by confidence priority:

```
high (Tier 1)  >  moderate  >  low (Tiers 2, 3)
```

Within the same confidence level, the first row encountered (Tier 2 before Tier 3) takes priority. The merged result is saved as `trait_prevalence.parquet` and synced to HuggingFace.

## Cross-References via EBI OLS4

For each EFO trait, the pipeline queries the [EBI Ontology Lookup Service (OLS4)](https://www.ebi.ac.uk/ols4/) to retrieve cross-references to other ontologies:

| Cross-reference | Ontology | Use case |
|----------------|----------|----------|
| `xref_mondo` | MONDO | Disease ontology mapping |
| `xref_icd10` | ICD-10 | Clinical coding |
| `xref_snomed` | SNOMED-CT | Clinical terminology |

These cross-references are cached per EFO ID and stored alongside the prevalence data. They enable future enrichment (e.g. looking up prevalence from clinical databases indexed by ICD-10 codes).

## Output: AbsoluteRisk Model

Each absolute risk estimate is wrapped in a Pydantic model with full provenance:

| Field | Type | Description |
|-------|------|-------------|
| `absolute_risk` | float | Estimated disease probability (0-1), e.g. 0.132 = 13.2% |
| `population_prevalence` | float | Baseline prevalence used, e.g. 0.11 = 11% |
| `risk_ratio` | float | User's risk relative to population average, e.g. 1.20× |
| `method` | str | `or_per_sd` or `auc_bivariate` |
| `confidence` | str | Data quality: `high`, `moderate`, or `low` |
| `prevalence_source` | str | Source of prevalence data (e.g. `WHO`, `gwas_catalog_cohort`) |
| `prevalence_type` | str | `lifetime`, `point`, or `cohort` |
| `effect_size_citation` | str | Paper citation for the OR/AUROC value |
| `caveats` | list[str] | Warnings about estimation quality |

## Integration Points

### In PRSCatalog

`PRSCatalog.absolute_risk(pgs_id, z_score, sex=None)` orchestrates the full lookup:

1. Looks up the score's EFO trait ID from `scores.parquet`
2. Retrieves OR and AUROC from `best_performance.parquet`
3. Finds prevalence from `trait_prevalence.parquet` (sex-specific match preferred)
4. Resolves the paper citation from `publications.parquet`
5. Calls `estimate_absolute_risk()` with all parameters

### In the Dagster Pipeline

Two new Dagster assets:
- **`gwas_studies`** (group: `download`) — downloads and parses GWAS Catalog data
- **`trait_prevalence`** (group: `compute`) — merges all tiers into the prevalence table

The `hf_prs_percentiles` asset enriches precomputed reference distributions with absolute risk columns:
- `abs_risk_at_mean` — absolute risk for a person at the population mean z-score (for context)
- `abs_risk_method` — which method was used
- `abs_risk_prevalence` — the prevalence value used

### In the Web UI

The PRS results table shows absolute risk alongside the percentile, including:
- The risk value (e.g. "13.2%")
- Risk ratio vs. population (e.g. "1.20×")
- Prevalence source and confidence
- Paper citation for the effect size
- A disclaimer callout explaining limitations

## Assumptions and Limitations

### Model assumptions

1. **PRS follows a normal distribution** in both cases and controls. This is generally well-supported for large polygenic scores (by the Central Limit Theorem) but may not hold for scores with few large-effect variants.

2. **Constant effect across the distribution.** The OR-per-SD model assumes a log-linear relationship between PRS z-score and disease odds across the full range. In reality, OR may vary at distribution extremes.

3. **Equal variance in cases and controls.** The AUC-bivariate model assumes the PRS has the same variance in both groups. This is approximately true when the PRS explains a small fraction of total disease liability.

4. **Population homogeneity.** Both models assume the prevalence and effect sizes apply to the individual's ancestry group. In practice, PRS effect sizes and prevalence both vary by ancestry. The pipeline uses European-ancestry performance metrics by default (as these dominate PGS Catalog evaluations).

5. **Independence from environmental factors.** The models do not account for gene-environment interactions, epigenetics, or non-genetic risk factors (diet, exercise, smoking, medication, etc.).

### Known biases in prevalence data

- **Tier 2 and 3 prevalence (case-control cohort fractions) are ascertainment-biased.** GWAS and PGS evaluation studies intentionally recruit cases at higher rates than in the general population. These fractions should NOT be interpreted as population prevalence — they are a last-resort proxy flagged with `confidence: low`.

- **Tier 1 prevalence is population-averaged.** It does not account for age-specific or ancestry-specific variation. A 25-year-old and a 65-year-old with the same PRS percentile have very different absolute risks for age-related diseases.

- **Sex-specific prevalence is sparse.** Where available (e.g. breast cancer prevalence for females only), sex-specific matching is used. For most traits, only overall population prevalence is available.

### What these estimates are NOT

- **Not a clinical diagnosis.** Absolute risk from PRS alone is a statistical estimate, not a medical prediction. Clinical risk assessment integrates family history, biomarkers, imaging, and other factors.

- **Not validated for clinical decision-making.** The absolute risk estimates have not been calibrated against prospective cohort outcomes. They should be treated as informational, not actionable.

- **Not ancestry-adjusted.** Most PRS effect sizes in the PGS Catalog come from European-ancestry studies. Risk estimates for non-European individuals may be less accurate.

## Future Improvements

1. **Heritability-based liability threshold model.** Using SNP heritability (h²) and the liability threshold model to derive absolute risk from the proportion of genetic variance explained by the PRS, without needing per-score OR or AUROC.

2. **Age-stratified prevalence.** Integrating age-specific incidence data (e.g. from cancer registries) to provide age-appropriate risk estimates.

3. **Ancestry-specific effect sizes and prevalence.** Using ancestry-matched performance metrics and prevalence data where available.

4. **LLM-assisted prevalence estimation (Tier 4).** For traits not covered by any automated source, using structured LLM queries with literature grounding to estimate prevalence ranges.

## References

- Choi SW, Mak TS, O'Reilly PF. *Tutorial: a guide to performing polygenic risk score analyses.* Nature Protocols. 2020;15:2759–2772. doi:10.1038/s41596-020-0353-1
- Pain O, et al. *Evaluation of polygenic prediction methodology within a reference-standardized framework.* PLoS Genetics. 2021;17(5):e1009021. ([GenoPred](https://github.com/opain/GenoPred))
- Lambert SA, et al. *The Polygenic Score Catalog as an open database for reproducibility and systematic evaluation.* Nature Genetics. 2021;53(4):420–425. doi:10.1038/s41588-021-00783-5
- Sollis E, et al. *The NHGRI-EBI GWAS Catalog: knowledgebase and deposition resource.* Nucleic Acids Research. 2023;51(D1):D1003–D1011. doi:10.1093/nar/gkac1010

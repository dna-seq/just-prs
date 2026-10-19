# Trait regrouping plan

How the ~5,340 PGS Catalog scores should be grouped and named for users, and in what order the
direction curation (`curate-traits`) works through them. Rules and file formats live in
[curation-rules.md](curation-rules.md); the annotation design is in
[score-annotation-plan.md](score-annotation-plan.md).

## Why the current groups do not work

The By Trait view groups either by ontology label (`trait_efo`) or by the paper's reported trait
(`trait_reported`). A survey of the catalog (2026-09-27: 5,337 scores, 696 ontology groups, 1,732
reported names) shows both keys are wrong in predictable ways:

| Problem | Size | Examples |
|---|---|---|
| **Tiny groups** | 239 groups hold one score; 436 hold ≤ 3 | `Epiphyses, Slipped` (slipped disc), `dentures` |
| **Composite labels** (`a\|b`) | 89 groups, 508 scores | `glomerular filtration rate\|creatine level`, `tracheal cancer\|bronchus cancer\|lung cancer` |
| **Sex strata as separate groups** | ~20 groups | `body mass index\|female`, `systolic blood pressure\|male`, `brain aneurysm\|female` |
| **Synonyms split** | 58 reported names span >1 group | prostate cancer vs carcinoma, colorectal cancer vs carcinoma, `asthma` vs `Asthma`, glucose measurement vs blood glucose amount, reticulocyte count vs amount |
| **Umbrella groups** mixing unrelated measures | 18 groups with ≥ 8 reported names | `blood protein amount` (82 different proteins), `neuroimaging measurement\|brain volume` (222 brain regions), `diet measurement` (17 foods), `frailty measurement` (9 factors) |
| **Ontology jargon as the display name** | 224 groups, 1,803 scores end in *measurement / amount / disorder / quality / trait* | `strand of hair color`, `alcohol consumption quality`, `coronary artery disorder`, `bone tissue density` |
| **Opposite directions inside one group** | 82 flip-worded scores in 35 groups; strong negative 1000G pairs in most large groups | see below |

## Direction traps

A group is only safe to average when a higher score means the same thing for every member. These
are the patterns that break it. Each one needs a phenotype with an explicit `sign` against the
concept axis, not a new group.

1. **Markers that run opposite to the concept.** Kidney function: eGFR is `+1`, but serum
   creatinine and cystatin C are `−1` (higher level means worse filtration). Lung function (FEV1,
   FVC) vs COPD. Bone density vs osteoporosis. HDL vs "low HDL".
2. **One-hot categories.** Hair color (black, dark brown, blonde, red), ease of tanning (very
   tanned … never tan, only burn), sun protection (always vs never), chronotype (morning vs evening).
3. **Negatively worded binary outcomes.** "Never smoker", "Childless", "Never eat sugar",
   "Longevity" next to "All-cause mortality" in `age at death`, "looking younger" next to "looking
   older" in facial aging.
4. **Age-at-onset scores.** "Age asthma diagnosed", "Age diabetes diagnosed", "Age of smoking
   initiation", "Age stopped smoking": a higher score is a *later* event, so the sign is `−1` on a
   risk axis.
5. **Variance scores.** Blood-count groups include "… count variance" scores (variability, not
   level). They are a separate phenotype and cannot be aligned with the level.
6. **Factor and PC scores.** Frailty factors, risk-taking principal component, INFLA-score. Direction
   comes from the loadings.
7. **Case–case contrasts.** Thyroid cancer vs benign goiter, ACPA-positive vs ACPA-negative RA.
8. **Drug response.** Statin LDL lowering, beta-blocker survival benefit: out of scope for the
   underlying trait.

Strong negative 1000G correlations inside a single-phenotype group (LDL r = −0.965, CAD −0.955, CKD
−0.971, tanning −0.956, hair color −0.947) are the highest-priority checks: if they hold up, some
users see an inverted percentile.

## Target structure

Group by curated **concept**, the level of the domain → concept → phenotype hierarchy in
`curation/clusters.yaml`. A concept is one question a person would ask ("how is my kidney
function?"); its phenotypes carry the sign. The concept `label` is the display name and must be
plain English, not an ontology label.

Merge rules:

- synonyms and case variants → one concept (prostate cancer + prostate carcinoma);
- composite ontology labels → the concept they measure (the three eGFR groups, creatinine and
  cystatin C → `kidney.kidney_function`);
- sex-stratified groups → the same phenotype with `strata: {sex: …}`;
- subtypes (ER+ breast cancer, ACPA+ RA, late-onset AD) → phenotypes of the parent concept, or a
  sibling concept when a person would read them as a different question;
- disease vs its quantitative measure → sibling concepts in one domain (hypertension vs systolic
  blood pressure; CKD vs kidney function), so the disease view is never averaged with a biomarker.

Split rules:

- umbrella groups split into one concept per measured thing (`blood protein amount` → one concept
  per protein; `diet measurement` → one concept per food);
- bulk single-paper families are handled with a paper `template` and kept out of the default
  consumer view: brain imaging (≈ 330 scores), circulating proteins (82), blood-count variance.

Planned domains (keys are what goes in `clusters.yaml`):

| Domain | Main concepts |
|---|---|
| `cardiovascular` | coronary artery disease, atrial fibrillation, heart failure, stroke, venous thromboembolism, blood pressure, hypertension, ECG intervals, aortic disease, peripheral artery disease |
| `lipids` | LDL, HDL, triglycerides, total cholesterol, lipoprotein(a), apolipoproteins |
| `endocrine` | type 1 / type 2 / any diabetes, glycemic traits, thyroid function and disease, sex hormones |
| `anthropometry` | height, BMI and obesity, fat distribution, lean mass, birth weight |
| `kidney` | kidney function (eGFR / creatinine / cystatin C), chronic kidney disease, albuminuria, urate and gout |
| `liver` | liver enzymes, bilirubin, fatty liver, cirrhosis |
| `blood` | red cells, white cells, platelets, reticulocytes, CRP |
| `respiratory` | lung function, asthma, COPD, fibrosis, sleep apnea |
| `immune` | autoimmune and allergic diseases |
| `cancer` | one concept per site |
| `neurology` / `psychiatry` | dementia and AD, Parkinson's, MS, migraine; depression, bipolar, schizophrenia, ADHD, anxiety, neuroticism |
| `cognition` | cognitive ability, educational attainment |
| `substance_use` | smoking, alcohol use, alcohol use disorder, coffee and tea, cannabis |
| `lifestyle` | physical activity, diet, sun exposure, screen time |
| `sleep` | chronotype, sleep duration, insomnia, daytime napping |
| `pigmentation` | skin tanning, hair color, skin color, balding |
| `reproduction` | menarche, menopause, age at first birth, number of children |
| `eye`, `musculoskeletal`, `digestive`, `skin` | disease and measurement concepts per organ |
| `aging` | lifespan, frailty, facial aging, self-rated health |
| `brain_imaging`, `circulating_proteins` | bulk template domains, hidden by default |

## Search never loses a name

Merging groups makes a score harder to find unless every old name still leads to it. The build
emits `concept_search_terms.parquet` (grain `cluster, phenotype, term`) with:

- the concept and phenotype labels and keys;
- curated `aliases` on concepts and phenotypes in `clusters.yaml` (plain words and opposite
  phrasings: "bad cholesterol", "night owl", "evening person", "mortality");
- every catalog name of every member score: ontology label (composite `a|b` labels split) and ID,
  reported trait, score name.

Searching "creatine level" therefore still lands on *Kidney function*, and "never smoker" on
*Smoking* with the `never_smoker` phenotype (sign −1) highlighted. Opposite phrasings find the
same concept; the result shows which direction the matched phenotype runs.

In the UI people search in the **By Trait grid**, so that is where the terms must work: each
trait-group row gets a searchable "also known as" field built from these terms, and the grid's
filter matches it as well as the displayed name. `PRSCatalog.search()` and just-prs-mcp use the
same table.

## Published vs review

`score_annotations.parquet` (published) holds only resolved rows. Everything a human should look
at goes to `curation_review.parquet`, same columns plus `review_reason`: no direction, an explicit
`review_reason` set by the curator, low-confidence-only evidence, or disagreeing lead variants.
Every row carries `provenance_json` (verbatim quotes with locators) and source links (PGS Catalog
score and publication, PubMed, Europe PMC, DOI) plus the names of the cached raw sources, so a
reviewer agent can reopen the evidence without searching. Both files are pushed together.

## Missing tooling

- `curation/trait_map.yaml`: ontology ID / label (optionally plus a reported-trait regex) →
  `concept.phenotype`, with no direction claim. It gives every score a default group before its
  direction is curated; a per-score annotation overrides it. Its terms feed the search table.
- `PRSCatalog` joins the curation tables; the UI gets a *Curated concept* grouping mode next to
  ontology and reported trait. Scores in a concept without a curated direction are shown but not
  averaged.
- A `curation-review` skill: a reviewer agent that walks `curation_review.parquet`, reopens the
  linked sources, and proposes fixes for a human to accept.

## Order of work

| Batch | Traits | Why first |
|---|---|---|
| 1 | skin tanning + sun exposure, hair color, smoking, chronotype, alcohol, kidney function, lung function, LDL | one-hot and flip wording, opposite-sign markers, strongest negative pairs |
| 2 | CAD / MI / angina, height, BMI and obesity, type 2 diabetes (in rounds), blood pressure | large groups with many negative pairs |
| 3 | blood counts (variance split), reproduction (childless, age-at events), sleep, aging and frailty | age-at, factor, and variance traps |
| 4 | cancers, autoimmune, psychiatric, eye, digestive | mostly catalog-only; synonym merges |
| 5 | brain imaging, circulating proteins, diet | bulk templates |

Each batch runs through the `curate-traits` skill: taxonomy keys are seeded in `clusters.yaml`
first so parallel agents reuse them, then one `trait-curator` agent per trait, then a serial merge.

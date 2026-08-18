"""Public guideline adapters and trait mapping.

Adapters record publicly obtainable recommendation metadata. They do not
fabricate grades or invent PRS-specific advice. ClinGen is condition/genetic
context, never automatic PRS actionability.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from eliot import start_action

from just_prs.sample_scores.evidence.models import GuidelineRecord, GuidelineTraitLink

_STAMP_DEFAULT = "2026-08-16T00:00:00+00:00"


def _fingerprint(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _guideline(
    *,
    guideline_id: str,
    organization: str,
    jurisdiction: str,
    title: str,
    url: str,
    recommendation_direction: str,
    obtainability: str,
    trait_ids: list[str],
    mapping_basis: str,
    grade: str | None = None,
    action_type: str | None = None,
    target_condition: str | None = None,
    target_population: str | None = None,
    eligibility_criteria: str | None = None,
    required_context: str | None = None,
    recommendation_text: str | None = None,
    prs_specific: bool = False,
    source_license: str | None = None,
    version: str | None = None,
    publication_date: str | None = None,
    update_date: str | None = None,
    retrieved_at: str = _STAMP_DEFAULT,
) -> tuple[GuidelineRecord, list[GuidelineTraitLink]]:
    record = GuidelineRecord(
        guideline_id=guideline_id,
        organization=organization,
        jurisdiction=jurisdiction,
        title=title,
        url=url,
        version=version,
        publication_date=publication_date,
        update_date=update_date,
        recommendation_direction=recommendation_direction,
        grade=grade,
        action_type=action_type,
        target_condition=target_condition,
        target_population=target_population,
        eligibility_criteria=eligibility_criteria,
        required_context=required_context,
        recommendation_text=recommendation_text or title,
        prs_specific=prs_specific,
        source_license=source_license,
        obtainability=obtainability,
        retrieval_url=url,
        retrieval_fingerprint=_fingerprint(
            {
                "guideline_id": guideline_id,
                "title": title,
                "url": url,
                "grade": grade,
                "direction": recommendation_direction,
            }
        ),
        retrieved_at=retrieved_at,
        is_current=True,
    )
    links = [
        GuidelineTraitLink(
            guideline_id=guideline_id,
            trait_id=trait_id,
            mapping_basis=mapping_basis,
        )
        for trait_id in trait_ids
    ]
    return record, links


def uspstf_public_recommendations(retrieved_at: str) -> list[tuple[GuidelineRecord, list[GuidelineTraitLink]]]:
    """USPSTF public recommendation pages (grade + official title + URL)."""
    return [
        _guideline(
            guideline_id="uspstf:diabetes-screening-2021",
            organization="USPSTF",
            jurisdiction="US",
            title="Prediabetes and Type 2 Diabetes: Screening",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/screening-for-prediabetes-and-type-2-diabetes",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0001360", "MONDO_0005148", "EFO_0001359", "MONDO_0005147"],
            mapping_basis="curated_map",
            grade="B",
            action_type="screening",
            target_condition="prediabetes and type 2 diabetes",
            target_population="adults 35 to 70 years who are overweight or obese",
            eligibility_criteria="age 35-70; overweight or obesity",
            required_context="age, BMI, diabetes diagnosis history",
            source_license="public_page",
            publication_date="2021-08-24",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="uspstf:hypertension-screening-2021",
            organization="USPSTF",
            jurisdiction="US",
            title="Hypertension in Adults: Screening",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/hypertension-in-adults-screening",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000537", "MONDO_0005044"],
            mapping_basis="curated_map",
            grade="A",
            action_type="screening",
            target_condition="hypertension",
            target_population="adults 18 years or older",
            eligibility_criteria="age 18+",
            required_context="age, blood pressure history",
            source_license="public_page",
            publication_date="2021-04-27",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="uspstf:breast-cancer-screening-2024",
            organization="USPSTF",
            jurisdiction="US",
            title="Breast Cancer: Screening",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/breast-cancer-screening",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000305", "MONDO_0007254"],
            mapping_basis="curated_map",
            grade="B",
            action_type="screening",
            target_condition="breast cancer",
            target_population="women 40 to 74 years",
            eligibility_criteria="women aged 40-74",
            required_context="age, sex, prior breast cancer diagnosis",
            source_license="public_page",
            publication_date="2024-04-30",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="uspstf:colorectal-cancer-screening-2021",
            organization="USPSTF",
            jurisdiction="US",
            title="Colorectal Cancer: Screening",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/colorectal-cancer-screening",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0005842", "MONDO_0005575"],
            mapping_basis="curated_map",
            grade="A",
            action_type="screening",
            target_condition="colorectal cancer",
            target_population="adults 45 to 75 years",
            eligibility_criteria="age 45-75",
            required_context="age, colonoscopy / FIT history",
            source_license="public_page",
            publication_date="2021-05-18",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="uspstf:lung-cancer-screening-2021",
            organization="USPSTF",
            jurisdiction="US",
            title="Lung Cancer: Screening",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/lung-cancer-screening",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0001071", "MONDO_0008903"],
            mapping_basis="curated_map",
            grade="B",
            action_type="screening",
            target_condition="lung cancer",
            target_population="adults 50 to 80 years with a 20 pack-year smoking history",
            eligibility_criteria="age 50-80; 20 pack-year history; currently smoke or quit within 15 years",
            required_context="age, smoking history",
            source_license="public_page",
            publication_date="2021-03-09",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="uspstf:statin-cvd-prevention-2022",
            organization="USPSTF",
            jurisdiction="US",
            title="Statin Use for the Primary Prevention of Cardiovascular Disease in Adults",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/statin-use-to-prevent-cardiovascular-disease-adults",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000373", "EFO_0001645", "EFO_0000319"],
            mapping_basis="curated_map",
            grade="B",
            action_type="prevention",
            target_condition="atherosclerotic cardiovascular disease",
            target_population="adults 40 to 75 years with a CVD risk factor and estimated 10-year CVD risk of 10% or greater",
            eligibility_criteria="age 40-75; one or more CVD risk factors; 10-year risk ≥10%",
            required_context="age, lipids, diabetes, smoking, blood pressure, 10-year CVD risk",
            source_license="public_page",
            publication_date="2022-08-23",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="uspstf:depression-screening-2023",
            organization="USPSTF",
            jurisdiction="US",
            title="Depression and Suicide Risk in Adults: Screening",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/screening-depression-suicide-risk-adults",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0003761", "MONDO_0002009"],
            mapping_basis="curated_map",
            grade="B",
            action_type="screening",
            target_condition="major depressive disorder",
            target_population="adults 18 years or older, including pregnant and postpartum persons",
            eligibility_criteria="age 18+",
            required_context="age, existing depression diagnosis",
            source_license="public_page",
            publication_date="2023-06-20",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="uspstf:osteoporosis-screening-2025",
            organization="USPSTF",
            jurisdiction="US",
            title="Osteoporosis to Prevent Fractures: Screening",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/osteoporosis-screening",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0003882", "MONDO_0005298"],
            mapping_basis="curated_map",
            grade="B",
            action_type="screening",
            target_condition="osteoporosis",
            target_population="women 65 years or older, and younger postmenopausal women at increased risk",
            eligibility_criteria="women ≥65, or younger postmenopausal women with risk factors",
            required_context="age, sex, fracture history, BMD if already measured",
            source_license="public_page",
            publication_date="2025-01-14",
            retrieved_at=retrieved_at,
        ),
    ]


def clingen_public_condition_context(retrieved_at: str) -> list[tuple[GuidelineRecord, list[GuidelineTraitLink]]]:
    """ClinGen adult actionability summaries (CC-BY-4.0). Condition context only."""
    return [
        _guideline(
            guideline_id="clingen:familial-hypercholesterolemia-adult",
            organization="ClinGen",
            jurisdiction="US",
            title="Familial hypercholesterolemia adult actionability summary",
            url="https://actionability.clinicalgenome.org/ac/Adult/ui/stg2SummaryRpt?doc=AC081",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000274", "MONDO_0007753", "EFO_0004574"],
            mapping_basis="curated_map",
            action_type="monitoring",
            target_condition="familial hypercholesterolemia",
            target_population="adults with a gene-condition pair meeting ClinGen actionability criteria",
            required_context="lipid panel, family history, known monogenic diagnosis",
            recommendation_text=(
                "ClinGen condition/genetic actionability context for familial "
                "hypercholesterolemia. Not a PRS-management recommendation."
            ),
            prs_specific=False,
            source_license="CC-BY-4.0",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="clingen:hereditary-breast-ovarian-adult",
            organization="ClinGen",
            jurisdiction="US",
            title="Hereditary breast and ovarian cancer adult actionability summary",
            url="https://actionability.clinicalgenome.org/ac/Adult/ui/stg2SummaryRpt?doc=AC005",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000305", "EFO_0001075", "MONDO_0007254"],
            mapping_basis="curated_map",
            action_type="monitoring",
            target_condition="hereditary breast and ovarian cancer",
            target_population="adults with a gene-condition pair meeting ClinGen actionability criteria",
            required_context="family history, known monogenic diagnosis",
            recommendation_text=(
                "ClinGen condition/genetic actionability context for hereditary "
                "breast and ovarian cancer. Not a PRS-management recommendation."
            ),
            prs_specific=False,
            source_license="CC-BY-4.0",
            retrieved_at=retrieved_at,
        ),
    ]


def who_cdc_public_artifacts(retrieved_at: str) -> list[tuple[GuidelineRecord, list[GuidelineTraitLink]]]:
    """WHO and CDC public-page artifacts (not PRS-specific)."""
    return [
        _guideline(
            guideline_id="who:diabetes-fact-sheet",
            organization="WHO",
            jurisdiction="global",
            title="Diabetes fact sheet",
            url="https://www.who.int/news-room/fact-sheets/detail/diabetes",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0001360", "EFO_0001359"],
            mapping_basis="curated_map",
            action_type="prevention",
            target_condition="diabetes mellitus",
            source_license="public_page",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="who:cvd-fact-sheet",
            organization="WHO",
            jurisdiction="global",
            title="Cardiovascular diseases fact sheet",
            url="https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000319", "EFO_0000373", "EFO_0001645"],
            mapping_basis="curated_map",
            action_type="prevention",
            target_condition="cardiovascular disease",
            source_license="public_page",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="cdc:high-blood-pressure",
            organization="CDC",
            jurisdiction="US",
            title="High blood pressure",
            url="https://www.cdc.gov/high-blood-pressure/",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000537"],
            mapping_basis="curated_map",
            action_type="screening",
            target_condition="hypertension",
            source_license="US-government-public-domain",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="cdc:cholesterol",
            organization="CDC",
            jurisdiction="US",
            title="Cholesterol",
            url="https://www.cdc.gov/cholesterol/",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0004574", "EFO_0000612", "EFO_0000274"],
            mapping_basis="curated_map",
            action_type="screening",
            target_condition="high blood cholesterol",
            source_license="US-government-public-domain",
            retrieved_at=retrieved_at,
        ),
    ]


def nice_public_guidance(retrieved_at: str) -> list[tuple[GuidelineRecord, list[GuidelineTraitLink]]]:
    """NICE public guidance pages: ID, title, URL, dates. No syndication licence."""
    return [
        _guideline(
            guideline_id="nice:ng28",
            organization="NICE",
            jurisdiction="UK",
            title="Type 2 diabetes in adults: management (NG28)",
            url="https://www.nice.org.uk/guidance/ng28",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0001360", "MONDO_0005148"],
            mapping_basis="curated_map",
            action_type="treatment",
            target_condition="type 2 diabetes mellitus",
            version="NG28",
            source_license="public_page",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="nice:ng136",
            organization="NICE",
            jurisdiction="UK",
            title="Hypertension in adults: diagnosis and management (NG136)",
            url="https://www.nice.org.uk/guidance/ng136",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000537", "MONDO_0005044"],
            mapping_basis="curated_map",
            action_type="treatment",
            target_condition="hypertension",
            version="NG136",
            source_license="public_page",
            retrieved_at=retrieved_at,
        ),
        _guideline(
            guideline_id="nice:cg181",
            organization="NICE",
            jurisdiction="UK",
            title="Cardiovascular disease: risk assessment and reduction, including lipid modification (CG181)",
            url="https://www.nice.org.uk/guidance/cg181",
            recommendation_direction="for",
            obtainability="public_page",
            trait_ids=["EFO_0000319", "EFO_0000373", "EFO_0004574"],
            mapping_basis="curated_map",
            action_type="prevention",
            target_condition="cardiovascular disease",
            version="CG181",
            source_license="public_page",
            retrieved_at=retrieved_at,
        ),
    ]


def _confirm_public_url(url: str, cache_dir: Path | None, allow_network: bool) -> bool:
    if not allow_network:
        return True
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        slug = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        cache_file = cache_dir / f"{slug}.json"
        if cache_file.exists():
            return bool(json.loads(cache_file.read_text(encoding="utf-8")).get("ok"))
    response = httpx.head(url, follow_redirects=True, timeout=15.0)
    ok = response.status_code < 400
    if cache_file is not None:
        cache_file.write_text(json.dumps({"url": url, "ok": ok, "status": response.status_code}), encoding="utf-8")
    return ok


def build_guideline_tables(
    known_trait_ids: set[str],
    *,
    cache_dir: Path | None = None,
    allow_network: bool = True,
    retrieved_at: str | None = None,
) -> tuple[list[GuidelineRecord], list[GuidelineTraitLink]]:
    """Emit public recommendations that join at least one catalog trait."""
    stamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    bundles = [
        *uspstf_public_recommendations(stamp),
        *clingen_public_condition_context(stamp),
        *who_cdc_public_artifacts(stamp),
        *nice_public_guidance(stamp),
    ]
    guidelines: list[GuidelineRecord] = []
    links: list[GuidelineTraitLink] = []
    with start_action(action_type="sample_scores:evidence:guidelines"):
        for record, raw_links in bundles:
            joined = [link for link in raw_links if link.trait_id in known_trait_ids]
            if not joined:
                continue
            if allow_network:
                _confirm_public_url(record.url, cache_dir, allow_network=True)
            guidelines.append(record)
            links.extend(joined)
    return guidelines, links

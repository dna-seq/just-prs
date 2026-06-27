from prs_ui.mixin import (
    _concise_trait_label,
    _genome_file_label,
    _merge_prs_results,
    _trait_group_display_label,
    _trait_heritability_summary,
    native_superpopulation_from_ancestry,
    reference_population_codes,
    result_grid_height,
)
from just_prs.viz import build_prs_ai_prompt


def test_trait_group_label_uses_concise_efo_label_with_reported_aliases() -> None:
    rows = [
        {
            "pgs_id": "PGS001232",
            "trait": "Fluid intelligence score",
            "trait_efo": "intelligence",
            "trait_efo_id": "EFO_0004337",
        },
        {
            "pgs_id": "PGS003724",
            "trait": "Intelligence quotient",
            "trait_efo": "intelligence",
            "trait_efo_id": "EFO_0004337",
        },
    ]

    label, reported_traits = _trait_group_display_label(rows)

    assert label == "intelligence"
    assert reported_traits == ["Fluid intelligence score", "Intelligence quotient"]


def test_trait_group_label_keeps_single_reported_trait_simple() -> None:
    rows = [
        {
            "pgs_id": "PGS001232",
            "trait": "Fluid intelligence score",
            "trait_efo": "intelligence",
            "trait_efo_id": "EFO_0004337",
        },
    ]

    label, reported_traits = _trait_group_display_label(rows)

    assert label == "intelligence"
    assert reported_traits == ["Fluid intelligence score"]


def test_trait_group_label_strips_preconcatenated_alias_suffix() -> None:
    rows = [
        {
            "pgs_id": "PGS000001",
            "trait": "Type 1 diabetes (T1D)",
            "trait_efo": (
                "type 1 diabetes mellitus (Type 1 diabetes (T1D); "
                "Type 1 diabetes; Insulin-dependent diabetes mellitus (time-to-event))"
            ),
            "trait_efo_id": "MONDO_0005147",
        }
    ]

    label, reported_traits = _trait_group_display_label(rows)

    assert label == "type 1 diabetes mellitus"
    assert reported_traits == ["Type 1 diabetes (T1D)"]


def test_concise_trait_label_preserves_regular_parenthetical_names() -> None:
    assert _concise_trait_label("body mass index (BMI)") == "body mass index (BMI)"


def test_merge_prs_results_prepends_and_replaces_by_pgs_id() -> None:
    existing = [
        {"pgs_id": "PGS000001", "score": 1},
        {"pgs_id": "PGS000002", "score": 2},
    ]
    new_rows = [
        {"pgs_id": "PGS000003", "score": 3},
        {"pgs_id": "PGS000001", "score": 10},
    ]

    merged = _merge_prs_results(existing, new_rows)

    assert merged == [
        {"pgs_id": "PGS000003", "score": 3},
        {"pgs_id": "PGS000001", "score": 10},
        {"pgs_id": "PGS000002", "score": 2},
    ]


def test_genome_file_label_strips_normalized_suffix() -> None:
    assert _genome_file_label("/tmp/livia.vcf.gz.normalized.parquet") == "livia.vcf.gz"
    assert _genome_file_label("/tmp/anton.parquet") == "anton"


def test_trait_heritability_summary_deduplicates_metrics() -> None:
    rows = [
        {
            "heritability_metrics": [
                {"population": "European", "h2": "0.550", "source": "Pan-UKBB"},
                {"population": "European", "h2": "0.550", "source": "Pan-UKBB"},
            ],
            "heritability_detail": "h² means population-level heritability.",
        },
    ]

    summary, detail, metrics = _trait_heritability_summary(rows)

    assert summary == "European h²=0.550 (Pan-UKBB)"
    assert detail == "h² means population-level heritability."
    assert len(metrics) == 1


def test_individual_result_grid_height_accounts_for_grouped_headers() -> None:
    assert result_grid_height(3, 6, grouped_headers=True) == "262px"
    assert result_grid_height(3, 4) == "222px"


def test_native_superpopulation_mapping_prefers_model_ancestry() -> None:
    assert native_superpopulation_from_ancestry("European", fallback="AFR") == "EUR"
    assert native_superpopulation_from_ancestry("East Asian", fallback="EUR") == "EAS"
    assert native_superpopulation_from_ancestry("Multiple ancestries", fallback="SAS") == "SAS"


def test_reference_population_codes_keep_native_first_and_deduplicate() -> None:
    assert reference_population_codes("EUR", ["AFR", "EUR", "SAS"]) == ["EUR", "AFR", "SAS"]


def test_trait_ai_prompt_prioritizes_risk_and_h2_over_agreement() -> None:
    row = {
        "trait": "type 1 diabetes mellitus",
        "trait_efo_id": "MONDO_0005147",
        "n_models": 34,
        "usable_models": 9,
        "pgs_ids": "PGS001297, PGS004063",
        "genome_file": "livia.vcf.gz",
        "best_pgs_id": "PGS001297",
        "best_model_pctl": 99.4,
        "best_quality": "High",
        "typical_percentile": 80.5,
        "percentile_range": "64.4–99.4",
        "percentile_std": 11.2,
        "consistency": "Wide spread",
        "overall_signal": "Above average",
        "best_absolute_risk": "12.0% (pop. avg: 8.0%)",
        "population_average": "8.0%",
        "risk_vs_average": "1.50x",
        "risk_agreement": "2 methods agree on elevated risk",
        "heritability": "European h²=0.550 (Pan-UKBB)",
        "heritability_detail": "h² means population-level heritability.",
        "high_confidence_models": 9,
        "high_confidence_median": 80.5,
    }

    prompt = build_prs_ai_prompt("trait_summary", row=row, limit=6000)

    assert "Genome/VCF input: livia.vcf.gz" in prompt
    assert "Absolute risk (best model): 12.0% (pop. avg: 8.0%)" in prompt
    assert "Heritability (h²): European h²=0.550 (Pan-UKBB)" in prompt
    assert "Priority: give a quick bottom-line risk interpretation first" in prompt
    assert "2. **Risk in real terms**" in prompt

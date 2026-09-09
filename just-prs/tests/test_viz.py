import urllib.parse

import polars as pl

from just_prs.viz import (
    build_prs_ai_prompt,
    fine_population_html,
    fine_population_label,
    plot_trait_scores,
    trait_report_html,
    wrap_overflow_model_table,
)


def test_plot_trait_scores_accepts_preformatted_absolute_risk() -> None:
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "superpopulation": ["EUR"],
            "mean": [0.0],
            "std": [1.0],
            "trait_reported": ["type 1 diabetes mellitus"],
            "n_variants": [1000],
        },
    )
    user_results = [
        {
            "pgs_id": "PGS000001",
            "score": "0.2",
            "percentile": "58.0",
            "match_rate": "76.0",
            "auroc": "0.712",
            "absolute_risk": "10.0% (pop. avg: 8.3%)",
            "population_prevalence": None,
            "risk_ratio": "1.2",
            "reliable": True,
        },
    ]

    chart = plot_trait_scores(
        "type 1 diabetes mellitus",
        distributions,
        user_results=user_results,
        show_table=True,
    )

    assert chart.to_dict()

    html = trait_report_html(chart, "type 1 diabetes mellitus", user_results)

    assert "PRS Report: type 1 diabetes mellitus" in html
    assert "10.0% (pop. avg: 8.3%)" in html
    assert "Median (selected)" in html
    assert "Median (high quality)" not in html
    assert "Median (all models)" not in html
    assert "<!-- prs-dashboard" in html
    assert "Download PNG" in html
    assert "__PRS_DOWNLOAD_PNG__" in html
    assert "__PRS_VIEW__" in html


def test_plot_trait_scores_has_no_quality_checkbox_bindings() -> None:
    """The High/Moderate/Low/Very-low checkbox row was dropped as visual
    clutter — quality lives in dot colors/tooltips and the report table's
    per-model checkboxes handle visibility. Only modelSelect remains."""
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001", "PGS000002"],
            "superpopulation": ["EUR", "EUR"],
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "trait_reported": ["type 1 diabetes mellitus", "type 1 diabetes mellitus"],
            "n_variants": [1000, 800],
        },
    )
    user_results = [
        {
            "pgs_id": "PGS000001",
            "score": 0.2,
            "percentile": 58.0,
            "z_score": 0.2,
            "pct_EUR": 58.0,
            "match_rate": 76.0,
            "quality_label": "High",
            "reliable": True,
        },
        {
            "pgs_id": "PGS000002",
            "score": -0.4,
            "percentile": 34.0,
            "z_score": -0.4,
            "pct_EUR": 34.0,
            "match_rate": 20.0,
            "quality_label": "Low",
            "reliable": True,
        },
    ]

    spec = plot_trait_scores(
        "type 1 diabetes mellitus",
        distributions,
        user_results=user_results,
        show_table=False,
    ).to_dict()
    params = spec.get("params") or []
    param_names = {param.get("name") for param in params}

    assert "modelSelect" in param_names
    assert not {"showHigh", "showModerate", "showLow", "showVeryLow"} & param_names
    assert "modelScope" not in param_names
    assert "percentileSource" not in param_names


def test_plot_trait_scores_median_line_matches_scoped_card_median() -> None:
    """Regression (screenshot bug): chart said 'Median: 94th' while the card said 83.3.

    The chart's median annotation must be the same scoped median percentile the
    dashboard card shows — never a private upper-middle-element recomputation.
    """
    distributions = pl.DataFrame(
        {
            "pgs_id": [f"PGS00000{i}" for i in range(1, 7)],
            "superpopulation": ["EUR"] * 6,
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "trait_reported": ["intelligence"] * 6,
            "n_variants": [1000] * 6,
        },
    )
    percentiles = [10.0, 20.0, 30.0, 90.0, 95.0, 99.0]  # true median 60, upper-middle 90
    user_results = [
        {
            "pgs_id": f"PGS00000{i + 1}",
            "score": 0.1,
            "percentile": pct,
            "match_rate": 0.9,
            "quality_label": "High",
        }
        for i, pct in enumerate(percentiles)
    ]

    spec = plot_trait_scores(
        "intelligence",
        distributions,
        user_results=user_results,
        show_table=False,
        model_scope="all",
        percentile_source="native",
    ).to_dict()

    import json

    spec_json = json.dumps(spec)
    assert "Median: 60th" in spec_json
    assert "Median: 90th" not in spec_json


def test_plot_trait_scores_selects_by_pgs_ids_across_traits() -> None:
    """PGS-ID selection: only the named scores are plotted, even across traits,
    and the derived title lists the traits of the selected scores."""
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001", "PGS000002", "PGS000003"],
            "superpopulation": ["EUR"] * 3,
            "mean": [0.0] * 3,
            "std": [1.0] * 3,
            "trait_reported": ["Breast cancer", "intelligence", "Breast cancer"],
            "n_variants": [1000, 800, 600],
        },
    )
    user_results = [
        {"pgs_id": "PGS000001", "score": 0.2, "percentile": 58.0, "quality_label": "High"},
        {"pgs_id": "PGS000002", "score": -0.1, "percentile": 40.0, "quality_label": "High"},
    ]

    spec = plot_trait_scores(
        "",
        distributions,
        user_results=user_results,
        pgs_ids=["PGS000001", "PGS000002"],
        show_table=False,
    ).to_dict()

    import json

    spec_json = json.dumps(spec)
    plotted = {p for p in ("PGS000001", "PGS000002", "PGS000003") if p in spec_json}
    assert plotted == {"PGS000001", "PGS000002"}
    assert "Breast cancer, intelligence" in spec_json


def test_plot_trait_scores_pgs_ids_unknown_id_raises() -> None:
    import pytest

    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "superpopulation": ["EUR"],
            "mean": [0.0],
            "std": [1.0],
            "trait_reported": ["Breast cancer"],
            "n_variants": [1000],
        },
    )
    with pytest.raises(ValueError, match="PGS999999"):
        plot_trait_scores("", distributions, pgs_ids=["PGS999999"])


def test_wrap_overflow_model_table_hides_rows_after_ten() -> None:
    header = "<tr><th>PGS ID</th></tr>"
    rows = [f"<tr><td>fixed</td></tr>"] + [f"<tr><td>PGS{i:06d}</td></tr>" for i in range(15)]
    html = wrap_overflow_model_table(header, rows)
    assert html.count("class=\"model-table\"") == 2
    assert "Show 5 more models" in html
    assert html.index("PGS000000") < html.index("class=\"more-models\"")
    assert html.index("class=\"more-models\"") < html.index("PGS000010")
    short = wrap_overflow_model_table(header, rows[:8])
    assert "more-models" not in short
    assert short.count("class=\"model-table\"") == 1


def test_trait_report_per_sample_ancestry_legend_and_subtitle() -> None:
    """Auto-detected per-sample ancestries surface in the sample legend table,
    which replaces the redundant "Samples: … · Ancestry: …" subtitle line."""
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "superpopulation": ["EUR"],
            "mean": [0.0],
            "std": [1.0],
            "trait_reported": ["type 1 diabetes mellitus"],
            "n_variants": [1000],
        },
    )
    multi = {
        "Anton": [{"pgs_id": "PGS000001", "score": 0.2, "percentile": 58.0, "reliable": True}],
        "Livia": [{"pgs_id": "PGS000001", "score": -0.1, "percentile": 40.0, "reliable": True}],
    }
    chart = plot_trait_scores(
        "type 1 diabetes mellitus",
        distributions,
        user_results=multi["Anton"],
        multi_user_results=multi,
        sample_name="Anton, Livia",
    )

    html = trait_report_html(
        chart,
        "type 1 diabetes mellitus",
        multi["Anton"],
        multi_user_results=multi,
        sample_files={
            "Anton": {
                "file": "/g/anton.vcf", "build": "GRCh38",
                "ancestry": "EUR", "ancestry_confidence": 1.0,
                "fine_population": "CEU", "fine_confidence": 0.62,
            },
            "Livia": {"file": "/g/livia.vcf.gz", "build": "GRCh38", "ancestry": "AFR"},
        },
    )

    # Population and sub-population are separate columns; the cohort column
    # only appears when at least one sample has a fine call, with a footnote
    # explaining it is the nearest reference cohort, not a nationality.
    assert "<th>Population</th>" in html
    assert "<th>Closest 1000G Cohort</th>" in html
    assert "a reference point, not a nationality" in html
    # Sub-ancestry resolves the 1000G code to a readable name linked to IGSR,
    # with the official cohort description as a tooltip.
    assert "Northern/Western European (CEU)" in html
    assert "https://www.internationalgenome.org/data-portal/population/CEU" in html
    assert 'title="Utah residents (CEPH) with Northern and Western European ancestry"' in html
    assert "African (AFR)" in html
    # Confidence percentages render next to both the population and the cohort;
    # samples without a confidence value show none (no fake 0%).
    assert ">100%</span>" in html
    assert ">62%</span>" in html
    # The footnote warns that unrepresented populations (e.g. Slavic) map to the
    # nearest cohort with reduced confidence.
    assert "Slavic" in html
    # Unchecking a model in the table also removes its dots from the chart:
    # the recompute script collects the hidden pgs_ids and filters every
    # pgs_id-carrying data row when re-embedding the spec.
    assert "hiddenPgs" in html
    assert "v2.pgs_id" in html
    # Ask AI sits above the number table (and the later footnote blocks) so
    # people can ask before scrolling a model grid.
    assert html.index('class="ai-buttons"') < html.index('class="model-table"')
    assert html.index('class="ai-buttons"') < html.index("Closest 1000G Cohort is the nearest")
    # The legend table IS the header — no duplicated subtitle line.
    assert "Samples:" not in html
    assert '<div class="subtitle">' not in html

    # Without ancestry metadata (old callers) the legend keeps its 3-column shape.
    html_plain = trait_report_html(
        chart,
        "type 1 diabetes mellitus",
        multi["Anton"],
        multi_user_results=multi,
        sample_files={
            "Anton": {"file": "/g/anton.vcf", "build": "GRCh38"},
            "Livia": {"file": "/g/livia.vcf.gz", "build": "GRCh38"},
        },
    )
    assert "<th>Population</th>" not in html_plain
    assert "<th>Closest 1000G Cohort</th>" not in html_plain

    # No sample_files (e.g. --results JSON input) → the classic subtitle returns.
    html_no_files = trait_report_html(
        chart,
        "type 1 diabetes mellitus",
        multi["Anton"],
        multi_user_results=multi,
    )
    assert "Ancestry: European (EUR)" in html_no_files
    assert "Samples:" in html_no_files


def test_fine_population_labels_resolve_1000g_codes() -> None:
    """1000G cohort codes resolve to readable names + IGSR links; codes the
    registry does not know (e.g. HGDP's descriptive names) pass through."""
    assert fine_population_label("IBS") == "Iberian/Spanish (IBS)"
    assert fine_population_label("Russian") == "Russian"  # HGDP-style, as-is

    html = fine_population_html("IBS")
    assert "https://www.internationalgenome.org/data-portal/population/IBS" in html
    assert 'title="Iberian populations in Spain"' in html
    assert "Iberian/Spanish (IBS)" in html
    assert fine_population_html("Russian") == "Russian"  # no fake IGSR link


def test_infer_vcf_ancestry_reads_fingerprint_cache(tmp_path) -> None:
    """A cached inference is returned without re-reading the genome or the
    ancestry model; an UNKNOWN cache entry falls through to None (EUR fallback)."""
    from just_prs.cli import _infer_vcf_ancestry, _load_result_cache, _save_result_cache, _vcf_fingerprint

    vcf = tmp_path / "sample.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    fp = _vcf_fingerprint(vcf)
    cache = _load_result_cache(cache_dir)
    cache[f"__ancestry__GRCh38_{fp}"] = {
        "superpopulation": "EAS",
        "confidence": 0.93,
        "fine_population": "CHB",
        "fine_confidence": 0.71,
    }
    cache[f"__ancestry__GRCh37_{fp}"] = {
        "superpopulation": "UNKNOWN", "confidence": 0.0,
        "fine_population": None, "fine_confidence": None,
    }
    _save_result_cache(cache, cache_dir)

    call = _infer_vcf_ancestry(vcf, "GRCh38", cache_dir)
    assert call is not None
    assert call["superpopulation"] == "EAS"
    assert call["fine_population"] == "CHB"
    assert _infer_vcf_ancestry(vcf, "GRCh37", cache_dir) is None


def test_trait_prompt_prioritizes_sample_risk_and_heritability() -> None:
    user_results = [
        {
            "pgs_id": "PGS000001",
            "score": 0.2,
            "percentile": 82.0,
            "match_rate": 0.76,
            "quality_label": "High",
            "absolute_risk": 0.12,
            "population_prevalence": 0.08,
            "risk_ratio": 1.5,
            "risk_method": "h2-liability",
            "heritability_metrics": [
                {
                    "population": "European",
                    "h2": "0.550",
                    "source": "Pan-UKBB",
                    "risk": "12.0%",
                    "ratio": "1.50x",
                    "confidence": "medium",
                }
            ],
        }
    ]

    prompt = build_prs_ai_prompt(
        "trait_results",
        user_results=user_results,
        trait="type 1 diabetes mellitus",
        ancestry="EUR",
        limit=6000,
        sample_name="livia.vcf.gz",
    )

    assert "Genome/VCF input: livia.vcf.gz" in prompt
    assert "Absolute risk (only model with risk data): 12.0% (pop. avg: 8.0%) [h2-liability]" in prompt
    assert "Heritability (h²): European h²=0.550 (Pan-UKBB)" in prompt
    assert "h²-liability risk estimates: European h²=0.550 (Pan-UKBB): risk 12.0%, 1.50x vs average, medium" in prompt
    assert "For disease traits, discuss absolute risk and risk elevation vs population average" in prompt
    assert "2. **Risk in real terms**" in prompt
    assert "Do not spend the main answer grouping where models agree/disagree" in prompt


def test_trait_prompt_handles_non_disease_traits_without_disease_framing() -> None:
    prompt = build_prs_ai_prompt(
        "trait_results",
        user_results=[
            {
                "pgs_id": "PGS000002",
                "score": 0.1,
                "percentile": 70.0,
                "match_rate": 0.8,
                "quality_label": "High",
                "heritability_metrics": [
                    {
                        "population": "European",
                        "h2": "0.300",
                        "source": "Pan-UKBB",
                    }
                ],
            }
        ],
        trait="intelligence",
        ancestry="EUR",
        limit=6000,
    )

    assert "For non-disease traits (for example longevity, intelligence" in prompt
    assert "do NOT use disease language such as 'lifetime risk', 'screening', or 'diagnosis'" in prompt
    assert "Interpret the percentile as genetic predisposition/tendency" in prompt
    assert "for sport/performance/body/behavior traits" in prompt
    assert "trainability/environment" in prompt
    assert "Do not invent a medical action plan" in prompt
    assert "ethical caveats" not in prompt


def test_trait_report_displays_heritability_in_visible_html() -> None:
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "superpopulation": ["EUR"],
            "mean": [0.0],
            "std": [1.0],
            "trait_reported": ["type 1 diabetes mellitus"],
            "n_variants": [1000],
        },
    )
    user_results = [
        {
            "pgs_id": "PGS000001",
            "score": "0.2",
            "percentile": "58.0",
            "heritability_metrics": [
                {
                    "population": "European",
                    "h2": "0.550",
                    "source": "Pan-UKBB",
                }
            ],
        }
    ]
    chart = plot_trait_scores(
        "type 1 diabetes mellitus",
        distributions,
        user_results=user_results,
    )

    html = trait_report_html(chart, "type 1 diabetes mellitus", user_results)

    assert "Heritability (h²)" in html
    assert "European h²=0.550 (Pan-UKBB)" in html
    assert "<th>h²</th>" in html


def test_trait_report_lists_detected_sample_heritability_only() -> None:
    """The h² card must not dump every Pan-UKBB ancestry for a EUR sample."""
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "superpopulation": ["EUR"],
            "mean": [0.0],
            "std": [1.0],
            "trait_reported": ["intelligence"],
            "n_variants": [1000],
        },
    )
    user_results = [
        {
            "pgs_id": "PGS000001",
            "score": "0.2",
            "percentile": "58.0",
            "match_rate": 0.80,
            "heritability_metrics": [
                {"population": "African", "h2": "0.250", "source": "pan_ukbb"},
                {"population": "East Asian", "h2": "0.190", "source": "pan_ukbb"},
                {"population": "Combined population", "h2": "0.400", "source": "pan_ukbb"},
                {"population": "European", "h2": "0.243", "source": "pan_ukbb"},
            ],
        }
    ]
    chart = plot_trait_scores(
        "intelligence",
        distributions,
        user_results=user_results,
    )
    html = trait_report_html(
        chart,
        "intelligence",
        user_results,
        sample_files={"Anton": {"file": "anton.vcf", "build": "GRCh38", "ancestry": "EUR"}},
    )

    assert "European h²=0.243 (pan_ukbb)" in html
    assert "African" not in html
    assert "East Asian" not in html
    assert "Combined population" not in html

    mixed_html = trait_report_html(
        chart,
        "intelligence",
        user_results,
        sample_files={
            "Anton": {"file": "anton.vcf", "build": "GRCh38", "ancestry": "EUR"},
            "Livia": {"file": "livia.vcf", "build": "GRCh38", "ancestry": "AFR"},
        },
    )
    assert "European h²=0.243 (pan_ukbb)" in mixed_html
    assert "African h²=0.250 (pan_ukbb)" in mixed_html
    assert "East Asian" not in mixed_html
    assert "Combined population" not in mixed_html


def test_trait_report_ai_prompt_contains_sample_name() -> None:
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "superpopulation": ["EUR"],
            "mean": [0.0],
            "std": [1.0],
            "trait_reported": ["type 1 diabetes mellitus"],
            "n_variants": [1000],
        },
    )
    user_results = [{"pgs_id": "PGS000001", "score": "0.2", "percentile": "58.0"}]
    chart = plot_trait_scores(
        "type 1 diabetes mellitus",
        distributions,
        user_results=user_results,
    )

    html = trait_report_html(
        chart,
        "type 1 diabetes mellitus",
        user_results,
        sample_name="anton.vcf",
    )
    encoded_prompt = html.split("https://claude.ai/new?q=", maxsplit=1)[1].split('"', maxsplit=1)[0]
    prompt = urllib.parse.unquote(encoded_prompt)

    assert "Genome/VCF input: anton.vcf" in prompt


def test_multi_sample_prompt_compares_every_sample() -> None:
    multi = {
        "Anton": [
            {
                "pgs_id": "PGS000001",
                "score": 0.2,
                "percentile": 72.0,
                "match_rate": 0.9,
                "quality_label": "High",
            }
        ],
        "Livia": [
            {
                "pgs_id": "PGS000001",
                "score": -0.1,
                "percentile": 41.0,
                "match_rate": 0.88,
                "quality_label": "High",
            }
        ],
    }
    prompt = build_prs_ai_prompt(
        "trait_results",
        user_results=multi["Anton"],
        trait="intelligence",
        ancestry="EUR",
        limit=6000,
        multi_user_results=multi,
        sample_files={
            "Anton": {
                "file": "/g/anton.vcf", "build": "GRCh38",
                "ancestry": "EUR", "fine_population": "CEU",
            },
            "Livia": {"file": "/g/livia.vcf.gz", "build": "GRCh38", "ancestry": "AFR"},
        },
    )

    assert 'across 2 samples' in prompt
    assert "Anton: genome anton.vcf, GRCh38, European (EUR) · Northern/Western European (CEU)" in prompt
    assert "Livia: genome livia.vcf.gz, GRCh38, African (AFR)" in prompt
    assert "== PER-SAMPLE SUMMARY ==" in prompt
    assert "Anton, median percentile 72.0" in prompt
    assert "Livia, median percentile 41.0" in prompt
    assert "== PER-MODEL COMPARISON ==" in prompt
    assert "Anton=72.0 (90%)" in prompt
    assert "Livia=41.0 (88%)" in prompt
    assert "Do not assume the samples are relatives" in prompt
    assert "The per-model details above are for" not in prompt


def test_trait_report_html_multi_sample_prompt_matches_builder() -> None:
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "superpopulation": ["EUR"],
            "mean": [0.0],
            "std": [1.0],
            "trait_reported": ["intelligence"],
            "n_variants": [1000],
        },
    )
    multi = {
        "Anton": [{"pgs_id": "PGS000001", "score": 0.2, "percentile": 72.0, "match_rate": 0.9}],
        "Livia": [{"pgs_id": "PGS000001", "score": -0.1, "percentile": 41.0, "match_rate": 0.88}],
    }
    chart = plot_trait_scores(
        "intelligence",
        distributions,
        user_results=multi["Anton"],
        multi_user_results=multi,
        sample_name="Anton, Livia",
    )
    html = trait_report_html(
        chart,
        "intelligence",
        multi["Anton"],
        multi_user_results=multi,
        sample_files={
            "Anton": {"file": "/g/anton.vcf", "build": "GRCh38", "ancestry": "EUR"},
            "Livia": {"file": "/g/livia.vcf.gz", "build": "GRCh38", "ancestry": "AFR"},
        },
    )
    encoded_prompt = html.split("https://claude.ai/new?q=", maxsplit=1)[1].split('"', maxsplit=1)[0]
    prompt = urllib.parse.unquote(encoded_prompt)
    assert "Anton=72.0" in prompt
    assert "Livia=41.0" in prompt
    assert "https://grok.com/?q=" in html


def test_trait_report_html_shows_catalog_reported_trait() -> None:
    """Mapped ontology 'aging rate' must not hide the opposite reported phenotypes."""
    distributions = pl.DataFrame(
        {
            "pgs_id": ["PGS001071", "PGS001072"],
            "superpopulation": ["EUR", "EUR"],
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "trait_reported": ["aging rate", "aging rate"],
            "n_variants": [6782, 39],
        },
    )
    user_results = [
        {
            "pgs_id": "PGS001071",
            "trait": "aging rate",
            "trait_reported": "Facial aging, looking 'about your age'",
            "score": 0.1,
            "percentile": 60.8,
            "match_rate": 62.6,
            "quality_label": "High",
        },
        {
            "pgs_id": "PGS001072",
            "trait": "aging rate",
            "score": -0.2,
            "percentile": 27.4,
            "match_rate": 61.5,
            "quality_label": "Moderate",
        },
    ]
    user_results[1]["trait"] = "Facial aging, looking 'older than you are'"

    chart = plot_trait_scores("aging rate", distributions, user_results=user_results)
    html = trait_report_html(chart, "aging rate", user_results)

    assert "Reported Trait" in html
    assert "Facial aging, looking 'about your age'" in html
    assert "Facial aging, looking 'older than you are'" in html
    assert "about…" not in html
    assert "older…" not in html

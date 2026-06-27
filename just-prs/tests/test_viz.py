import urllib.parse

import polars as pl

from just_prs.viz import build_prs_ai_prompt, plot_trait_scores, trait_report_html


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
    assert "Absolute risk (best model): 12.0% (pop. avg. 8.0%) [h2-liability]" in prompt
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

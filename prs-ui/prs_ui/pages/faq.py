"""FAQ and project overview for the public PRS UI."""

import reflex as rx


def _faq_item(question: str, answer: rx.Component) -> rx.Component:
    """Native disclosure block for FAQ content."""
    return rx.el.details(
        rx.el.summary(
            rx.hstack(
                rx.icon("circle-help", size=16),
                rx.text(question, size="3", weight="bold"),
                spacing="2",
                align="center",
            ),
            style={"cursor": "pointer", "listStyle": "none"},
        ),
        rx.box(answer, padding_top="10px", padding_left="24px"),
        width="100%",
        padding="14px",
        border="1px solid var(--gray-5)",
        border_radius="10px",
        background="var(--gray-1)",
    )


def faq_panel() -> rx.Component:
    """Public FAQ with citizen-science guidance for PRS computation."""
    return rx.vstack(
        rx.vstack(
            rx.badge("About this project", color_scheme="green", size="2"),
            rx.heading("Polygenic Risk Scores, Without the Catalog Plumbing", size="6"),
            rx.text(
                "This app helps you explore polygenic risk scores from the PGS Catalog "
                "against a genome file. The main workflows are intentionally simple: "
                "pick individual PRS models, or start from a trait and compute all "
                "associated models together.",
                size="3",
                color="gray",
                max_width="900px",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        rx.grid(
            rx.callout(
                "Use By PRS when you already know a PGS ID or want to compare specific models.",
                icon="list-checks",
                color_scheme="blue",
                size="1",
            ),
            rx.callout(
                "Use By Trait when you want to search by disease or phenotype first.",
                icon="layers",
                color_scheme="green",
                size="1",
            ),
            rx.callout(
                "The technical metadata and scoring-file browsers are developer tools, "
                "so they are no longer shown in the main app.",
                icon="wrench",
                color_scheme="gray",
                size="1",
            ),
            columns={"initial": "1", "md": "3"},
            gap="3",
            width="100%",
        ),
        _faq_item(
            "What is a polygenic risk score?",
            rx.text(
                "A polygenic risk score combines many genetic variants into one number "
                "using weights from scientific studies (GWAS). It estimates genetic "
                "predisposition for a trait relative to a reference population. It is "
                "not a diagnosis, and it does not include environment, family history, "
                "age, lifestyle, or clinical measurements.",
                size="2",
                line_height="1.6",
            ),
        ),
        _faq_item(
            "What data do I need?",
            rx.text(
                "The standalone UI works with VCF files. Upload one VCF, confirm or "
                "select the genome build, then choose scores in By PRS or traits in "
                "By Trait. The app normalizes the VCF once and reuses the normalized "
                "genotypes across both tabs.",
                size="2",
                line_height="1.6",
            ),
        ),
        _faq_item(
            "Does a high PRS mean I will get a disease?",
            rx.vstack(
                rx.text(
                    "No. Every complex trait has a heritability — the fraction of variation "
                    "in a population explained by genetics. For most common diseases "
                    "heritability is moderate (roughly 30–60 %). PRS never capture all of "
                    "that heritability: current models typically explain only 5–15 % of "
                    "total trait variance. This gap (missing heritability) exists because "
                    "PRS are built from common variants with individually tiny effects, "
                    "while rare variants, structural variation, and gene–environment "
                    "interactions also contribute.",
                    size="2",
                    line_height="1.6",
                ),
                rx.text(
                    "There is also a causality gap. GWAS variants used in PRS are usually "
                    "not the causal variants — they are tag SNPs in linkage disequilibrium "
                    "(LD) with the true causal loci. A PRS is a statistical proxy, not a "
                    "mechanistic readout. A high PRS shifts your estimated risk upward "
                    "relative to the reference population, but environment, age, sex, "
                    "lifestyle, and clinical biomarkers often matter as much or more.",
                    size="2",
                    line_height="1.6",
                ),
                spacing="2",
            ),
        ),
        _faq_item(
            "Why do several PRS for the same trait give different answers?",
            rx.text(
                "This is normal. The PGS Catalog often has many scores for the same "
                "broad trait, but they may have been trained on different cohorts, "
                "ancestries, phenotype definitions, variant sets, and statistical "
                "methods. Showing only a few 'best' scores would create a false sense "
                "of certainty. Prefer scores with better published evaluation metrics, "
                "higher variant match rates, and agreement with other high-quality "
                "models for the same trait. The trait summary view is designed to help "
                "you see consensus and outliers rather than overreacting to one score.",
                size="2",
                line_height="1.6",
            ),
        ),
        _faq_item(
            "Why is my coverage / match rate so low?",
            rx.vstack(
                rx.text(
                    "When you see a low match rate (e.g. 12 %) it means your genome file "
                    "only contains that fraction of the variants the PRS model expects. "
                    "Common reasons:",
                    size="2",
                    line_height="1.6",
                ),
                rx.text(
                    "Microarray-based consumer tests (23andMe, AncestryDNA, MyHeritage, "
                    "etc.) are not genome sequencing — they use genotyping chips that "
                    "measure a fixed set of ~600–700k pre-selected SNP positions out of "
                    "~3 billion base pairs. A PRS model may need variants that are simply "
                    "not on the chip. Without imputation (statistical inference of missing "
                    "genotypes from reference panels), microarray-derived VCFs will have "
                    "low coverage for many PRS models. Imputation support in just-prs is "
                    "in progress. Some consumer services (e.g. Dante Labs, ITDNA) offer "
                    "real whole-genome sequencing — if yours provides a 30×+ WGS VCF, "
                    "coverage should be substantially better.",
                    size="2",
                    line_height="1.6",
                ),
                rx.text(
                    "Exome sequencing covers only protein-coding regions (~1–2 % of the "
                    "genome), while most GWAS tag SNPs sit in non-coding regions. "
                    "Low-pass WGS (< 4×) may not call low-confidence variants reliably. "
                    "Genome build mismatches (GRCh37 vs GRCh38) will also cause positions "
                    "not to match. Whole-genome sequencing at 30× or higher typically "
                    "covers the vast majority of PRS variants directly.",
                    size="2",
                    line_height="1.6",
                ),
                spacing="2",
            ),
        ),
        _faq_item(
            "How is score quality determined?",
            rx.vstack(
                rx.text(
                    "just-prs computes a synthetic quality score (0–100) from the model's "
                    "published metadata, based on four factors: (1) discrimination metric "
                    "— models are tiered by what performance data is available (AUROC or "
                    "C-index is strongest, then beta, then OR/HR, then nothing); "
                    "(2) cohort size — larger validation cohorts score higher; "
                    "(3) match rate — fraction of scoring variants found in the sample; "
                    "(4) harmonized penalty — 10 % reduction for coordinate-lifted scores.",
                    size="2",
                    line_height="1.6",
                ),
                rx.text(
                    "After PRS computation on real genomes, a combined quality score "
                    "blends the synthetic score (40 %) with practical signals: match-rate "
                    "consistency (25 %), percentile stability (15 %), and absolute-risk "
                    "concordance (20 %). The combined score drives the color-coded quality "
                    "label (High / Normal / Moderate / Low) shown in the results.",
                    size="2",
                    line_height="1.6",
                ),
                spacing="2",
            ),
        ),
        _faq_item(
            "Why does ancestry matter?",
            rx.text(
                "PRS models are strongest in populations similar to the training cohort. "
                "Many published scores come from European-ancestry-heavy cohorts. "
                "Accuracy drops across populations because (1) linkage disequilibrium "
                "patterns vary — a tag SNP that works well in one ancestry group may tag "
                "poorly in another, and (2) allele frequencies and effect sizes differ, "
                "shifting score distributions. Reference percentiles across ancestry "
                "panels do not prove the model works equally well in every population.",
                size="2",
                line_height="1.6",
            ),
        ),
        _faq_item(
            "What does absolute risk mean?",
            rx.text(
                "Absolute risk converts a relative PRS percentile into a real-world "
                "probability using trait prevalence and published performance data. "
                "This is useful for context, but it is only as good as the underlying "
                "prevalence estimate, model quality, and study population. When the "
                "evidence is weak or missing, the result should be treated with caution.",
                size="2",
                line_height="1.6",
            ),
        ),
        _faq_item(
            "Is this medical advice?",
            rx.text(
                "No. PRS results are research and educational outputs. They can be "
                "useful for learning or hypothesis generation, but health decisions "
                "should be made with qualified medical professionals and appropriate "
                "clinical testing.",
                size="2",
                line_height="1.6",
            ),
        ),
        _faq_item(
            "What about privacy?",
            rx.text(
                "Genomic files are sensitive personal data. When you run this app "
                "locally, your VCF stays on your machine except for reference/catalog "
                "data the app downloads. Do not upload private genomes to a server you "
                "do not control.",
                size="2",
                line_height="1.6",
            ),
        ),
        spacing="4",
        width="100%",
        max_width="1100px",
        margin_x="auto",
        padding="16px",
        align="stretch",
    )

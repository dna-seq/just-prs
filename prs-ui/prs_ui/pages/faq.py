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
                "using weights from scientific studies. It estimates genetic tendency "
                "for a trait relative to a reference population. It is not a diagnosis, "
                "and it does not include environment, family history, age, lifestyle, "
                "or clinical measurements.",
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
            "How should I interpret the result?",
            rx.text(
                "Look at the percentile, model quality, variant match rate, and whether "
                "different scores for the same trait agree. A high percentile means your "
                "score is high compared with the selected reference group, not that you "
                "will certainly develop the condition. Low match rates or low model "
                "quality should make the result less trusted.",
                size="2",
                line_height="1.6",
            ),
        ),
        _faq_item(
            "Why are there By PRS and By Trait tabs?",
            rx.text(
                "By PRS is for targeted analysis of known PGS Catalog models. By Trait "
                "starts from a disease or phenotype name and computes related models "
                "together, which is usually easier when you are exploring a health topic "
                "rather than validating one specific score.",
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

from prs_ui.mixin import _trait_group_display_label


def test_trait_group_label_uses_efo_label_with_reported_aliases() -> None:
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

    assert label == "intelligence (Fluid intelligence score; Intelligence quotient)"
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

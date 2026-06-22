"""Regression tests for server-side grid selection preservation.

These cover the bug where changing the sort/filter column in the "By Trait"
selection grid silently dropped the checkbox selection: the trait grid is keyed
by the ``trait`` column, but the inherited re-sync matched rows by ``pgs_id``
(which trait rows do not carry), so it pushed an empty selection model on every
sort and the next click collapsed the selection.

The selection logic is exercised through the pure helpers
``loaded_grid_selection_model`` and ``merge_loaded_grid_selection`` that both
the By-PRS and By-Trait handlers delegate to.
"""

from prs_ui.mixin import (
    loaded_grid_selection_model,
    merge_loaded_grid_selection,
)


def _trait_rows(order: list[str]) -> list[dict]:
    """Trait-grid rows in a given order, keyed by ``trait`` (no ``pgs_id``)."""
    return [
        {"__row_id__": i, "trait": trait, "n_models": 10 - i}
        for i, trait in enumerate(order)
    ]


def _pgs_rows(order: list[str]) -> list[dict]:
    """By-PRS-grid rows in a given order, keyed by ``pgs_id``."""
    return [{"__row_id__": i, "pgs_id": pid} for i, pid in enumerate(order)]


# ---------------------------------------------------------------------------
# Root-cause demonstration: the wrong key field clears the model.
# ---------------------------------------------------------------------------

def test_pgs_keyed_sync_finds_nothing_in_trait_rows() -> None:
    """The original bug: re-syncing trait rows by ``pgs_id`` matches nothing."""
    rows = _trait_rows(["t2d", "asthma", "bmi"])
    # The (buggy) inherited path keyed by pgs_id -> empty model.
    model = loaded_grid_selection_model(rows, ["t2d", "asthma"], "pgs_id")
    assert model == {"type": "include", "ids": []}


def test_trait_keyed_sync_projects_onto_current_order() -> None:
    """The fix: re-syncing by ``trait`` yields the correct positional ids."""
    rows = _trait_rows(["t2d", "asthma", "bmi"])
    model = loaded_grid_selection_model(rows, ["t2d", "asthma"], "trait")
    assert model == {"type": "include", "ids": [0, 1]}

    # After a re-sort the same traits sit at new positions; the model tracks them.
    reordered = _trait_rows(["bmi", "height", "t2d", "asthma"])
    model2 = loaded_grid_selection_model(reordered, ["t2d", "asthma"], "trait")
    assert model2 == {"type": "include", "ids": [2, 3]}


# ---------------------------------------------------------------------------
# Full reproduction: sort, then click a new row -> selection must grow, not reset.
# ---------------------------------------------------------------------------

def test_sort_then_click_preserves_prior_trait_selection() -> None:
    selected = ["t2d", "asthma"]

    # Sort #1 (by n_models): user has t2d@0, asthma@1 selected.
    rows_v1 = _trait_rows(["t2d", "asthma", "bmi", "height"])
    model_v1 = loaded_grid_selection_model(rows_v1, selected, "trait")
    assert model_v1["ids"] == [0, 1]

    # Sort #2 (by EFO): rows reorder. The re-sync re-projects the selection.
    rows_v2 = _trait_rows(["bmi", "t2d", "height", "asthma"])
    model_v2 = loaded_grid_selection_model(rows_v2, selected, "trait")
    assert model_v2["ids"] == [1, 3]  # t2d@1, asthma@3 -- NOT empty

    # User now ticks a new trait (height@2). MUI sends the synced model + height.
    click = {"type": "include", "ids": [1, 3, 2]}
    merged = merge_loaded_grid_selection(rows_v2, selected, click, "trait")
    # Grew to 3 (did not collapse to 1); within-scope items follow row order.
    assert set(merged) == {"t2d", "asthma", "height"}
    assert merged == ["t2d", "height", "asthma"]


def test_regression_without_resync_would_collapse_to_one() -> None:
    """Document the old failure mode: if the model was cleared (empty), the next
    click sends only the clicked row and a *full-replace* handler collapses to 1.

    The merge helper preserves loaded-scope semantics, so even fed the buggy
    empty-model click it does not invent selections -- but it would still drop
    the prior two because they are in the loaded scope yet absent from the event.
    This is exactly why the re-sync (tested above) is required.
    """
    selected = ["t2d", "asthma"]
    rows_v2 = _trait_rows(["bmi", "t2d", "height", "asthma"])
    # Client model was wrongly cleared -> click sends only the new row id (2).
    buggy_click = {"type": "include", "ids": [2]}
    merged = merge_loaded_grid_selection(rows_v2, selected, buggy_click, "trait")
    assert merged == ["height"]  # collapses -- demonstrates why re-sync matters


# ---------------------------------------------------------------------------
# Off-scope preservation: filtered-out / off-page selections survive.
# ---------------------------------------------------------------------------

def test_merge_preserves_out_of_scope_selection() -> None:
    selected = ["t2d", "asthma", "bmi"]
    # Only two of the three selected traits are currently loaded (filtered view).
    loaded = _trait_rows(["t2d", "height", "asthma"])
    # User unchecks asthma (id 2); t2d (id 0) stays checked.
    click = {"type": "include", "ids": [0]}
    merged = merge_loaded_grid_selection(loaded, selected, click, "trait")
    # bmi was off-scope -> preserved (listed first); asthma in-scope, unticked -> removed.
    assert merged == ["bmi", "t2d"]


# ---------------------------------------------------------------------------
# The By-PRS path (the working one) is unaffected by the shared helpers.
# ---------------------------------------------------------------------------

def test_pgs_path_round_trips() -> None:
    rows = _pgs_rows(["PGS000001", "PGS000002", "PGS000003"])
    model = loaded_grid_selection_model(rows, ["PGS000001", "PGS000003"], "pgs_id")
    assert model == {"type": "include", "ids": [0, 2]}

    click = {"type": "include", "ids": [0, 2, 1]}
    merged = merge_loaded_grid_selection(rows, ["PGS000001", "PGS000003"], click, "pgs_id")
    # All three loaded and ticked; within-scope items follow row order.
    assert merged == ["PGS000001", "PGS000002", "PGS000003"]

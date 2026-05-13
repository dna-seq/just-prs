"""Shared MUI DataGrid styling for the PRS UI."""

import reflex as rx


DATA_GRID_SCROLL_CLASS = "prs-ui-data-grid-scroll"


def data_grid_scroll_container(grid: rx.Component) -> rx.Component:
    """Wrap a MUI DataGrid so Firefox and Chromium expose both scrollbars."""
    return rx.box(
        grid,
        class_name=DATA_GRID_SCROLL_CLASS,
        width="100%",
        min_width="0",
    )


def data_grid_scroll_css() -> rx.Component:
    """Global scrollbar CSS for MUI X DataGrid v8 — cross-browser (Firefox + Chrome).

    MUI X DataGrid v8 architecture: the .MuiDataGrid-virtualScroller intentionally
    uses overflow:hidden and is scrolled by MUI's JS.  The actual scrollable elements
    are the overlay scrollbar divs (.MuiDataGrid-scrollbar--vertical /
    .MuiDataGrid-scrollbar--horizontal).  We must NOT hide those divs and must NOT
    override overflow on the virtualScroller — doing so breaks wheel and drag scrolling.

    Strategy: leave MUI's scroll mechanism untouched; style the overlay scrollbar
    elements so they render as slim, visible scrollbars in both Firefox and Chromium.
    """
    return rx.el.style(
        f"""
        .{DATA_GRID_SCROLL_CLASS} {{
            width: 100%;
            min-width: 0;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-root {{
            min-width: 0;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-main {{
            overflow: hidden;
        }}

        /* ---- Vertical overlay scrollbar ---- */
        /* Firefox */
        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--vertical {{
            scrollbar-width: thin;
            scrollbar-color: var(--gray-8) transparent;
        }}

        /* Chromium / Safari */
        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--vertical::-webkit-scrollbar {{
            width: 8px;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--vertical::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--vertical::-webkit-scrollbar-thumb {{
            background: var(--gray-8);
            border-radius: 4px;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--vertical::-webkit-scrollbar-thumb:hover {{
            background: var(--gray-10);
        }}

        /* ---- Horizontal overlay scrollbar ---- */
        /* Firefox */
        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--horizontal {{
            scrollbar-width: thin;
            scrollbar-color: var(--gray-8) transparent;
        }}

        /* Chromium / Safari */
        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--horizontal::-webkit-scrollbar {{
            height: 8px;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--horizontal::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--horizontal::-webkit-scrollbar-thumb {{
            background: var(--gray-8);
            border-radius: 4px;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar--horizontal::-webkit-scrollbar-thumb:hover {{
            background: var(--gray-10);
        }}
        """
    )

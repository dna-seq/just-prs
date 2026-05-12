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
    """Global scrollbar CSS for MUI X DataGrid — cross-browser (Firefox + Chrome).

    Strategy: enable native browser scrollbars on the actual scroll container
    (.MuiDataGrid-virtualScroller) and hide MUI's custom overlay scrollbar divs.
    Firefox never renders -webkit-scrollbar pseudo-elements and may also suppress
    MUI's overlay scrollbars; native scrollbars are always visible in both browsers.
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

        /* Native scrollbars on the actual scroll container — visible in all browsers */
        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-virtualScroller {{
            overflow: auto !important;
            /* Firefox */
            scrollbar-width: thin !important;
            scrollbar-color: var(--gray-8) transparent;
        }}

        /* Chromium / Safari */
        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-virtualScroller::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-virtualScroller::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-virtualScroller::-webkit-scrollbar-thumb {{
            background: var(--gray-8);
            border-radius: 4px;
        }}

        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-virtualScroller::-webkit-scrollbar-thumb:hover {{
            background: var(--gray-10);
        }}

        /* Hide MUI's custom overlay scrollbar elements — native scrollbars above replace them */
        .{DATA_GRID_SCROLL_CLASS} .MuiDataGrid-scrollbar {{
            display: none !important;
        }}
        """
    )

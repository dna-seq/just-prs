"""Reusable Reflex component for rendering Vega-Lite specifications.

Wraps ``react-vega`` v8's ``VegaEmbed`` React component so any Altair chart
can be displayed in a Reflex application by passing its ``.to_dict()`` output
as the ``spec`` prop.  ``VegaEmbed`` auto-detects whether the spec is Vega or
Vega-Lite from its ``$schema`` field, so both formats work out of the box.

Usage::

    from prs_ui.components.vega_chart import VegaLiteChart

    # From an Altair chart object:
    spec = my_altair_chart.to_dict()
    VegaLiteChart.create(spec=spec)

    # Or use the convenience helper:
    from prs_ui.components.vega_chart import vega_chart
    vega_chart(spec=state.my_spec, width="100%")

The component is self-contained: it declares its own npm dependencies
(``react-vega``, ``vega``, ``vega-lite``, ``vega-embed``) and works in
any Reflex app without additional configuration.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


class VegaLiteChart(rx.NoSSRComponent):
    """Render a Vega or Vega-Lite specification using react-vega v8.

    Wraps ``react-vega``'s ``VegaEmbed`` component, which uses ``vega-embed``
    internally.  The spec type (Vega vs Vega-Lite) is auto-detected from the
    ``$schema`` field.

    Props:
        spec: Vega-Lite JSON specification as a Python dict (from
            ``altair_chart.to_dict()``).
        options: ``vega-embed`` ``EmbedOptions`` dict.  Common keys:
            ``actions`` (bool or dict — toolbar visibility),
            ``renderer`` (``"canvas"`` or ``"svg"``),
            ``theme`` (``"dark"`` etc.),
            ``hover`` (bool — enable hover processing).
    """

    library: str = "react-vega@^8.0.0"
    tag: str = "VegaEmbed"
    is_default: bool = False

    lib_dependencies: list[str] = [
        "vega@^6.2.0",
        "vega-lite@^6.4.0",
        "vega-embed@^7.1.0",
    ]

    spec: rx.Var[dict[str, Any]]
    options: rx.Var[dict[str, Any]]


def vega_chart(
    spec: rx.Var[dict[str, Any]] | dict[str, Any] | None = None,
    actions: bool | dict[str, Any] = True,
    renderer: str = "canvas",
    **style_kwargs: Any,
) -> rx.Component:
    """Convenience wrapper for ``VegaLiteChart.create()``.

    Args:
        spec: Vega-Lite spec dict (typically from ``chart.to_dict()``).
            Can be a reactive ``rx.Var`` bound to state.
        actions: Vega-Embed toolbar visibility.
        renderer: ``"canvas"`` or ``"svg"``.
        **style_kwargs: CSS style props forwarded to the component
            (e.g. ``width="100%"``, ``max_width="800px"``).
    """
    options: dict[str, Any] = {"renderer": renderer, "actions": actions}
    props: dict[str, Any] = {"options": options}
    if spec is not None:
        props["spec"] = spec
    return VegaLiteChart.create(**props, **style_kwargs)

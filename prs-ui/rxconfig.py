import reflex as rx
from reflex_components_radix.plugin import RadixThemesPlugin
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="prs_ui",
    disable_plugins=[SitemapPlugin],
    plugins=[
        RadixThemesPlugin(theme=rx.theme(appearance="light", accent_color="blue")),
    ],
)

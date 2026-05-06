import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="prs_ui",
    disable_plugins=[SitemapPlugin],
)

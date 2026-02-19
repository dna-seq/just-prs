from importlib.metadata import metadata

from just_prs.prs_catalog import PRSCatalog

_meta = metadata("just-prs")
__version__: str = _meta["Version"]
__package_name__: str = _meta["Name"]

__all__ = ["PRSCatalog", "__version__", "__package_name__"]

from importlib.metadata import metadata

_meta = metadata("just-prs")
__version__: str = _meta["Version"]
__package_name__: str = _meta["Name"]

"""Question 11 package."""

from .run import run


def _reload_submodules():
    import sys
    import importlib

    submods = [
        ".analysis",
        ".config",
        ".extract",
        ".io_gdf",
        ".posts",
        ".text_utils",
        ".topics",
    ]
    for sm in submods:
        fullname = __name__ + sm
        if fullname in sys.modules:
            importlib.reload(sys.modules[fullname])

_reload_submodules()

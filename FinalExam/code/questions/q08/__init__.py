"""Question 08 package.

Dev-friendly usage (your exact pattern):
    from importlib import reload
    import questions.q08 as q08
    reload(q08)
    print(q08.run())
"""

from __future__ import annotations

from importlib import import_module, reload as _reload

_SUBMODULES = [
    "config",
    "text_utils",
    "io_gdf",
    "posts",
    "analysis",
    "extract",
    "topics",
    "run",
]

def run(*args, **kwargs):
    """Reload Q08 submodules and execute `run.run()`.

    This makes `reload(questions.q08)` + `questions.q08.run()` behave nicely in notebooks.
    """
    pkg = __name__
    mods = []
    for m in _SUBMODULES:
        mod = import_module(pkg + "." + m)
        mods.append(mod)


    for mod in mods:
        _reload(mod)

    return mods[-1].run(*args, **kwargs)

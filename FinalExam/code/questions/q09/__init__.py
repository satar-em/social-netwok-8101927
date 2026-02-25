"""Question 09 package.

Expose a single public entrypoint: run().
"""

from importlib import import_module, reload as _reload


def run(*args, **kwargs):

    mod = import_module(".run", __name__)
    mod = _reload(mod)
    return mod.run(*args, **kwargs)


__all__ = ["run"]

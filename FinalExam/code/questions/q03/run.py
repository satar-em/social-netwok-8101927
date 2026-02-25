from __future__ import annotations

from pathlib import Path
from .analysis import main

def run(
    output_dir: str= "outputs/q03",
    **kwargs
        ):
    """Convenience wrapper for running Question 3."""
    return main(out_dir=Path(output_dir), **kwargs)

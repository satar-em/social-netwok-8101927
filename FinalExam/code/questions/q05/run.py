from __future__ import annotations

from pathlib import Path
from .config import Q5Config
from .pipeline import run_pipeline


def run(
    *,
    output_dir: str | Path = "outputs/q05",
    data_zip: str | Path = "data.zip",
    extract_dir: str | Path = "data_extracted",
    platforms = None,
    seed: int = 42,
    run_infomap: bool = True,
    infomap_preset: str = "fast",
    run_lpa: bool = False,
    export_gexf: bool = False,
):
    """Convenience wrapper used by allQuestion.ipynb."""
    cfg = Q5Config(
        seed=int(seed),
        platforms=list(platforms) if platforms is not None else None,
        run_infomap=bool(run_infomap),
        infomap_preset=str(infomap_preset),
        run_lpa=bool(run_lpa),
        export_gexf=bool(export_gexf),
    )
    return run_pipeline(
        output_dir=output_dir,
        data_zip=data_zip,
        extract_dir=extract_dir,
        config=cfg,
    )

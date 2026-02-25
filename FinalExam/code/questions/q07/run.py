from __future__ import annotations

from pathlib import Path
from .analysis import main as _main

def run(
    out_dir: str | Path = Path("outputs/q07"),
    *,
    zip_path: str | Path = Path("data.zip"),
    extract_dir: str | Path = Path("data_extracted"),
    days=(15, 16, 17, 18, 19),
    make_plots: bool = False,
):
    """Run Q07 and return a small result object you can print.

    Notes:
    - `importlib.reload(questions.q07)` reloads the package module, but not necessarily its submodules.
      If you're editing submodules (analysis.py, extract.py, ...), reload those too.
    """
    return _main(
        out_dir=Path(out_dir),
        zip_path=Path(zip_path),
        extract_dir=Path(extract_dir),
        days=days,
        make_plots=make_plots,
    )

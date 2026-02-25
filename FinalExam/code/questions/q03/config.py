from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

@dataclass(frozen=True)
class Q03Config:
    """Configuration for Question 3 (k-core / k-shell on undirected graphs)."""
    base_dir: Path = Path("extracted_data")
    zip_path: Optional[Path] = Path("data.zip")
    platforms: Tuple[str, ...] = ("twitter", "telegram", "instagram")
    figure_dpi: int = 300

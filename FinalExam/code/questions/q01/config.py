from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


PLATFORMS: List[str] = ["twitter", "telegram", "instagram"]
METRICS: List[str] = ["posts", "views", "followers", "importance"]


@dataclass(frozen=True)
class Q1Paths:

    project_root: Path

    @property
    def default_data_zip(self) -> Path:

        return self.project_root / "data.zip"

    @property
    def default_extract_dir(self) -> Path:
        return self.project_root / "data_unzipped"

    def default_output_dir(self) -> Path:
        return self.project_root / "outputs" / "q01"

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Q5Config:
    seed: int = 42
    platforms: Optional[List[str]] = None
    run_infomap: bool = True
    infomap_preset: str = "fast"
    run_lpa: bool = False
    export_gexf: bool = False

    def normalized(self) -> "Q5Config":
        if self.platforms is None:
            self.platforms = ["twitter", "telegram", "instagram"]
        return self

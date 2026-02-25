"""Configuration for HW Q6 (ported from the notebook).

Usage:
    import questions.q06 as q06
    q06.run()

Override:
    q06.run(output_dir="outputs/q06", data_zip="../../data.zip", extract_dir="data_extracted")
"""

from __future__ import annotations
from datetime import time

SEED: int = 42


DATA_ZIP: str = "data.zip"
EXTRACT_DIR: str = "data_extracted"


OUTPUT_DIR: str = "outputs/q06"

PLATFORMS = ["twitter", "telegram", "instagram"]


CUT_LOCAL_TIME = time(18, 0, 0)
MIN_POSTS_FOR_DECISION: int = 2

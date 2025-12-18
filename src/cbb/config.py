from __future__ import annotations

from pathlib import Path

# Repo root = two levels up from this file: src/cbb/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "espn"
PROC_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

OUTPUTS_DIR = REPO_ROOT / "outputs"

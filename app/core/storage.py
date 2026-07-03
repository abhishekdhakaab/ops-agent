"""Persist completed agent trajectories for later training conversion."""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

TRAJ_PATH = Path("data/trajectories.jsonl")

def log_trajectory(record: Dict[str, Any]) -> None:
    """Append a timestamped trajectory without mutating the caller's record."""
    TRAJ_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "ts": datetime.utcnow().isoformat() + "Z"}

    with TRAJ_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

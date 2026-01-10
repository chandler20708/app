import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_PATH = Path("data/change_log.csv")
LOG_FIELDS = ["timestamp", "page", "action", "parameters"]
RANDOM_SEED = 42


def _ensure_log_file() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def log_action(page: str, action: str, parameters: Optional[Dict[str, Any]] = None) -> None:
    """Append a single action to the shared audit log."""
    _ensure_log_file()
    payload = parameters or {}
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "page": page,
        "action": action,
        "parameters": json.dumps(payload, default=str),
    }
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)


def load_log() -> List[Dict[str, Any]]:
    _ensure_log_file()
    with LOG_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def seed_everything() -> None:
    random.seed(RANDOM_SEED)
    try:
        import numpy as np
        np.random.seed(RANDOM_SEED)
    except Exception:
        pass

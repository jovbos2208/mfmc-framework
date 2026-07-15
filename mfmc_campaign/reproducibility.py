from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def derive_seed(global_seed: int, *parts: Any) -> int:
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    derived = int(digest[:16], 16)
    return int((derived + int(global_seed)) % (2**32 - 1))


def get_run_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        info["git_hash"] = result.stdout.strip()
    except Exception:
        info["git_hash"] = None

    return info


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

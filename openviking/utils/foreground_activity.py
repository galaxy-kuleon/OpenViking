"""Cross-container foreground-user activity lease reader.

OpenWebUI owns lease creation. OpenViking reads it before dequeuing new
background work and between independently schedulable semantic nodes;
already-running model or I/O calls are never interrupted.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

DEFAULT_ACTIVITY_DIR = "/run/kg-user-activity"
DEFAULT_FOREGROUND_RECHECK_SECONDS = 0.5


def activity_snapshot(now: float | None = None) -> dict:
    current = time.time() if now is None else float(now)
    root = Path(os.environ.get("KG_USER_ACTIVITY_DIR", DEFAULT_ACTIVITY_DIR))
    active = []
    invalid = 0
    try:
        leases = list(root.glob("*.json"))
    except OSError:
        leases = []
    for lease in leases:
        try:
            payload = json.loads(lease.read_text(encoding="utf-8"))
            expires_at = float(payload["expires_at"])
            if expires_at > current:
                active.append(
                    {
                        "attempt": str(payload.get("attempt") or lease.stem),
                        "started_at": float(payload.get("started_at") or 0),
                        "expires_at": expires_at,
                    }
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            invalid += 1
    return {
        "active": bool(active),
        "active_leases": len(active),
        "invalid_leases": invalid,
        "background_eligible": not active,
        "activity_dir": str(root),
    }


def foreground_is_active() -> bool:
    return bool(activity_snapshot()["active"])


async def wait_for_background_eligibility() -> None:
    """Yield between background units while a foreground lease is active.

    An already-running model or I/O call is never interrupted.  Callers place
    this boundary before starting the next independently schedulable unit.
    """
    try:
        interval = float(
            os.environ.get(
                "KG_FOREGROUND_RECHECK_SECONDS",
                str(DEFAULT_FOREGROUND_RECHECK_SECONDS),
            )
        )
    except (TypeError, ValueError):
        interval = DEFAULT_FOREGROUND_RECHECK_SECONDS
    interval = max(0.05, interval)
    while foreground_is_active():
        await asyncio.sleep(interval)

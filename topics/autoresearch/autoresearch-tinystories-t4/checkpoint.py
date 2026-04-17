"""
checkpoint.py — Local checkpoint read/write for AutoResearch crash recovery.

Responsibilities:
  - Save agent state to checkpoint.json after each experiment
  - Load state on startup so a crashed run can resume without re-measuring baseline
  - Clear the checkpoint when a run completes normally

Checkpoint file: checkpoint.json (same directory as agent.py)

Schema:
  run_id            : str | null   — Firestore run ID (null if Firestore unavailable)
  current_val_bpb   : float        — best val_bpb achieved so far
  experiment_number : int          — last completed experiment index
  experiment_history: list[dict]   — last 10 experiment records (same format as agent.py)
  saved_at          : str          — ISO timestamp of last save
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.json"


# ── Public API ────────────────────────────────────────────────────────────────

def save(
    run_id: Optional[str],
    current_val_bpb: float,
    experiment_number: int,
    experiment_history: list[dict],
) -> None:
    """
    Write current agent state to checkpoint.json.

    Called after every experiment so a crash can resume from the last
    completed experiment rather than re-measuring baseline from scratch.

    Args:
        run_id            : Firestore run ID (may be None if Firestore is down).
        current_val_bpb   : Best val_bpb achieved so far in this run.
        experiment_number : Index of the last completed experiment.
        experiment_history: List of experiment result dicts (last 10).
    """
    state = {
        "run_id": run_id,
        "current_val_bpb": current_val_bpb,
        "experiment_number": experiment_number,
        "experiment_history": experiment_history,
        "saved_at": _now(),
    }
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load() -> Optional[dict]:
    """
    Load checkpoint state from checkpoint.json if it exists.

    Returns:
        dict with keys: run_id, current_val_bpb, experiment_number,
        experiment_history, saved_at — or None if no checkpoint found.
    """
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        state = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        # Validate required keys are present
        required = {"current_val_bpb", "experiment_number", "experiment_history"}
        if not required.issubset(state):
            print(f"WARNING: checkpoint.json is missing required fields — ignoring.")
            return None
        return state
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Could not read checkpoint.json: {e} — ignoring.")
        return None


def clear() -> None:
    """
    Delete checkpoint.json after a run completes normally.

    Prevents a stale checkpoint from being picked up by the next run.
    """
    try:
        if CHECKPOINT_PATH.exists():
            os.remove(CHECKPOINT_PATH)
    except OSError as e:
        print(f"WARNING: Could not remove checkpoint.json: {e}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

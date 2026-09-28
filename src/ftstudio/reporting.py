"""File-based progress reporting between the training subprocess and the UI.

Every run directory contains:

* ``config.yaml``    - the exact configuration (secrets redacted)
* ``status.json``    - current state, progress and timing (atomically rewritten)
* ``metrics.jsonl``  - one JSON object per logging / evaluation event
* ``train.log``      - full stdout / stderr of the process
* ``STOP``           - created by the UI to request a graceful stop
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

STATUS_FILE = "status.json"
METRICS_FILE = "metrics.jsonl"
LOG_FILE = "train.log"
STOP_FILE = "STOP"
CONFIG_FILE = "config.yaml"

TERMINAL_STATES = {"completed", "failed", "stopped"}


class RunReporter:
    """Written to from the training process."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / STATUS_FILE
        self.metrics_path = self.run_dir / METRICS_FILE
        self.stop_path = self.run_dir / STOP_FILE
        self._status: dict[str, Any] = {
            "state": "starting",
            "message": "",
            "step": 0,
            "max_steps": 0,
            "epoch": 0.0,
            "started_at": time.time(),
            "updated_at": time.time(),
            "pid": os.getpid(),
        }
        self._write_status()

    def update(self, **fields: Any) -> None:
        self._status.update(fields)
        self._status["updated_at"] = time.time()
        self._write_status()

    def state(self, state: str, message: str = "", **fields: Any) -> None:
        self.update(state=state, message=message, **fields)

    def log_metrics(self, record: dict[str, Any]) -> None:
        record = {k: _jsonable(v) for k, v in record.items()}
        record.setdefault("time", time.time())
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def _write_status(self) -> None:
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._status, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.status_path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# Readers (UI side)
# --------------------------------------------------------------------------- #


def read_status(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / STATUS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def read_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / METRICS_FILE
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partially written last line
    return records


def tail_file(path: str | Path, max_bytes: int = 40_000) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    # progress bars use carriage returns; keep only the last state of each line
    lines = [ln.split("\r")[-1] for ln in text.split("\n")]
    return "\n".join(lines)


def request_stop(run_dir: str | Path) -> None:
    (Path(run_dir) / STOP_FILE).touch()


def list_runs(output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / CONFIG_FILE).exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def list_checkpoints(run_dir: str | Path) -> list[Path]:
    """Loadable model directories in a run: the final adapter first, then checkpoints."""
    run_dir = Path(run_dir)
    found = []
    final = run_dir / "final_model"
    if final.exists():
        found.append(final)
    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    return found + ckpts


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

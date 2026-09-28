"""Background job management: training / export jobs run in separate processes.

Running jobs in a subprocess keeps the web UI responsive, isolates crashes
(e.g. CUDA OOM) and guarantees all GPU memory is released when a job ends.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import FinetuneConfig
from .reporting import CONFIG_FILE, LOG_FILE, TERMINAL_STATES, read_status, request_stop


@dataclass
class Job:
    kind: str  # "train" | "export"
    run_dir: Path
    process: subprocess.Popen
    started_at: float = field(default_factory=time.time)

    @property
    def log_path(self) -> Path:
        return self.run_dir / LOG_FILE

    def is_running(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.process.poll()


def make_run_dir(cfg: FinetuneConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.training.output_root) / f"{cfg.training.run_name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _child_env(cfg: FinetuneConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    src_dir = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    # Pass the token through the environment so it is never written to disk.
    if cfg.model.hf_token:
        env["HF_TOKEN"] = cfg.model.hf_token
    if cfg.training.report_to == "wandb":
        env.setdefault("WANDB_PROJECT", "ftstudio")
    return env


class JobManager:
    """Keeps track of one active job at a time (a single GPU is assumed)."""

    def __init__(self):
        self.job: Job | None = None

    @property
    def busy(self) -> bool:
        return self.job is not None and self.job.is_running()

    def _spawn(self, kind: str, run_dir: Path, args: list[str], cfg: FinetuneConfig) -> Job:
        if self.busy:
            raise RuntimeError(f"A {self.job.kind} job is already running: {self.job.run_dir}")
        log_fh = (run_dir / LOG_FILE).open("ab")
        cmd = [sys.executable, "-u", "-m", "ftstudio.cli", *args]
        log_fh.write(f"$ {' '.join(cmd)}\n".encode())
        log_fh.flush()
        kwargs = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True  # own process group -> clean kill
        process = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=_child_env(cfg), cwd=os.getcwd(), **kwargs
        )
        log_fh.close()  # the child keeps its own handle
        self.job = Job(kind=kind, run_dir=run_dir, process=process)
        return self.job

    def start_training(self, cfg: FinetuneConfig) -> Job:
        if self.busy:
            raise RuntimeError(f"A {self.job.kind} job is already running: {self.job.run_dir}")
        run_dir = make_run_dir(cfg)
        cfg_path = cfg.save(run_dir / CONFIG_FILE, redact_secrets=True)
        return self._spawn("train", run_dir, ["train", "--config", str(cfg_path), "--run-dir", str(run_dir)], cfg)

    def start_export(self, cfg: FinetuneConfig, source_dir: str | Path, out_dir: str | Path) -> Job:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg.save(out_dir / "export_config.yaml", redact_secrets=True)
        args = ["export", "--config", str(cfg_path), "--source", str(source_dir), "--out", str(out_dir)]
        return self._spawn("export", out_dir, args, cfg)

    def stop(self, force: bool = False) -> str:
        if not self.busy:
            return "No job is running."
        job = self.job
        if not force:
            request_stop(job.run_dir)
            return "Graceful stop requested - the current step will finish and a checkpoint will be saved."
        self._kill(job)
        return "Job terminated."

    @staticmethod
    def _kill(job: Job) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
            else:
                job.process.terminate()
            job.process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(job.process.pid), signal.SIGKILL)
                else:
                    job.process.kill()
            except ProcessLookupError:
                pass

    def state(self) -> dict:
        """Current job status merged from the process and ``status.json``."""
        if self.job is None:
            return {"state": "idle"}
        status = read_status(self.job.run_dir)
        rc = self.job.returncode
        if rc is not None and status.get("state") not in TERMINAL_STATES:
            # Process died without reporting (e.g. killed, segfault, import error)
            status["state"] = "completed" if rc == 0 else "failed"
            if rc != 0:
                status["message"] = status.get("message") or f"Process exited with code {rc}. See the log."
        status["kind"] = self.job.kind
        status["run_dir"] = str(self.job.run_dir)
        status["running"] = rc is None
        return status

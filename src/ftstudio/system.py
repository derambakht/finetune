"""Hardware / environment introspection for the UI (never imports unsloth)."""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def gpu_info() -> list[dict]:
    """GPU list via nvidia-smi (cheap, no CUDA context needed)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, used, util, temp = parts[:6]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "memory_total_gb": round(float(total) / 1024, 1),
                "memory_used_gb": round(float(used) / 1024, 1),
                "utilization": _num(util),
                "temperature": _num(temp),
            }
        )
    return gpus


def _num(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def system_markdown() -> str:
    lines = [f"**Python** {platform.python_version()} · **OS** {platform.system()} {platform.release()}"]
    try:
        import psutil

        vm = psutil.virtual_memory()
        lines.append(f"**RAM** {vm.used / 1024**3:.1f} / {vm.total / 1024**3:.1f} GB · **CPU cores** {psutil.cpu_count()}")
    except ImportError:
        pass

    gpus = gpu_info()
    if gpus:
        for g in gpus:
            lines.append(
                f"**GPU {g['index']}** {g['name']} · {g['memory_used_gb']} / {g['memory_total_gb']} GB"
                + (f" · {g['utilization']:.0f}% util" if g["utilization"] is not None else "")
                + (f" · {g['temperature']:.0f}°C" if g["temperature"] is not None else "")
            )
    else:
        lines.append("⚠️ **No NVIDIA GPU detected** - training requires a CUDA GPU (the UI still works for configuring).")

    pkgs = ["unsloth", "torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "gradio"]
    versions = [f"`{p}` {package_version(p) or '❌ not installed'}" for p in pkgs]
    lines.append(" · ".join(versions))
    return "  \n".join(lines)


def total_gpu_memory_gb() -> float | None:
    gpus = gpu_info()
    return gpus[0]["memory_total_gb"] if gpus else None

"""Shared helpers for the training / evaluation / export scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root
DATA_DIR = ROOT / "data" / "gas_grouped"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "ml" / "runs"


def runtime_data_yaml() -> str:
    """Write a copy of data.yaml with an ABSOLUTE dataset path and return its location.

    Ultralytics resolves a relative `path:` against its own settings folder, which differs per
    machine. An absolute path makes training work from any folder on Windows 11 / Linux.
    """
    src = DATA_DIR / "data.yaml"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found - copy the grouped dataset to data/gas_grouped first")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "_data_abs.yaml"
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("path:"):
            line = f"path: {DATA_DIR.as_posix()}"
        lines.append(line)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)

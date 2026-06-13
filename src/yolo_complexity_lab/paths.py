from __future__ import annotations

from pathlib import Path


def find_workspace_root(start: Path | None = None) -> Path:
    """Find /home/alejandro/OpenCode-style root by walking up to outputs/."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "outputs").exists() and (candidate / ".projects").exists():
            return candidate
    return Path.cwd().resolve()


def default_export_dir() -> Path:
    root = find_workspace_root()
    return root / "outputs" / "university" / "complejidad-computacional" / "PROYECTO001_YOLO_COMPLEXITY_LAB_Alejandro_Padilla" / "results"

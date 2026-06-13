from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .paths import default_export_dir


def write_results_csv(df, export_dir: Path | None = None) -> Path:
    export_dir = export_dir or default_export_dir()
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = export_dir / f"benchmark_yolo_complexity_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path


def write_plot_html(fig, name: str, export_dir: Path | None = None) -> Path:
    export_dir = export_dir or default_export_dir()
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")
    path = export_dir / f"{safe}_{timestamp}.html"
    fig.write_html(path)
    return path

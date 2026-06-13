from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
import zipfile

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


def build_plot_html_zip(figures: list[tuple[str, object]]) -> bytes:
    """Return a ZIP containing one standalone-ish HTML file per Plotly figure."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, fig in figures:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_") or "grafico"
            html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            archive.writestr(f"{safe}.html", html)
    return buffer.getvalue()

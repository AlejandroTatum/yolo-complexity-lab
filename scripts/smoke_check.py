from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from yolo_complexity_lab.catalog import MODEL_CATALOG, catalog_rows
from yolo_complexity_lab.paths import default_export_dir, find_workspace_root
from yolo_complexity_lab.system_info import system_info_dict


def main() -> int:
    assert len(MODEL_CATALOG) == 5, "Debe haber cinco modelos planificados, incluyendo la demo local opcional."
    required = {"yolov8_gestures", "yolo11n", "yolo11s", "ssdlite_mobilenet_v3", "fasterrcnn_mobilenet_fpn"}
    assert required.issubset(MODEL_CATALOG), "Catálogo incompleto."
    for row in catalog_rows():
        assert "O(" in row["Big-O inferencia"], row
        assert "O(" in row["Big-O postproceso"], row
    root = find_workspace_root(ROOT)
    export_dir = default_export_dir()
    assert root.exists(), root
    assert (ROOT / "assets" / "demo_person_dog_fruit.jpg").exists(), "Falta imagen demo persona/perro/fruta."
    assert "PROYECTO001_YOLO_COMPLEXITY_LAB_Alejandro_Padilla" in str(export_dir)
    info = system_info_dict()
    assert "python" in info
    print("Smoke check OK")
    print(f"Workspace root: {root}")
    print(f"Export dir: {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import streamlit as st

from yolo_complexity_lab.benchmark import BenchmarkConfig, DETECTION_ORANGE_BGR, benchmark_model, run_frame
from yolo_complexity_lab.catalog import MODEL_CATALOG, catalog_rows
from yolo_complexity_lab.environment import detect_capabilities
from yolo_complexity_lab.exporting import write_results_csv
from yolo_complexity_lab.loaders import load_model
from yolo_complexity_lab.paths import default_export_dir
from yolo_complexity_lab.sources import (
    frames_from_webcam,
    read_image_file,
    repeat_frame,
    sample_coco_frame,
)
from yolo_complexity_lab.system_info import system_info_dict

CAPABILITIES = detect_capabilities(ROOT, os.environ)

IMGZ = 416
MODEL_COLOR = {"YOLO11n": "#FF6600", "SSDlite": "#10b981", "Faster R-CNN": "#eab308"}
LABEL_ES = {"person": "persona", "dog": "perro", "banana": "banana", "horse": "caballo"}
# Detecciones reales sobre la imagen demo (mismo pipeline: 416 px, conf 0,25, IoU 0,45)
DEMO_DETECTIONS = {
    "YOLO11n": [
        {"box": [0.3167, 0.063, 0.5009, 0.999], "label": "person", "conf": 0.62},
        {"box": [0.0015, 0.2792, 0.4985, 0.9982], "label": "person", "conf": 0.43},
    ],
    "SSDlite": [
        {"box": [0.0172, 0.2515, 0.4811, 0.9883], "label": "person", "conf": 0.85},
        {"box": [0.7371, 0.3559, 0.9946, 0.9869], "label": "dog", "conf": 0.78},
        {"box": [0.3178, 0.053, 0.6328, 0.9794], "label": "person", "conf": 0.68},
    ],
    "Faster R-CNN": [
        {"box": [0.0255, 0.23, 0.5073, 0.9896], "label": "person", "conf": 0.87},
        {"box": [0.2601, 0.0418, 0.5219, 0.977], "label": "person", "conf": 0.41},
        {"box": [0.5766, 0.3592, 0.9916, 0.9688], "label": "banana", "conf": 0.37},
    ],
}

st.set_page_config(
    page_title="YOLO Complexity Lab",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOURCE_HELP_ALL = {
    "Demo persona/perro/fruta": "Usa una imagen local con persona, perro y banana para comparar reconocimiento y falsos positivos.",
    "Subir imagen": "Repite tu propia imagen para medir latencia sin depender de un video.",
    "Webcam local (OpenCV)": "Captura frames de la cámara local para la demo en vivo; depende de la cámara y la luz.",
}

SOURCE_HELP = {
    key: value
    for key, value in SOURCE_HELP_ALL.items()
    if CAPABILITIES.webcam or key != "Webcam local (OpenCV)"
}

# Rutas de comparación: en la nube la webcam no está disponible, así que se ocultan
PRESET_MODELS_ALL = {
    "YOLO en vivo": ["yolo11n"],
    "Comparación CNN vs YOLO": [
        "fasterrcnn_mobilenet_fpn",
        "ssdlite_mobilenet_v3",
        "yolo11n",
    ],
    "Pesos personalizados": ["yolov8_gestures"],
}

PRESET_HELP_ALL = {
    "YOLO en vivo": "Corre YOLO11n en tiempo real con la webcam local.",
    "Comparación CNN vs YOLO": "Compara CNN de dos etapas y una etapa contra YOLO en tiempo y complejidad.",
}

PRESET_HELP_ALL["Pesos personalizados"] = "Corre los pesos opcionales best.pt de la raíz cuando estén disponibles."
PRESET_MODELS = {
    key: value
    for key, value in PRESET_MODELS_ALL.items()
    if (CAPABILITIES.streaming or key != "YOLO en vivo")
    and (CAPABILITIES.custom_weights or key != "Pesos personalizados")
}
PRESET_HELP = {
    key: value
    for key, value in PRESET_HELP_ALL.items()
    if key in PRESET_MODELS
}

DEVICE_HELP = {
    "auto": "Usa la GPU cuando PyTorch detecta CUDA; si no, usa la CPU.",
    "cpu": "Fuerza la ejecución en CPU. Más comparable entre máquinas, pero más lento.",
    "cuda:0": "Fuerza la primera GPU NVIDIA. Si no está disponible, el loader vuelve a la CPU.",
}
DEVICE_OPTIONS = CAPABILITIES.device_options

NAVIGATION_TABS = ("Resumen", "Benchmark", "Acerca de")
AUTHOR_NAME = "Alejandro Padilla"

METRIC_EXPLANATIONS = {
    "Latencia": "Tiempo que tarda el modelo en procesar un frame. Menor es mejor.",
    "FPS": "Frames por segundo efectivos. Mayor es mejor para uso en tiempo real.",
    "GFLOPs": "Operaciones aproximadas por frame (1 MAC ≈ 2 FLOPs), usado como proxy de complejidad.",
    "Parámetros": "Cantidad de pesos aprendidos; afecta tamaño, memoria y capacidad.",
    "Big-O": "Describe cómo crece el costo con la resolución, capas, canales o cajas candidatas."
}


@st.cache_resource(show_spinner=False)
def cached_load_model(spec_key: str, device: str):
    return load_model(spec_key, device)


def inject_css() -> None:
    """Solo lo que el tema nativo (.streamlit/config.toml) no resuelve.

    El tema tailwind trae paleta, tipografía Inter, bordes, radios, sidebar y
    estilos de widgets; acá quedan los componentes propios del lab y el hack
    para conservar el control de reabrir la sidebar colapsada.
    """
    st.markdown(
        """
<style>
:root {
  --ink: #0f172a;
  --muted: #64748b;
  --hair: #e2e8f0;
  --accent: #2563eb;
}

::selection { background: rgba(37, 99, 235, 0.15); }

/* Header/toolbar: conservar solo el control para reabrir la sidebar. */
[data-testid="stHeader"] { background: transparent; pointer-events: none; }
[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] { pointer-events: auto; }
[data-testid="stDecoration"], .stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: flex !important; }
[data-testid="stToolbar"] button[data-testid="stBaseButton-header"],
[data-testid="stToolbar"] button[data-testid="stMainMenuButton"] { display: none !important; }
[data-testid="stToolbar"] [data-testid="stExpandSidebarButton"] { pointer-events: auto; }

.side-rule { border: none; border-top: 1px solid var(--hair); margin: 0.85rem 0 !important; }
.side-label {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.09em; color: var(--muted); margin-bottom: 0.4rem;
}
.env-note { background: #f1f5f9; border-radius: 8px; padding: 0.6rem 0.7rem; font-size: 0.84rem; line-height: 1.5; }
.env-note .ok { color: #059669; font-weight: 600; }
.env-note .off { color: var(--muted); font-weight: 600; }

.page-head { border-bottom: 1px solid var(--hair); padding-bottom: 0.85rem; }
.brand {
  font-size: 0.78rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: 0.12em; color: var(--muted); display: inline-block;
}
.brand::after {
  content: ""; display: block; width: 56px; height: 3px; border-radius: 2px;
  margin-top: 0.3rem; background: linear-gradient(90deg, #FF6600, #3b82f6);
}
.commercial-head {
  font-size: clamp(1.7rem, 3.4vw, 2.5rem); font-weight: 800;
  letter-spacing: -0.03em; line-height: 1.12; color: var(--ink);
  margin: 0.5rem 0 0.45rem 0; max-width: 950px;
}
.commercial-head .hl { color: #FF6600; }
.app-context { color: var(--muted); font-size: 0.95rem; }
.run-note { color: var(--muted); font-size: 0.88rem; margin-top: 0.65rem; }

.section-title { color: var(--ink); font-size: 1.1rem; font-weight: 650; margin: 1.7rem 0 0.55rem 0; }
.chart-label { font-size: 0.9rem; font-weight: 600; color: var(--ink); margin-bottom: 0.15rem; }

/* Cards del Resumen y Acerca de */
.glass-card { border: 1px solid var(--hair); border-radius: 10px; background: #ffffff; padding: 0.75rem 0.9rem; height: 100%; }
.glass-card h3 { margin: 0 0 0.3rem 0; font-size: 0.95rem; color: var(--ink); }
.glass-card p, .glass-card li { color: var(--muted); font-size: 0.88rem; line-height: 1.45; margin: 0; }
.card-accent-blue { border-top: 3px solid #3b82f6; }
.card-accent-green { border-top: 3px solid #10b981; }
.card-accent-amber { border-top: 3px solid #eab308; }
.card-accent-orange { border-top: 3px solid #FF6600; }
.card-accent-violet { border-top: 3px solid #7c3aed; }
.small-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); display: block; margin-bottom: 0.35rem; }

.arch-strip { display: flex; align-items: stretch; gap: 0.6rem; }
.arch-node { flex: 1; border: 1px solid var(--hair); border-radius: 10px; background: #ffffff; padding: 0.7rem 0.85rem; }
.arch-node h4 { margin: 0 0 0.25rem 0; font-size: 0.92rem; color: var(--ink); }
.arch-node p { margin: 0; color: var(--muted); font-size: 0.85rem; line-height: 1.45; }
.arch-arrow { align-self: center; color: var(--accent); font-weight: 800; font-size: 1.1rem; }

/* Chips de modelo: el seleccionado se llena con su color de familia */
[role="radiogroup"][aria-label="Modelo"] { display: flex; justify-content: center; gap: 0.4rem; }
[role="radiogroup"][aria-label="Modelo"] button[role="radio"] {
  border-radius: 999px; font-weight: 600; font-size: 0.92rem;
  border: 1.5px solid var(--hair); background: #ffffff;
}
[role="radiogroup"][aria-label="Modelo"] button[role="radio"]:hover { border-color: #94a3b8; }
[role="radiogroup"][aria-label="Modelo"] button[role="radio"][aria-checked="true"],
[role="radiogroup"][aria-label="Modelo"] button[role="radio"][aria-checked="true"] * { color: #ffffff !important; }
[role="radiogroup"][aria-label="Modelo"] button[role="radio"][aria-checked="true"] { background: var(--accent); border-color: var(--accent); }
[role="radiogroup"][aria-label="Modelo"] button[role="radio"]:nth-of-type(1)[aria-checked="true"] { background: #FF6600; border-color: #FF6600; }
[role="radiogroup"][aria-label="Modelo"] button[role="radio"]:nth-of-type(2)[aria-checked="true"] { background: #10b981; border-color: #10b981; }
[role="radiogroup"][aria-label="Modelo"] button[role="radio"]:nth-of-type(3)[aria-checked="true"] { background: #eab308; border-color: #d4a406; }

.input-frame { border: 2px solid var(--hair); border-radius: 12px; padding: 6px; background: #ffffff; overflow: hidden; }
.input-img { width: 100%; display: block; border-radius: 8px; transition: transform 200ms ease-out; }
.input-frame:hover .input-img { transform: scale(1.015); }
.input-badge { text-align: center; font-size: 0.92rem; color: var(--muted); margin-top: 0.45rem; }
.input-badge strong { color: var(--ink); }
.input-badge .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; }

.kpi-strip { display: flex; border-top: 1px solid var(--hair); border-bottom: 1px solid var(--hair); }
.kpi { flex: 1; padding: 0.65rem 0.9rem; border-left: 1px solid var(--hair); }
.kpi:first-child { border-left: none; padding-left: 0.1rem; }
.kpi-tick { display: block; width: 22px; height: 3px; border-radius: 2px; margin-bottom: 0.4rem; }
.kpi-label { display: block; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.kpi-value { display: block; font-size: 1.5rem; font-weight: 650; font-variant-numeric: tabular-nums; margin-top: 0.1rem; }
.kpi-note { display: block; font-size: 0.82rem; color: var(--muted); margin-top: 0.05rem; }

.conclusion-panel {
  background: #eff6ff; border-radius: 10px; padding: 0.8rem 1rem;
  font-size: 0.96rem; line-height: 1.55; color: var(--ink); margin-top: 0.8rem;
}

.table-scroll { overflow-x: auto; }
table.min { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
table.min th {
  text-align: left; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--muted); padding: 0.4rem 0.75rem 0.4rem 0;
  border-bottom: 1px solid var(--hair);
}
table.min td { padding: 0.45rem 0.75rem 0.45rem 0; border-bottom: 1px solid #f1f5f9; }
table.min tr:last-child td { border-bottom: none; }
table.min td.num, table.min th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.min td.fam { color: var(--muted); }
table.min td.best { font-weight: 650; color: var(--accent); }
table.min td.model-cell { font-weight: 600; }

.stApp :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button { width: 100%; min-height: 48px; font-size: 1rem; font-weight: 700; }

@media (max-width: 800px) {
  .kpi-strip { flex-wrap: wrap; }
  .kpi { flex: 1 1 40%; padding: 0.5rem 0.7rem; }
  .kpi:nth-child(odd) { border-left: none; padding-left: 0.1rem; }
  .kpi:nth-child(n+3) { border-top: 1px solid var(--hair); }
  .arch-strip { flex-direction: column; }
  .arch-arrow { transform: rotate(90deg); text-align: center; }
}
</style>
        """,
        unsafe_allow_html=True,
    )

def dependency_warning() -> None:
    missing = []
    for package_name, import_name in [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
    ]:
        try:
            __import__(import_name)
        except Exception:
            missing.append(package_name)
    if missing:
        st.warning(
            "Faltan dependencias necesarias para inferencia: "
            + ", ".join(missing)
            + ". Instalalas con `python -m pip install -r requirements.txt` dentro del entorno `.venv`."
        )


OPERATIONAL_MESSAGES = {
    "cold_start": (
        "La primera corrida puede tardar más mientras se descargan los pesos del modelo y se inicializa la caché local. "
        "Las corridas siguientes reutilizan los pesos en caché."
    ),
    "loading": "Cargando modelos y midiendo frames. Dejá esta pestaña abierta hasta que termine la corrida.",
    "empty": (
        "Todavía no hay evidencia de benchmark. Elegí una fuente y ejecutá una medición "
        "para generar resultados."
    ),
    "partial": (
        "Hay resultados parciales disponibles. Revisá las advertencias de los modelos de arriba "
        "y reintentá los que fallaron tras revisar sus pesos y dependencias."
    ),
    "failure": (
        "No se produjeron resultados utilizables. Revisá dependencias, pesos, acceso a la fuente "
        "y disponibilidad del dispositivo, y reintentá."
    ),
    "active_stream": (
        "El streaming en vivo está activo. Los frames se miden continuamente; apretá Detener "
        "para terminar y ver el resumen."
    ),
}


def render_operational_state(state: str, detail: str = "") -> None:
    """Show concise status and recovery guidance for a user-visible run state."""
    message = OPERATIONAL_MESSAGES.get(state, "")
    if detail:
        message = f"{message} {detail}".strip()
    renderer = st.error if state == "failure" else st.warning if state == "partial" else st.info
    renderer(message)


def render_cold_start_notice() -> None:
    """Explain model initialization once per session, immediately before loading."""
    if st.session_state.get("cold_start_notice_shown", False):
        return
    st.session_state["cold_start_notice_shown"] = True
    render_operational_state("cold_start")


def render_evidence_path(df: pd.DataFrame | None = None) -> None:
    if df is None or df.empty:
        st.info(
            "La evidencia no está disponible hasta que corras un benchmark. Usá la pestaña Benchmark "
            "para elegir una fuente, medir los modelos, inspeccionar los resultados y descargar el CSV."
        )
        return
    st.success(
        f"Evidencia disponible para {len(df)} modelo(s): entrada medida → métricas de tiempo "
        "→ detecciones visuales → exportación a CSV."
    )


def render_card(title: str, body: str, accent: str = "blue") -> None:
    st.markdown(
        f"""
<div class="glass-card card-accent-{accent}">
  <h3>{title}</h3>
  <p>{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def source_frames(source_kind: str, total_needed: int, imgsz: int) -> tuple[list[object], object | None]:
    if source_kind == "Demo persona/perro/fruta":
        frame = sample_coco_frame()
        return repeat_frame(frame, total_needed), frame

    if source_kind == "Subir imagen":
        # El uploader vive en la sección Entrada; acá se reutiliza el frame cacheado.
        frame = st.session_state.get("uploaded_image_frame")
        if frame is None:
            return [], None
        return repeat_frame(frame, total_needed), frame

    if source_kind == "Webcam local (OpenCV)":
        if not CAPABILITIES.webcam:
            st.error("La webcam no está disponible en este entorno. Elegí la demo o subí una imagen.")
            return [], None
        camera_index = int(st.session_state.get("camera_index", 0))
        try:
            frames = frames_from_webcam(camera_index, limit=total_needed)
        except Exception as exc:
            st.error(
                f"No se pudo acceder a la webcam índice {camera_index}: {exc}. "
                "Revisá permisos, conectá una cámara o elegí otra fuente."
            )
            return [], None
        if not frames:
            st.error(
                f"La webcam índice {camera_index} no devolvió frames. "
                "Revisá permisos, conexión y disponibilidad de la cámara."
            )
        preview = frames[0] if frames else None
        return frames, preview

    frame = sample_coco_frame()
    return repeat_frame(frame, total_needed), frame


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def draw_detections(img, detections: list[dict], color_hex: str):
    height, width = img.shape[:2]
    color = hex_to_bgr(color_hex)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        p1 = (int(x1 * width), int(y1 * height))
        p2 = (int(x2 * width), int(y2 * height))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{LABEL_ES.get(det['label'], det['label'])} {det['conf']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tag_bottom = max(text_h + 8, p1[1])
        cv2.rectangle(img, (p1[0], tag_bottom - text_h - 8), (p1[0] + text_w + 6, tag_bottom), color, -1)
        cv2.putText(img, label, (p1[0] + 3, tag_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def encode_png(img) -> str:
    ok, encoded = cv2.imencode(".png", img)
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@st.cache_resource(show_spinner=False)
def get_loaded(model_display: str):
    from yolo_complexity_lab.loaders import load_model

    key = {"YOLO11n": "yolo11n", "SSDlite": "ssdlite_mobilenet_v3", "Faster R-CNN": "fasterrcnn_mobilenet_fpn"}[model_display]
    return load_model(key, "cpu")


@st.cache_data(show_spinner=False)
def detect_live(model_display: str, image_bytes: bytes, conf: float, iou: float) -> tuple[list[dict], float]:
    """Corre el modelo elegido sobre una imagen (416 px) y devuelve cajas normalizadas y ms."""
    import torch

    from yolo_complexity_lab.catalog import MODEL_CATALOG
    from yolo_complexity_lab.loaders import load_model

    key = {"YOLO11n": "yolo11n", "SSDlite": "ssdlite_mobilenet_v3", "Faster R-CNN": "fasterrcnn_mobilenet_fpn"}[model_display]
    loaded = load_model(key, "cpu")
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    resized = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # los modelos esperan RGB; cv2 decodifica BGR

    detections: list[dict] = []
    start = time.perf_counter()
    if MODEL_CATALOG[key].backend == "ultralytics":
        results = loaded.model.predict(source=rgb, imgsz=416, conf=conf, iou=iou, device="cpu", verbose=False)
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            names = getattr(results[0], "names", {})
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                class_id = int(box.cls.item())
                detections.append({
                    "box": [v / 416 for v in xyxy],
                    "label": str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id),
                    "conf": round(float(box.conf.item()), 2),
                })
    else:
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        with torch.inference_mode():
            outputs = loaded.model([tensor])
        out0 = outputs[0]
        from torchvision.ops import nms

        keep = nms(out0["boxes"], out0["scores"], iou)
        names = loaded.class_names
        for i in keep.tolist():
            score = float(out0["scores"][i])
            if score < conf:
                continue
            class_id = int(out0["labels"][i])
            label = names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, str(class_id))
            detections.append({
                "box": [v / 416 for v in out0["boxes"][i].tolist()],
                "label": str(label),
                "conf": round(score, 2),
            })
    elapsed_ms = (time.perf_counter() - start) * 1000
    return detections, round(elapsed_ms, 1)


def detections_summary(detections: list[dict]) -> str:
    counts: dict[str, int] = {}
    for det in detections:
        label = LABEL_ES.get(det["label"], det["label"])
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "sin detecciones"
    return ", ".join(f"{label} ({n})" for label, n in counts.items())


def render_input_preview(model_name: str, frame_bgr, detections: list[dict], color_hex: str) -> None:
    """Imagen de entrada con cajas superpuestas y badge informativo."""
    if model_name != "Sin cajas" and detections:
        frame_bgr = draw_detections(frame_bgr.copy(), detections, color_hex)
    encoded = encode_png(frame_bgr)
    st.markdown(
        f'<div class="input-frame" style="border-color:{color_hex if model_name != "Sin cajas" else "var(--hair)"}">'
        f'<img class="input-img" src="data:image/png;base64,{encoded}" alt="Entrada con detecciones de {model_name}"/></div>',
        unsafe_allow_html=True,
    )
    if model_name != "Sin cajas":
        st.markdown(
            f'<div class="input-badge"><span class="dot" style="background:{color_hex}"></span>'
            f"<strong>{model_name}</strong> · {detections_summary(detections)}</div>",
            unsafe_allow_html=True,
        )


def run_webcam_benchmark_streaming(loaded, imgsz: int, confidence: float, iou: float, device: str, camera_index: int, measure_frames: int | None = None) -> dict[str, list]:
    """Run real-time webcam streaming while collecting benchmark metrics."""
    if not CAPABILITIES.streaming:
        st.error("El streaming en vivo no está disponible en este entorno. Elegí una fuente de imagen soportada.")
        return {}

    import cv2
    import time
    import statistics

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
        return float(ordered[index])
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        st.error(
            f"No se pudo abrir la webcam índice {camera_index}. "
            "Revisá permisos, conexión y disponibilidad de la cámara."
        )
        return {}

    st.session_state["streaming_active"] = True
    
    placeholder_video = st.empty()
    placeholder_stats = st.empty()
    stop_placeholder = st.empty()
    
    timings = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "total": [],
        "detections": [],
    }
    
    frame_count = 0

    # Initialize the stop flag in session state and render the button once.
    if "stream_stop_requested" not in st.session_state:
        st.session_state["stream_stop_requested"] = False

    stop_col = stop_placeholder.columns([4, 1])[1]
    if stop_col.button("Stop", key="stop_btn_stream"):
        st.session_state["stream_stop_requested"] = True

    render_operational_state("active_stream")
    consecutive_read_failures = 0
    consecutive_frame_failures = 0

    try:
        while True:
            # Si el usuario solicitó parar, salimos del loop
            if st.session_state.get("stream_stop_requested", False):
                break
            if measure_frames is not None and frame_count >= measure_frames:
                break
            
            success, frame = cap.read()
            if not success:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 3:
                    st.error(
                        f"La webcam índice {camera_index} dejó de entregar frames. "
                        "Revisá permisos y conexión, y reintentá."
                    )
                    break
                continue
            consecutive_read_failures = 0
            
            # Redimensionar
            frame_resized = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            
            # Medir inferencia
            detections_count = 0
            annotated_rgb = frame_resized.copy()
            
            try:
                if loaded.spec.backend == "ultralytics":
                    # YOLO backend con timing
                    start_total = time.perf_counter()
                    
                    results = loaded.model.predict(
                        source=frame_resized,
                        imgsz=imgsz,
                        conf=confidence,
                        iou=iou,
                        device=device,
                        verbose=False,
                    )
                    
                    total_ms = (time.perf_counter() - start_total) * 1000
                    
                    # Dibujar manualmente para mantener la misma convención visual
                    # que el benchmark: OpenCV en BGR, Streamlit en RGB.
                    annotated_frame = frame_resized.copy()
                    try:
                        boxes = getattr(results[0], "boxes", None)
                        names = getattr(results[0], "names", {})
                        if boxes is not None:
                            for box in boxes:
                                xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                                if len(xyxy) < 4:
                                    continue
                                x1, y1, x2, y2 = xyxy[:4]
                                class_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                                conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                                name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                                label = f"{name} {conf:.2f}"
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), DETECTION_ORANGE_BGR, 2)
                                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                label_top = max(0, y1 - text_h - 8)
                                cv2.rectangle(annotated_frame, (x1, label_top), (x1 + text_w + 4, y1), DETECTION_ORANGE_BGR, -1)
                                cv2.putText(annotated_frame, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    except Exception:
                        annotated_frame = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    detections_count = len(getattr(results[0], "boxes", []) or [])
                    
                    # Extraer timings internos de YOLO
                    try:
                        speed = getattr(results[0], "speed", {}) or {}
                        preprocess_ms = float(speed.get("preprocess", 0.0))
                        inference_ms = float(speed.get("inference", 0.0))
                        postprocess_ms = float(speed.get("postprocess", 0.0))
                    except Exception:
                        preprocess_ms = 0.0
                        inference_ms = total_ms
                        postprocess_ms = 0.0
                    
                    timings["preprocess"].append(preprocess_ms)
                    timings["inference"].append(inference_ms)
                    timings["postprocess"].append(postprocess_ms)
                    timings["total"].append(total_ms)
                    timings["detections"].append(detections_count)
                    
                    frame_count += 1
                    consecutive_frame_failures = 0
                    
                    # Persist partial results every frame so pressing Stop (or a
                    # crash) still leaves usable data for the summary.
                    st.session_state["pending_streaming_results"] = {
                        "frames_measured": frame_count,
                        "latency_mean_ms": statistics.mean(timings["total"]),
                        "latency_median_ms": statistics.median(timings["total"]),
                        "latency_min_ms": min(timings["total"]),
                        "latency_max_ms": max(timings["total"]),
                        "latency_p95_ms": percentile(timings["total"], 95),
                        "fps_effective": 1000 / statistics.mean(timings["total"]) if statistics.mean(timings["total"]) > 0 else 0,
                        "preprocess_mean_ms": statistics.mean(timings["preprocess"]) if timings["preprocess"] else 0,
                        "inference_mean_ms": statistics.mean(timings["inference"]) if timings["inference"] else 0,
                        "postprocess_mean_ms": statistics.mean(timings["postprocess"]) if timings["postprocess"] else 0,
                        "detections_mean": statistics.mean(timings["detections"]) if timings["detections"] else 0,
                    }
                    st.session_state["pending_model_key"] = loaded.spec.key
                    st.session_state["pending_imgsz"] = imgsz
                    st.session_state["pending_device"] = device
                    # --------------------------------
                    
                    # Mostrar frame
                    with placeholder_video.container():
                        col_img, col_info = st.columns([3, 1])
                        with col_img:
                            frame_label = f"Frame en vivo {frame_count}" if measure_frames is None else f"Frame en vivo {frame_count}/{measure_frames}"
                            col_img.image(annotated_rgb, caption=frame_label, channels="RGB", width="stretch")
                        with col_info:
                            st.metric("Detecciones", detections_count)
                    
                            # Mostrar estadísticas en tiempo real
                            if frame_count > 0 and timings["total"]:
                                with placeholder_stats.container():
                                    col1, col2, col3, col4 = st.columns(4)
                            
                                    with col1:
                                        avg_latency = statistics.mean(timings["total"])
                                        st.metric("Latencia media", f"{avg_latency:.2f} ms")
                            
                                    with col2:
                                        if statistics.mean(timings["total"]) > 0:
                                            avg_fps = 1000 / statistics.mean(timings["total"])
                                        else:
                                            avg_fps = 0
                                        st.metric("FPS promedio", f"{avg_fps:.1f}")
                            
                                    with col3:
                                        st.metric("Frames capturados", frame_count)
                            
                                    with col4:
                                        avg_detections = statistics.mean(timings["detections"])
                                        st.metric("Detecciones promedio", f"{avg_detections:.1f}")

                            # Descarga CSV en vivo (actualiza cada iteración)
                            try:
                                import io
                                import pandas as _pd
                                df_live = _pd.DataFrame({
                                    "preprocess_ms": timings["preprocess"],
                                    "inference_ms": timings["inference"],
                                    "postprocess_ms": timings["postprocess"],
                                    "total_ms": timings["total"],
                                    "detections": timings["detections"],
                                })
                                csv_bytes = df_live.to_csv(index=False).encode("utf-8")
                                placeholder_stats.download_button(
                                    "Descargar CSV parcial",
                                    csv_bytes,
                                    file_name="streaming_partial_results.csv",
                                    mime="text/csv",
                                    key=f"download_stream_csv_{frame_count}",
                                )
                            except Exception:
                                pass
                else:
                    st.error("El streaming solo soporta modelos YOLO por ahora.")
                    break
                    
            except Exception as e:
                consecutive_frame_failures += 1
                st.warning(f"El frame {frame_count + 1} no pudo medirse: {str(e)[:100]}")
                if consecutive_frame_failures >= 3:
                    st.error(
                        "El streaming se detuvo tras errores repetidos de frame. "
                        "Revisá la compatibilidad del modelo y el dispositivo, y reintentá."
                    )
                    break
                continue
            
            time.sleep(0.02)
        
        # Resumen final
        if frame_count > 0 and timings["total"]:
            return {
                "frames_measured": frame_count,
                "latency_mean_ms": statistics.mean(timings["total"]),
                "latency_median_ms": statistics.median(timings["total"]),
                "latency_min_ms": min(timings["total"]),
                "latency_max_ms": max(timings["total"]),
                "latency_p95_ms": percentile(timings["total"], 95),
                "fps_effective": 1000 / statistics.mean(timings["total"]) if statistics.mean(timings["total"]) > 0 else 0,
                "preprocess_mean_ms": statistics.mean(timings["preprocess"]) if timings["preprocess"] else 0,
                "inference_mean_ms": statistics.mean(timings["inference"]) if timings["inference"] else 0,
                "postprocess_mean_ms": statistics.mean(timings["postprocess"]) if timings["postprocess"] else 0,
                "detections_mean": statistics.mean(timings["detections"]) if timings["detections"] else 0,
                "timings": timings,
            }
        return {}
        
    finally:
        cap.release()
        try:
            st.session_state["stream_stop_requested"] = False
            st.session_state["streaming_active"] = False
        except Exception:
            pass


def compact_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a student-friendly summary before the technical table."""
    summary = df.copy()
    if "input_size_px" in summary.columns:
        summary["input_pixels_n"] = summary["input_size_px"].astype(int) ** 2
    columns = [
        "model",
        "family",
        "latency_mean_ms",
        "latency_p95_ms",
        "fps_effective",
        "input_pixels_n",
        "gflops_approx",
        "parameters_millions",
        "recognized_classes",
    ]
    available = [col for col in columns if col in summary.columns]
    summary = summary[available].copy()
    return summary.rename(
        columns={
            "model": "Modelo",
            "family": "Familia",
            "latency_mean_ms": "Latencia media (ms)",
            "latency_p95_ms": "p95 (ms)",
            "fps_effective": "FPS",
            "input_pixels_n": "n = H×W",
            "gflops_approx": "GFLOPs aprox.",
            "parameters_millions": "Parámetros (M)",
            "recognized_classes": "Clases reconocidas",
        }
    )


def metric_cards(df: pd.DataFrame, presentation_mode: bool = False) -> None:
    if df.empty:
        return
    if presentation_mode:
        st.markdown(
            "<style>[data-testid=\"stMetric\"] { padding: 1.8rem; }</style>",
            unsafe_allow_html=True,
        )
    fastest = df.sort_values("latency_mean_ms").iloc[0]
    most_fps = df.sort_values("fps_effective", ascending=False).iloc[0]
    if "gflops_approx" in df.columns and df["gflops_approx"].notna().any():
        lowest_cost = df.sort_values("gflops_approx", na_position="last").iloc[0]
        cost_label = "Menor GFLOPs aprox."
        cost_value = _display_value(lowest_cost, "gflops_approx", " G", decimals=2)
        cost_delta = lowest_cost["model"]
    else:
        lowest_cost = df.sort_values("parameters_millions", na_position="last").iloc[0]
        cost_label = "Menos parámetros"
        cost_value = _display_value(lowest_cost, "parameters_millions", " M", decimals=2)
        cost_delta = lowest_cost["model"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Menor latencia media", _display_value(fastest, "latency_mean_ms", " ms", decimals=1), fastest["model"])
    c2.metric("Mayor FPS efectivo", _display_value(most_fps, "fps_effective", " FPS", decimals=1), most_fps["model"])
    c3.metric(cost_label, cost_value, cost_delta)


def comparison_winner_deltas(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Summarize measured winners without changing the benchmark rows."""
    metric_specs = (
        ("latency_mean_ms", "Menor latencia", False, "menor que"),
        ("fps_effective", "Mayor FPS efectivo", True, "mayor que"),
        ("gflops_approx", "Menor GFLOPs", False, "menor que"),
    )
    winners: dict[str, dict[str, object]] = {}
    for metric, label, higher_is_better, delta_direction in metric_specs:
        if metric not in df.columns or "model" not in df.columns:
            continue
        values = df[["model", metric]].copy()
        values[metric] = pd.to_numeric(values[metric], errors="coerce")
        values = values.dropna(subset=[metric]).sort_values(
            metric, ascending=not higher_is_better, kind="mergesort"
        )
        if values.empty:
            continue
        winner = values.iloc[0]
        runner_up = values.iloc[1] if len(values) > 1 else None
        delta_pct = None
        if runner_up is not None and float(runner_up[metric]) != 0:
            difference = float(winner[metric]) - float(runner_up[metric])
            delta_pct = round(abs(difference) / abs(float(runner_up[metric])) * 100, 1)
        winners[metric] = {
            "label": label,
            "model": str(winner["model"]),
            "value": float(winner[metric]),
            "delta_pct": delta_pct,
            "delta_direction": delta_direction,
        }
    return winners


def render_comparison_presentation(df: pd.DataFrame) -> None:
    """Present measured winners and a concise, evidence-bound conclusion."""
    winners = comparison_winner_deltas(df)
    if not winners:
        return

    st.markdown("<h3 class='section-title'>Comparación medida</h3>", unsafe_allow_html=True)
    columns = st.columns(len(winners))
    value_suffixes = {
        "latency_mean_ms": (" ms", 1),
        "fps_effective": (" FPS", 1),
        "gflops_approx": (" G", 2),
    }
    for column, (metric, result) in zip(columns, winners.items(), strict=False):
        suffix, decimals = value_suffixes[metric]
        value = f"{result['value']:.{decimals}f}{suffix}"
        delta = result["delta_pct"]
        delta_text = (
            f"{delta:.1f}% {result['delta_direction']} el siguiente modelo"
            if delta is not None
            else "Único modelo medido"
        )
        column.metric(result["label"], value, delta_text)

    latency_winner = winners.get("latency_mean_ms")
    fps_winner = winners.get("fps_effective")
    if latency_winner and fps_winner:
        measured_models = df["model"].dropna().astype(str).unique()
        if len(measured_models) == 1:
            conclusion = f"Solo se midió un modelo: {measured_models[0]}."
        elif latency_winner["model"] == fps_winner["model"]:
            conclusion = (
                f"{latency_winner['model']} lidera esta comparación medida tanto en latencia "
                f"como en FPS efectivo."
            )
        else:
            conclusion = (
                f"{latency_winner['model']} tiene la menor latencia medida, mientras que "
                f"{fps_winner['model']} alcanza el mayor FPS efectivo."
            )
        conclusion += " La conclusión está acotada a la entrada, el dispositivo y la configuración elegidos."
        st.markdown(
            f"""
<div class="glass-card card-accent-orange">
  <span class="small-label">Conclusión medida</span>
  <p>{conclusion}</p>
</div>
            """,
            unsafe_allow_html=True,
        )


def plot_results(df: pd.DataFrame) -> list[tuple[str, object]]:
    plots = []
    if df.empty:
        return plots

    color_map = {
        "YOLO": "#FF6600",
        "CNN one-stage": "#34d399",
        "CNN two-stage": "#fbbf24",
    }
    template = "plotly_white"
    common_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#172033"),
        margin=dict(l=28, r=18, t=48, b=72),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=330,
    )

    latency_fig = px.bar(
        df,
        x="model",
        y="latency_mean_ms",
        color="family",
        color_discrete_map=color_map,
        template=template,
        title="Latencia por modelo",
        labels={"latency_mean_ms": "ms", "model": ""},
        text="latency_mean_ms",
    )
    latency_fig.update_layout(**common_layout)
    latency_fig.update_traces(texttemplate="%{text:.1f} ms", textposition="outside", cliponaxis=False)
    latency_fig.update_xaxes(tickangle=-12)
    plots.append(("latencia_media", latency_fig))

    fps_fig = px.bar(
        df,
        x="model",
        y="fps_effective",
        color="family",
        color_discrete_map=color_map,
        template=template,
        title="FPS efectivo",
        labels={"fps_effective": "FPS", "model": ""},
        text="fps_effective",
    )
    fps_fig.update_layout(**common_layout)
    fps_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fps_fig.update_xaxes(tickangle=-12)
    plots.append(("fps_efectivo", fps_fig))

    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(latency_fig, width="stretch", config={"displayModeBar": False})
    with top_right:
        st.plotly_chart(fps_fig, width="stretch", config={"displayModeBar": False})

    if "gflops_approx" in df.columns and df["gflops_approx"].notna().any():
        hover_data = {
            "gflops_approx": ":.2f",
            "latency_mean_ms": ":.2f",
            "parameters_millions": ":.2f",
            "fps_effective": ":.1f",
        }
        if "recognized_classes" in df.columns:
            hover_data["recognized_classes"] = True
        complexity_fig = px.scatter(
            df,
            x="gflops_approx",
            y="latency_mean_ms",
            size="parameters_millions",
            color="family",
            color_discrete_map=color_map,
            hover_name="model",
            template=template,
            title="Complejidad computacional vs rendimiento en tiempo real",
            labels={
                "gflops_approx": "GFLOPs aproximados",
                "latency_mean_ms": "Mean latency (ms)",
                "parameters_millions": "Parameters (M)",
            },
            hover_data=hover_data,
        )
        complexity_fig.update_layout(**{**common_layout, "height": 390, "margin": dict(l=48, r=18, t=48, b=54)})
        st.plotly_chart(complexity_fig, width="stretch", config={"displayModeBar": False})
        plots.append(("gflops_vs_latencia", complexity_fig))

    return plots

def render_model_overview() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card(
            "1. YOLO en vivo",
            "Arrancá con la webcam local: una pasada por frame hace visible el trade-off del tiempo real.",
            "orange",
        )
    with c2:
        render_card(
            "2. Comparación",
            "Medí Faster R-CNN, SSDlite y YOLO con la misma entrada y la misma configuración.",
            "green",
        )
    with c3:
        render_card(
            "3. Conclusión",
            "Usá latencia, FPS y n = H×W para explicar el trade-off de complejidad medido.",
            "violet",
        )


def render_theory_bridge() -> None:
    st.markdown("<h2 class='section-title'>De la teoría a la evidencia medida</h2>", unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        (
            "Pregunta",
            "La detección de objetos debe localizar y clasificar varios objetos. La pregunta YOLO-vs-CNN es si una sola pasada reduce el costo de ejecución.",
            "amber",
        ),
        (
            "Método",
            "Corré la misma entrada por los detectores elegidos y compará tiempo, FPS efectivo, costo del modelo y clases reconocidas.",
            "orange",
        ),
        (
            "Costo",
            "Las convoluciones dominan el crecimiento: resolución, capas, canales y tamaño de kernel aumentan la cantidad de operaciones.",
            "violet",
        ),
        (
            "Evidencia",
            "El benchmark conecta Big-O con GFLOPs, parámetros, latencia, FPS y clases reconocidas, y exporta las filas medidas a CSV.",
            "green",
        ),
    ]
    for col, (title, body, accent) in zip(cols, cards, strict=False):
        with col:
            render_card(title, body, accent)


def render_metric_glossary() -> None:
    cols = st.columns(len(METRIC_EXPLANATIONS))
    accents = ["blue", "green", "violet", "amber", "blue"]
    for col, (name, description), accent in zip(cols, METRIC_EXPLANATIONS.items(), accents, strict=False):
        with col:
            render_card(name, description, accent)


def render_explanation_flow() -> None:
    steps = st.columns(3)
    content = [
        ("Tiempos", "Latencia media y p95 más FPS efectivo."),
        ("Complejidad", "n = H×W, GFLOPs y cantidad de parámetros."),
        ("Reconocimiento", "Un apoyo visual para esta entrada, no una evaluación formal de mAP."),
    ]
    accents = ["blue", "green", "violet"]
    for col, (title, body), accent in zip(steps, content, accents, strict=False):
        with col:
            render_card(title, body, accent)


def render_config_summary(
    selected_models: list[str],
    source_kind: str,
    device: str,
    imgsz: int,
    warmup_frames: int,
    measure_frames: int,
    include_complexity: bool,
    presentation_mode: bool = False,
) -> None:
    model_names = ", ".join(MODEL_CATALOG[key].display_name for key in selected_models) if selected_models else "Sin modelos elegidos"
    if presentation_mode:
        st.caption(f"Modelos: {model_names} | Fuente: {source_kind} | Dispositivo: {device} | {imgsz}×{imgsz} | Calentamiento: {warmup_frames} | Medidos: {measure_frames}")
    else:
        st.markdown(
            f"""
<div class="glass-card card-accent-violet">
  <span class="small-label">Configuración actual</span>
  <p>Modelos: {model_names} | Fuente: {source_kind} | Dispositivo: {device} | {imgsz}×{imgsz} | Calentamiento: {warmup_frames} | Medidos: {measure_frames}</p>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_benchmark_focus(imgsz: int, streaming_mode: bool, comparison_route: str) -> None:
    n_pixels = imgsz * imgsz
    baseline_n = 320 * 320
    growth = n_pixels / baseline_n
    mode_title = "YOLO en vivo" if streaming_mode else "Comparación medida"
    st.markdown("<h3 class='section-title'>Qué observar</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card(
            "Tiempos",
            "Menor latencia = más FPS. Para 30 FPS, cada frame debe tardar cerca de 33 ms o menos.",
            "blue",
        )
    with c2:
        render_card(
            "Complejidad n",
            f"n = H×W = {imgsz}×{imgsz} = {n_pixels:,} píxeles. Comparada con 320², procesa {growth:.2f}× más píxeles.",
            "violet",
        )
    with c3:
        render_card(
            mode_title,
            "YOLO procesa cada imagen en una sola pasada; los modelos two-stage agregan propuestas de regiones y más cómputo.",
            "green" if streaming_mode else "amber",
        )


def render_result_interpretation(df: pd.DataFrame, presentation_mode: bool = False) -> None:
    if df.empty:
        return
    fastest = df.sort_values("latency_mean_ms").iloc[0]
    highest_fps = df.sort_values("fps_effective", ascending=False).iloc[0]
    most_detections = df.sort_values("detections_mean", ascending=False).iloc[0]
    complexity_df = df.dropna(subset=["gflops_approx"])
    if not complexity_df.empty:
        lowest_gflops = complexity_df.sort_values("gflops_approx").iloc[0]
        complexity_sentence = (
            f"Menor GFLOPs aproximado: <strong>{lowest_gflops['model']}</strong> "
            f"con {lowest_gflops['gflops_approx']} G."
        )
    else:
        complexity_sentence = "GFLOPs no se calcularon en esta corrida."

    theory_sentence = (
        "Con el mismo n = H×W, la latencia y los FPS son la evidencia principal: "
        "cuando YOLO procesa más rápido, se visible el beneficio de su única pasada y de sus optimizaciones de arquitectura."
    )

    if presentation_mode:
        bullets = [
            f"<li>Menor latencia: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms por frame.</li>",
            f"<li>Mayor FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</li>",
            f"<li>Más detecciones para esta entrada: <strong>{most_detections['model']}</strong> — {most_detections.get('recognized_classes', 'sin detalle')}.</li>",
            f"<li>{complexity_sentence} GFLOPs es un proxy; el rendimiento en tiempo real lo definen la latencia y los FPS.</li>",
            f"<li>{theory_sentence}</li>",
        ]
        body = "<ul>" + "".join(bullets) + "</ul>"
    else:
        body = (
            f"<p>Menor latencia: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms por frame.</p>\n"
            f"<p>Mayor FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</p>\n"
            f"<p>Más detecciones para esta entrada: <strong>{most_detections['model']}</strong> — {most_detections.get('recognized_classes', 'sin detalle')}.</p>\n"
            f"<p>{complexity_sentence} GFLOPs es un proxy; el rendimiento en tiempo real lo definen la latencia y los FPS.</p>\n"
            f"<p>{theory_sentence}</p>"
        )

    st.markdown(
        f"""
<div class="glass-card card-accent-green">
  <span class="small-label">Interpretación</span>
  {body}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_summary(df: pd.DataFrame) -> None:
    if df.empty or "recognized_classes" not in df.columns:
        return
    st.markdown("<h3 class='section-title'>Reconocimiento por modelo</h3>", unsafe_allow_html=True)
    st.caption(
        "No reemplaza mAP: es una lectura cualitativa de la imagen usada en el benchmark. "
        "Úsala para revisar falsos negativos, detecciones débiles y diferencias entre familias."
    )
    cols = st.columns(min(3, len(df)))
    for col, (_, row) in zip(cols, df.iterrows(), strict=False):
        with col:
            confidence = row.get("avg_confidence")
            confidence_text = "—" if pd.isna(confidence) else f"{float(confidence):.2f}"
            render_card(
                str(row.get("model", "Model")),
                (
                    f"<strong>Detectado:</strong> {row.get('recognized_classes', 'Sin detecciones')}<br>"
                    f"<strong>Detección principal:</strong> {row.get('top_detection', '—')}<br>"
                    f"<strong>Confianza media:</strong> {confidence_text}"
                ),
                "green" if row.get("family") == "CNN one-stage" else "orange" if row.get("family") == "YOLO" else "amber",
            )


def _display_value(row: pd.Series, key: str, suffix: str = "", decimals: int = 3) -> str:
    value = row.get(key)
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def render_live_yolo_results(df: pd.DataFrame, csv_path: str | None = None, presentation_mode: bool = False) -> None:
    """Render YOLO live results as a practical demo, not a model comparison."""
    if df.empty:
        return

    if presentation_mode:
        st.markdown(
            "<style>[data-testid=\"stMetric\"] { padding: 1.8rem; }</style>",
            unsafe_allow_html=True,
        )

    row = df.iloc[0]
    st.markdown("<h3 class='section-title'>Resumen en vivo de YOLO11n</h3>", unsafe_allow_html=True)
    st.caption(
        "Esta sección no compara modelos: resume cómo rindió YOLO11n con la webcam local: "
        "tiempo por frame, FPS, detecciones y costo aproximado del modelo."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latencia media", _display_value(row, "latency_mean_ms", " ms"))
    c2.metric("FPS efectivo", _display_value(row, "fps_effective", " FPS"))
    c3.metric("Frames procesados", _display_value(row, "frames_measured", decimals=0))
    c4.metric("Detecciones promedio", _display_value(row, "detections_mean", decimals=1))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Latencia p95", _display_value(row, "latency_p95_ms", " ms"))
    c6.metric("Inferencia media", _display_value(row, "inference_mean_ms", " ms"))
    c7.metric("GFLOPs aprox.", _display_value(row, "gflops_approx", " G", decimals=4))
    c8.metric("Parámetros", _display_value(row, "parameters_millions", " M"))


    technical_columns = [
        "model",
        "device",
        "input_size_px",
        "latency_mean_ms",
        "latency_p95_ms",
        "fps_effective",
        "preprocess_mean_ms",
        "inference_mean_ms",
        "postprocess_mean_ms",
        "detections_mean",
        "gflops_approx",
        "parameters_millions",
        "model_size_mb",
    ]
    available = [col for col in technical_columns if col in df.columns]
    with st.expander("Ver datos técnicos de la corrida"):
        st.dataframe(df[available], width="stretch", hide_index=True)

    if csv_path:
        csv_name = Path(csv_path).name
        st.success(f"CSV generado: {csv_name}")
        st.caption(f"Ruta local: {csv_path}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV del resumen",
        csv_bytes,
        file_name="yolo_live_summary.csv",
        mime="text/csv",
        key="download_live_yolo_csv",
        on_click="ignore",
    )


def render_benchmark_results(df: pd.DataFrame, csv_path: str | None = None, presentation_mode: bool = False, is_streaming: bool = False) -> None:
    """Render persisted benchmark results and export actions.

    Streamlit reruns the script whenever a button, checkbox or download action is
    used. Keeping rendering in this helper lets us show the last benchmark from
    st.session_state instead of losing it after export actions.
    """
    if df.empty:
        render_operational_state("empty")
        return

    if is_streaming:
        render_live_yolo_results(df, csv_path, presentation_mode)
        return

    metric_cards(df, presentation_mode)
    render_comparison_presentation(df)

    render_detection_summary(df)

    st.markdown("<h3 class='section-title'>Resumen comparativo</h3>", unsafe_allow_html=True)
    st.dataframe(compact_results_table(df), width="stretch", hide_index=True)

    with st.expander("Ver tabla técnica completa"):
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("<h3 class='section-title'>Gráficas</h3>", unsafe_allow_html=True)
    plots = plot_results(df)

    if csv_path:
        csv_name = Path(csv_path).name
        st.success(f"CSV generado: {csv_name}")
        st.caption(f"Ruta local: {csv_path}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV de resultados",
        csv_bytes,
        file_name="benchmark_yolo_complexity.csv",
        mime="text/csv",
        key="download_results_csv",
        on_click="ignore",
    )

    # Exportación HTML removida a pedido del usuario

    # --- Vista de detecciones: mostrar el último frame anotado por cada modelo ---
    st.markdown("<h3 class='section-title'>Detecciones visuales (último frame medido)</h3>", unsafe_allow_html=True)
    st.caption("Estas imágenes muestran qué objetos detectó cada modelo en el último frame medido. Úsalas para inspeccionar el reconocimiento visual, los falsos positivos y los falsos negativos; no reemplazan mAP.")
    annotated_frames = st.session_state.get("annotated_frames", {})
    if annotated_frames:
        for model_key, frame_bgr in annotated_frames.items():
            if frame_bgr is not None:
                try:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    spec = MODEL_CATALOG.get(model_key)
                    model_name = spec.display_name if spec else model_key
                    st.image(frame_rgb, caption=f"{model_name}", channels="RGB", width="stretch")
                except Exception:
                    pass


def render_architecture_strip() -> None:
    """Show the one-stage detector pipeline referenced by the Big-O discussion."""
    nodes = [
        (
            "Backbone",
            "Las capas convolucionales extraen rasgos visuales. Acá vive la mayor parte del costo O(L × n × C² × K²).",
        ),
        (
            "Neck",
            "Fusiona rasgos de distintas escalas para que objetos chicos y grandes compartan la misma representación.",
        ),
        (
            "Head",
            "Predice cajas y clases en una sola pasada; NMS luego poda solapamientos en O(B²).",
        ),
    ]
    cells: list[str] = []
    for index, (title, body) in enumerate(nodes):
        if index:
            cells.append('<div class="arch-arrow">→</div>')
        cells.append(f'<div class="arch-node"><h4>{title}</h4><p>{body}</p></div>')
    st.markdown(
        f"""
<div class="arch-strip">
  {''.join(cells)}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_environment_summary() -> None:
    """Contextualize results with the hardware and runtime that produced them."""
    info = system_info_dict()
    cores = info.get("cpu_logical_cores")
    ram_gb = info.get("ram_total_gb")
    cores_text = f"{cores} núcleos lógicos" if cores else "cantidad de núcleos no disponible"
    ram_text = (
        f"{float(ram_gb):.0f} GB de RAM"
        if isinstance(ram_gb, (int, float)) and ram_gb
        else "RAM no disponible"
    )
    accelerator = (
        f"GPU CUDA: {info.get('cuda_device', 'CPU')}"
        if info.get("cuda_available")
        else "solo CPU (sin dispositivo CUDA detectado)"
    )
    st.markdown(
        f"""
<div class="glass-card card-accent-blue">
  <span class="small-label">Entorno que produjo estos números</span>
  <p>Python {info.get('python', '?')} · {info.get('platform', 'plataforma desconocida')} · {cores_text} · {ram_text}</p>
  <p>{accelerator}. La latencia y los FPS dependen del hardware: compará corridas medidas en la misma máquina.</p>
</div>
        """,
        unsafe_allow_html=True,
    )


inject_css()

dependency_warning()

with st.sidebar:
    st.markdown('<div class="side-label">Experimento</div>', unsafe_allow_html=True)
    previous_route = st.session_state.get("comparison_route", None)
    route_keys = list(PRESET_MODELS.keys())
    comparison_route = st.radio(
        "Ruta de comparación",
        options=route_keys,
        index=route_keys.index("Comparación CNN vs YOLO") if "Comparación CNN vs YOLO" in route_keys else 0,
        captions=[PRESET_HELP.get(key, "") for key in route_keys],
        label_visibility="collapsed",
        help="Define la historia del experimento; las opciones no disponibles en este entorno están ocultas.",
    )

    # Limpiar resultados viejos cuando cambia la ruta.
    if previous_route is not None and previous_route != comparison_route:
        st.session_state.pop("last_benchmark_df", None)
        st.session_state.pop("last_benchmark_csv_path", None)
        st.session_state.pop("annotated_frames", None)
        st.session_state.pop("pending_streaming_results", None)

    st.session_state["comparison_route"] = comparison_route
    selected_models = PRESET_MODELS[comparison_route]

    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Entrada</div>', unsafe_allow_html=True)
    source_options = list(SOURCE_HELP.keys())
    default_source = (
        "Webcam local (OpenCV)"
        if CAPABILITIES.webcam and comparison_route == "YOLO en vivo"
        else "Demo persona/perro/fruta"
    )
    source_kind = st.selectbox(
        "Fuente de frames",
        source_options,
        index=source_options.index(default_source),
        label_visibility="collapsed",
        help="De dónde salen los frames que se miden.",
    )

    streaming_mode = False
    if CAPABILITIES.webcam and source_kind == "Webcam local (OpenCV)":
        if CAPABILITIES.streaming and comparison_route == "YOLO en vivo":
            streaming_mode = st.checkbox("Modo en vivo", value=True, help="Procesa los frames de la cámara en tiempo real.")
        st.number_input("Índice de cámara", min_value=0, max_value=5, value=0, key="camera_index")

    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Ejecución</div>', unsafe_allow_html=True)
    device = st.selectbox(
        "Dispositivo",
        options=list(DEVICE_OPTIONS),
        index=list(DEVICE_OPTIONS).index(CAPABILITIES.device_default),
        label_visibility="collapsed",
        help="Dónde se ejecutan los modelos.",
    )
    with st.expander("Avanzado"):
        imgsz = st.select_slider("Resolución", options=[320, 416, 512, 640], value=416, help="Más resolución procesa más píxeles: n = H × W crece.")
        warmup_frames = st.number_input("Frames de calentamiento", min_value=0, max_value=30, value=3, disabled=streaming_mode, help="Estabilizan cachés antes de medir; no se reportan.")
        measure_frames = st.number_input("Frames medidos", min_value=1, max_value=300, value=20, disabled=streaming_mode, help="Los frames que entran en latencia y FPS.")
        confidence = st.slider("Confianza mínima", min_value=0.05, max_value=0.95, value=0.25, step=0.05, disabled=streaming_mode, help="Descarta detecciones débiles.")
        iou = st.slider("IoU para NMS", min_value=0.10, max_value=0.95, value=0.45, step=0.05, disabled=streaming_mode, help="Umbral para decidir si dos cajas son el mismo objeto.")
        include_complexity = st.checkbox("Estimar GFLOPs", value=True, disabled=streaming_mode, help="Pasada extra para contar operaciones; puede demorar.")

    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    cuda = "cuda:0" in CAPABILITIES.device_options
    cam = '<span class="ok">webcam disponible</span>' if CAPABILITIES.webcam and CAPABILITIES.streaming else '<span class="off">sin webcam (no se puede capturar desde el servidor)</span>'
    gpu = '<span class="ok">GPU CUDA detectada</span>' if cuda else '<span class="off">sin GPU · se mide en CPU</span>'
    st.markdown(f'<div class="env-note">Este entorno: {cam} · {gpu}</div>', unsafe_allow_html=True)


def render_navigation_tabs():
    return st.tabs(NAVIGATION_TABS)


def render_commercial_header() -> None:
    st.markdown(
        '<div class="page-head">'
        '<div class="brand">YOLO Complexity Lab</div>'
        '<h1 class="commercial-head">Tres detectores. Una imagen. '
        'Un veredicto en <span class="hl">milisegundos</span>.</h1>'
        '<div class="app-context">Faster R-CNN · SSDlite · YOLO11n sobre la misma entrada · '
        'CPU · 416 × 416 · confianza 0,25 · IoU 0,45</div>'
        '</div>',
        unsafe_allow_html=True,
    )


render_commercial_header()
overview_tab, benchmark_tab, about_tab = render_navigation_tabs()

with overview_tab:
    render_model_overview()
    render_evidence_path(st.session_state.get("last_benchmark_df"))
    render_theory_bridge()
    st.markdown("<h3 class='section-title'>Anatomía de un detector one-stage</h3>", unsafe_allow_html=True)
    render_architecture_strip()
    st.markdown("<h2 class='section-title'>Cómo leer el benchmark</h2>", unsafe_allow_html=True)
    render_explanation_flow()
    st.write("")
    with st.expander("Ver glosario de métricas"):
        render_metric_glossary()

with about_tab:
    st.markdown(
        f"""
<h2 class='section-title'>Acerca de este proyecto</h2>
<p style="color: var(--muted); font-size: 0.95rem;"><strong>Hecho por {AUTHOR_NAME}.</strong> Este experimento de portfolio examina la pregunta YOLO-vs-CNN con evidencia medida.</p>
        """,
        unsafe_allow_html=True,
    )
    purpose_col, method_col, evidence_col = st.columns(3)
    with purpose_col:
        render_card(
            "Propósito",
            "Entender cómo la arquitectura del detector afecta el tiempo de ejecución, el costo computacional y el reconocimiento visual.",
            "orange",
        )
    with method_col:
        render_card(
            "Método",
            "Correr la misma entrada y configuración por cada detector elegido, y comparar latencia, FPS, proxies de complejidad y clases reconocidas.",
            "green",
        )
    with evidence_col:
        render_card(
            "Camino de evidencia",
            "Abrí Benchmark, elegí una fuente, ejecutá la medición, inspeccioná los resultados y exportá la sesión como CSV.",
            "violet",
        )
    st.markdown("<h3 class='section-title'>Entorno</h3>", unsafe_allow_html=True)
    render_environment_summary()

with benchmark_tab:
    st.markdown("<h2 class='section-title'>Benchmark</h2>", unsafe_allow_html=True)
    st.caption("¿Cuánto tarda cada frame y cómo crece el costo cuando n = H×W aumenta?")

    total_needed = int(warmup_frames + measure_frames)
    frames, preview = (
        ([], None)
        if source_kind == "Webcam local (OpenCV)"
        else source_frames(source_kind, 1, imgsz)
    )

    # --- Entrada interactiva: elegís el modelo, montás tu foto ---------------
    overlay = "YOLO11n"
    uploaded = None
    if source_kind == "Subir imagen":
        uploaded = st.file_uploader(
            "Montá tu propia foto",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
            help="Se procesa en vivo con el modelo elegido y la misma configuración del benchmark.",
        )
        if uploaded is None:
            st.info("Subí una imagen o elegí la demo persona/perro/fruta para una prueba rápida.")
        else:
            st.session_state["uploaded_image_frame"] = read_image_file(uploaded)
            frames, preview = source_frames(source_kind, 1, imgsz)

    if preview is not None:
        overlay = st.pills(
            "Modelo",
            ["YOLO11n", "SSDlite", "Faster R-CNN", "Sin cajas"],
            selection_mode="single",
            default="YOLO11n",
            label_visibility="collapsed",
        )
        color = MODEL_COLOR.get(overlay, "var(--hair)")
        frame_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        live_dets: list[dict] | None = None
        if source_kind == "Subir imagen" and overlay != "Sin cajas":
            try:
                live_dets, _ = detect_live(overlay, uploaded.getvalue(), float(confidence), float(iou))
            except Exception as exc:
                st.warning(f"No se pudo procesar tu foto con {overlay}: {str(exc)[:120]}")
        detections = DEMO_DETECTIONS[overlay] if (overlay != "Sin cajas" and source_kind != "Subir imagen") else (live_dets or [])
        render_input_preview(overlay, frame_bgr, detections, color)

    # Acción primaria: grande, ancha, inmediatamente debajo de la imagen
    run = st.button(
        "Iniciar YOLO en vivo" if streaming_mode else "Ejecutar benchmark",
        type="primary",
        use_container_width=True,
    )

    if source_kind == "Webcam local (OpenCV)" and not run:
        frames, preview = [], None
        st.info("La webcam local solo se lee cuando ejecutás el benchmark, para evitar capturas innecesarias.")
    elif run:
        # Recargar frames para el benchmark completo
        if streaming_mode:
            frames, preview = [], None
        else:
            frames, preview = source_frames(source_kind, total_needed, imgsz)

    if run:
        if not selected_models:
            st.error("Elegí al menos un modelo para arrancar el benchmark.")
            st.stop()

        # Modo streaming con webcam
        if source_kind == "Webcam local (OpenCV)" and streaming_mode:
            if len(selected_models) > 1:
                st.warning("El streaming admite un modelo por vez. Se usará el primero de la lista.")

            model_key = selected_models[0]
            spec = MODEL_CATALOG[model_key]

            st.markdown(f"<h3>En vivo: {spec.display_name}</h3>", unsafe_allow_html=True)
            st.info(f"Capturando frames de la cámara {st.session_state.get('camera_index', 0)}. Apretá Detener para ver el resumen.")

            try:
                st.session_state.streaming_active = True
                render_operational_state("loading", f"Cargando {spec.display_name}.")
                render_cold_start_notice()
                loaded = cached_load_model(model_key, device)

                streaming_results = run_webcam_benchmark_streaming(
                    loaded,
                    imgsz=int(imgsz),
                    confidence=float(confidence),
                    iou=float(iou),
                    device=device,
                    camera_index=int(st.session_state.get('camera_index', 0)),
                    measure_frames=None,  # sigue hasta que el usuario apriete Detener
                )

                if streaming_results:
                    st.success("Streaming finalizado. Resumen de rendimiento:")

                    # Resumen simple del streaming, sin tabla de comparación.
                    from yolo_complexity_lab.complexity import estimate_for_loaded_model
                    complexity = estimate_for_loaded_model(loaded, int(imgsz)) if include_complexity else None

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Latencia media", f"{round(streaming_results['latency_mean_ms'], 1)} ms")
                        st.metric("FPS efectivo", f"{round(streaming_results['fps_effective'], 1)}")
                    with col2:
                        st.metric("Frames medidos", streaming_results["frames_measured"])
                        st.metric("Detecciones promedio", f"{round(streaming_results['detections_mean'], 1)}")
                    with col3:
                        st.metric("Preprocesamiento", f"{round(streaming_results['preprocess_mean_ms'], 1)} ms")
                        st.metric("Inferencia", f"{round(streaming_results['inference_mean_ms'], 1)} ms")

                    if complexity:
                        st.write(f"**Complejidad:** {complexity.gflops_approx} GFLOPs | {complexity.gmacs} GMACs | {complexity.conv_layers} capas Conv")

                    st.write(f"**Modelo:** {spec.display_name} | **Dispositivo:** {device} | **Resolución:** {imgsz}px")

                    # Guardar para referencia; no se muestra como comparación.
                    row = {
                        "model_key": spec.key,
                        "model": spec.display_name,
                        "family": spec.family,
                        "backend": spec.backend,
                        "device": device,
                        "input_size_px": int(imgsz),
                        "frames_measured": streaming_results["frames_measured"],
                        "warmup_frames": 0,
                        "latency_mean_ms": round(streaming_results["latency_mean_ms"], 3),
                        "latency_median_ms": round(streaming_results["latency_median_ms"], 3),
                        "latency_min_ms": round(streaming_results["latency_min_ms"], 3),
                        "latency_max_ms": round(streaming_results["latency_max_ms"], 3),
                        "latency_p95_ms": round(streaming_results.get("latency_p95_ms", streaming_results["latency_max_ms"]), 3),
                        "fps_effective": round(streaming_results["fps_effective"], 3),
                        "preprocess_mean_ms": round(streaming_results["preprocess_mean_ms"], 3),
                        "inference_mean_ms": round(streaming_results["inference_mean_ms"], 3),
                        "postprocess_mean_ms": round(streaming_results["postprocess_mean_ms"], 3),
                        "detections_mean": round(streaming_results["detections_mean"], 3),
                        "parameters": loaded.parameter_count,
                        "parameters_millions": round(loaded.parameter_count / 1e6, 3) if loaded.parameter_count is not None else None,
                        "model_size_mb": loaded.model_size_mb,
                        "model_size_note": loaded.size_note,
                        "macs": complexity.macs if complexity else None,
                        "gmacs_approx": complexity.gmacs if complexity else None,
                        "gflops_approx": complexity.gflops_approx if complexity else None,
                        "conv_layers_counted": complexity.conv_layers if complexity else None,
                        "linear_layers_counted": complexity.linear_layers if complexity else None,
                        "complexity_note": complexity.note if complexity else "No calculado.",
                        "big_o_inference": spec.inference_big_o,
                        "big_o_didactic": spec.didactic_big_o,
                        "big_o_postprocess": spec.postprocess_big_o,
                        "ram_delta_mb": 0.0,
                    }

                    df = pd.DataFrame([row])
                    export_path = write_results_csv(df)
                    st.session_state["last_benchmark_df"] = df
                    st.session_state["last_benchmark_csv_path"] = str(export_path)

            except Exception as exc:
                st.session_state["streaming_active"] = False
                render_operational_state("failure", f"Error de streaming: {exc}")

        # Modo benchmark estándar
        else:
            if not frames:
                st.error(
                    "No hay frames disponibles para la fuente elegida. "
                    "Elegí otra fuente o revisá la webcam y los permisos."
                )
                st.stop()

            config = BenchmarkConfig(
                imgsz=int(imgsz),
                warmup_frames=int(warmup_frames),
                measure_frames=int(measure_frames),
                confidence=float(confidence),
                iou=float(iou),
            )
            rows = []
            progress = st.progress(0)
            status = st.empty()
            st.session_state["annotated_frames"] = {}

            for index, model_key in enumerate(selected_models, start=1):
                spec = MODEL_CATALOG[model_key]
                status.markdown(f"## Cargando {spec.display_name} ({index} de {len(selected_models)})")
                render_operational_state("loading", f"Preparando {spec.display_name}.")
                try:
                    render_cold_start_notice()
                    loaded = cached_load_model(model_key, device)
                    row = benchmark_model(loaded, frames, config, include_complexity=include_complexity)
                    # El frame anotado se guarda aparte; no es parte del DataFrame.
                    annotated_frame = row.pop("last_annotated_frame", None)
                    if annotated_frame is not None:
                        if "annotated_frames" not in st.session_state:
                            st.session_state["annotated_frames"] = {}
                        st.session_state["annotated_frames"][model_key] = annotated_frame
                    rows.append(row)
                except Exception as exc:
                    st.error(
                        f"{spec.display_name} falló: {exc}. Revisá los pesos y las dependencias, y reintentá."
                    )
                progress.progress(index / len(selected_models))

            status.empty()
            progress.empty()

            if rows:
                if len(rows) < len(selected_models):
                    render_operational_state(
                        "partial",
                        f"{len(rows)} de {len(selected_models)} modelos completaron.",
                    )
                df = pd.DataFrame(rows)
                export_path = write_results_csv(df)
                st.session_state["last_benchmark_df"] = df
                st.session_state["last_benchmark_csv_path"] = str(export_path)
                st.session_state.pop("last_html_zip", None)
                st.session_state.pop("last_html_paths", None)
                render_benchmark_results(df, str(export_path), True)
            else:
                render_operational_state("failure")
    elif "pending_streaming_results" in st.session_state:
        st.success("Streaming finalizado. Mostrando el resumen de YOLO en vivo...")

        # Recuperar los datos guardados en el finally
        res = st.session_state.pop("pending_streaming_results")
        m_key = st.session_state.pop("pending_model_key")
        imgsz_val = st.session_state.pop("pending_imgsz")
        dev_val = st.session_state.pop("pending_device")

        spec = MODEL_CATALOG[m_key]
        loaded = cached_load_model(m_key, dev_val)

        from yolo_complexity_lab.complexity import estimate_for_loaded_model
        complexity = estimate_for_loaded_model(loaded, int(imgsz_val)) if include_complexity else None

        row = {
            "model_key": spec.key,
            "model": spec.display_name,
            "family": spec.family,
            "backend": spec.backend,
            "device": dev_val,
            "input_size_px": int(imgsz_val),
            "frames_measured": res["frames_measured"],
            "warmup_frames": 0,
            "latency_mean_ms": round(res["latency_mean_ms"], 3),
            "latency_median_ms": round(res["latency_median_ms"], 3),
            "latency_min_ms": round(res["latency_min_ms"], 3),
            "latency_max_ms": round(res["latency_max_ms"], 3),
            "latency_p95_ms": round(res.get("latency_p95_ms", res["latency_max_ms"]), 3),
            "fps_effective": round(res["fps_effective"], 3),
            "preprocess_mean_ms": round(res["preprocess_mean_ms"], 3),
            "inference_mean_ms": round(res["inference_mean_ms"], 3),
            "postprocess_mean_ms": round(res["postprocess_mean_ms"], 3),
            "detections_mean": round(res["detections_mean"], 3),
            "parameters": loaded.parameter_count,
            "parameters_millions": round(loaded.parameter_count / 1e6, 3) if loaded.parameter_count is not None else None,
            "model_size_mb": loaded.model_size_mb,
            "model_size_note": loaded.size_note,
            "macs": complexity.macs if complexity else None,
            "gmacs_approx": complexity.gmacs if complexity else None,
            "gflops_approx": complexity.gflops_approx if complexity else None,
            "conv_layers_counted": complexity.conv_layers if complexity else None,
            "linear_layers_counted": complexity.linear_layers if complexity else None,
            "complexity_note": complexity.note if complexity else "No calculado.",
            "big_o_inference": spec.inference_big_o,
            "big_o_didactic": spec.didactic_big_o,
            "big_o_postprocess": spec.postprocess_big_o,
            "ram_delta_mb": 0.0,
        }

        df = pd.DataFrame([row])
        export_path = write_results_csv(df)
        st.session_state["last_benchmark_df"] = df
        st.session_state["last_benchmark_csv_path"] = str(export_path)

        render_benchmark_results(df, str(export_path), True, is_streaming=True)

    elif "last_benchmark_df" in st.session_state:
        st.info("Mostrando el último benchmark completado. Podés descargar el CSV sin volver a medir.")
        render_benchmark_results(
            st.session_state["last_benchmark_df"],
            st.session_state.get("last_benchmark_csv_path"),
            True,
            is_streaming=streaming_mode,
        )
    else:
        if streaming_mode:
            st.info("Iniciá YOLO en vivo para ver latencia, FPS y detecciones de la cámara.")
        else:
            render_operational_state("empty")

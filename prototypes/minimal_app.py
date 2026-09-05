"""Prototipo minimalista del front de YOLO Complexity Lab (diseño + entrada viva).

Pase con la skill `impeccable` (`distill`, modo Operate) + iteración en español.
- Barra lateral jerárquica con el contrato real de capacidades
  (`yolo_complexity_lab.environment.detect_capabilities`).
- Entrada interactiva: la imagen demo trae las detecciones reales de cada
  modelo (mismo pipeline del benchmark: 416 px, conf 0,25, IoU 0,45) y el
  usuario puede montar su propia foto, que se procesa en vivo con los mismos
  modelos.
- KPIs y tabla de la corrida local medida del 2026-09-05; el botón de
  benchmark completo aún no está conectado.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_complexity_lab.environment import detect_capabilities

st.set_page_config(page_title="YOLO Complexity Lab", page_icon="🎯", layout="wide")

CAPABILITIES = detect_capabilities(ROOT, os.environ)

IMGZ = 416
CONF = 0.25
IOU = 0.45


def es_num(value: float, dec: int = 1) -> str:
    return f"{value:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


# --- Datos medidos (corrida real, 2026-09-05, CPU, 416x416) ------------------

MODELS = pd.DataFrame(
    [
        {"model": "YOLO11n", "family": "YOLO", "p50": 38.2, "p95": 63.2, "fps": 26.2, "gflops": 2.74, "params_m": 2.62, "size_mb": 5.4, "detected": "persona (2)", "conf": 0.53},
        {"model": "SSDlite", "family": "CNN one-stage", "p50": 76.8, "p95": 111.4, "fps": 13.0, "gflops": 1.17, "params_m": 3.44, "size_mb": 13.1, "detected": "persona (2), perro (1)", "conf": 0.77},
        {"model": "Faster R-CNN", "family": "CNN two-stage", "p50": 108.0, "p95": 194.3, "fps": 9.3, "gflops": 1.44, "params_m": 19.39, "size_mb": 74.0, "detected": "persona (2), banana (1)", "conf": 0.55},
    ]
).sort_values("p50").reset_index(drop=True)

FAMILY_COLOR = {"YOLO": "#FF6600", "CNN one-stage": "#10b981", "CNN two-stage": "#eab308"}
BEST = {"p50": 0, "p95": 0, "fps": 0, "gflops": 1, "params_m": 0, "size_mb": 0}

# Detecciones pre-extraídas de la imagen demo (mismo pipeline que detect_boxes)
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

MODEL_COLOR = {"YOLO11n": "#FF6600", "SSDlite": "#10b981", "Faster R-CNN": "#eab308"}
LABEL_ES = {"person": "persona", "dog": "perro", "banana": "banana", "horse": "caballo"}

# --- Rutas y fuentes según el contrato real de capacidades -------------------

ROUTES = {
    "YOLO en vivo": {"caption": "Tu webcam, medición frame a frame", "available": CAPABILITIES.streaming},
    "Comparación CNN vs YOLO": {"caption": "Faster R-CNN · SSDlite · YOLO11n sobre la misma entrada", "available": True},
    "Pesos personalizados": {"caption": "Tu best.pt desde la raíz del repo", "available": CAPABILITIES.custom_weights},
}
AVAILABLE_ROUTES = [name for name, spec in ROUTES.items() if spec["available"]]

SOURCE_OPTIONS = [
    name
    for name, available in {
        "Demo persona/perro/fruta": True,
        "Subir imagen": True,
        "Webcam local (OpenCV)": CAPABILITIES.webcam,
    }.items()
    if available
]
DEFAULT_SOURCE = "Webcam local (OpenCV)" if CAPABILITIES.webcam else "Demo persona/perro/fruta"


# --- Sistema de diseño -------------------------------------------------------


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

.sec-head { font-size: 1.1rem; font-weight: 650; color: var(--ink); margin: 1.7rem 0 0.55rem 0; }
.sec-head .dim { color: var(--muted); font-weight: 450; font-size: 0.92rem; }
.chart-label { font-size: 0.9rem; font-weight: 600; color: var(--ink); margin-bottom: 0.15rem; }
.chart-label .dim { color: var(--muted); font-weight: 450; font-size: 0.88rem; }

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
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"] > div:first-child { flex: 1 1 100% !important; }
  [data-testid="stHorizontalBlock"] > div:last-child { flex: 0 0 auto !important; min-width: 190px; }
  .stButton > button { white-space: nowrap; min-width: 176px; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def section_head(text: str, dim: str = "") -> None:
    suffix = f' <span class="dim">· {dim}</span>' if dim else ""
    st.markdown(f'<div class="sec-head">{text}{suffix}</div>', unsafe_allow_html=True)


def kpi_strip(items: list[tuple[str, str, str, str]]) -> None:
    cells = "".join(
        f'<div class="kpi"><span class="kpi-tick" style="background:{color}"></span>'
        f'<span class="kpi-label">{label}</span>'
        f'<span class="kpi-value">{value}</span><span class="kpi-note">{note}</span></div>'
        for label, value, note, color in items
    )
    st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)


def min_table(headers: list[str], rows: list[list[str]], numeric_cols: set[int], best_cols: dict[int, int] | None = None) -> None:
    best_cols = best_cols or {}
    head = "".join(
        f'<th class="{"num" if i in numeric_cols else ""}">{h}</th>' for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i in numeric_cols else ""} '
            f'{"best" if best_cols.get(i) == row_idx else ""} '
            f'{"fam" if headers[i] == "Familia" else ""} '
            f'{"model-cell" if i == 0 else ""}">{c}</td>'
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row_idx, row in enumerate(rows)
    )
    st.markdown(
        '<div class="table-scroll">'
        f'<table class="min"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        "</div>",
        unsafe_allow_html=True,
    )


def base_layout(height: int) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#52525b", size=11),
        margin=dict(l=8, r=46, t=10, b=8),
        height=height,
        showlegend=False,
    )


def hbar_chart(value_col: str, fmt, higher_better: bool, with_legend: bool) -> object:
    order = MODELS.sort_values(value_col, ascending=higher_better)
    fig = px.bar(order, y="model", x=value_col, color="family", color_discrete_map=FAMILY_COLOR, orientation="h", text=[fmt(v) for v in order[value_col]])
    layout = base_layout(height=200)
    layout["margin"] = dict(l=8, r=70, t=10, b=8)
    if with_legend:
        layout["showlegend"] = True
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10), title=None)
    fig.update_layout(**layout)
    fig.update_traces(textposition="outside", textfont=dict(size=10), cliponaxis=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10.5), title=None, categoryorder="array", categoryarray=order["model"].tolist())
    fig.update_xaxes(visible=False)
    return fig


def scatter_chart() -> object:
    fig = px.scatter(
        MODELS, x="gflops", y="p50", size="params_m", color="family", color_discrete_map=FAMILY_COLOR,
        hover_name="model", text="model", labels={"gflops": "GFLOPs aproximados", "p50": "Latencia media (ms)"},
    )
    layout = base_layout(height=230)
    layout["margin"] = dict(l=8, r=30, t=10, b=34)
    layout["showlegend"] = False
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=10, color="#3f3f46"),
        marker=dict(opacity=0.85, line=dict(width=0), sizeref=2.0 * MODELS["params_m"].max() / (14**2), sizemin=7),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#f4f4f5", zeroline=False, tickfont=dict(size=10), title=dict(font=dict(size=10.5)))
    fig.update_yaxes(gridcolor="#f4f4f5", zeroline=False, tickfont=dict(size=10), title=dict(font=dict(size=10.5)))
    return fig


def environment_note() -> None:
    cuda = "cuda:0" in CAPABILITIES.device_options
    cam = '<span class="ok">webcam disponible</span>' if CAPABILITIES.webcam and CAPABILITIES.streaming else '<span class="off">sin webcam (no se puede capturar desde el servidor)</span>'
    gpu = '<span class="ok">GPU CUDA detectada</span>' if cuda else '<span class="off">sin GPU · se mide en CPU</span>'
    st.markdown(f'<div class="env-note">Este entorno: {cam} · {gpu}</div>', unsafe_allow_html=True)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def draw_detections(img: np.ndarray, detections: list[dict], color_hex: str) -> np.ndarray:
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


def encode_png(img: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", img)
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@st.cache_data(show_spinner=False)
def demo_variant(model_name: str) -> str:
    img = cv2.imread(str(ROOT / "assets" / "demo_person_dog_fruit.jpg"))
    height, width = img.shape[:2]
    scale = min(1.0, 1000 / width)
    if scale < 1.0:
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    if model_name != "Sin cajas":
        img = draw_detections(img, DEMO_DETECTIONS[model_name], MODEL_COLOR[model_name])
    return encode_png(img)


# --- Inferencia en vivo para fotos subidas -----------------------------------


@st.cache_resource(show_spinner=False)
def get_loaded(model_display: str):
    from yolo_complexity_lab.loaders import load_model

    key = {"YOLO11n": "yolo11n", "SSDlite": "ssdlite_mobilenet_v3", "Faster R-CNN": "fasterrcnn_mobilenet_fpn"}[model_display]
    return load_model(key, "cpu")


@st.cache_data(show_spinner=False)
def detect_live(model_display: str, image_bytes: bytes) -> tuple[list[dict], float]:
    """Corre el modelo sobre la foto subida; devuelve cajas normalizadas y ms."""
    import torch

    from yolo_complexity_lab.catalog import MODEL_CATALOG
    from yolo_complexity_lab.loaders import load_model

    key = {"YOLO11n": "yolo11n", "SSDlite": "ssdlite_mobilenet_v3", "Faster R-CNN": "fasterrcnn_mobilenet_fpn"}[model_display]
    loaded = load_model(key, "cpu")
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    resized = cv2.resize(frame, (IMGZ, IMGZ), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # los modelos esperan RGB; cv2 decodifica BGR

    detections: list[dict] = []
    start = time.perf_counter()
    if MODEL_CATALOG[key].backend == "ultralytics":
        results = loaded.model.predict(source=rgb, imgsz=IMGZ, conf=CONF, iou=IOU, device="cpu", verbose=False)
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            names = getattr(results[0], "names", {})
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                class_id = int(box.cls.item())
                detections.append({
                    "box": [v / IMGZ for v in xyxy],
                    "label": str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id),
                    "conf": round(float(box.conf.item()), 2),
                })
    else:  # backend torchvision
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        with torch.inference_mode():
            outputs = loaded.model([tensor])
        out0 = outputs[0]
        from torchvision.ops import nms

        keep = nms(out0["boxes"], out0["scores"], IOU)
        names = loaded.class_names
        for i in keep.tolist():
            score = float(out0["scores"][i])
            if score < CONF:
                continue
            class_id = int(out0["labels"][i])
            label = names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, str(class_id))
            detections.append({
                "box": [v / IMGZ for v in out0["boxes"][i].tolist()],
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


inject_css()

# --- Barra lateral ------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="side-label">Experimento</div>', unsafe_allow_html=True)
    route = st.radio(
        "Ruta de comparación",
        AVAILABLE_ROUTES,
        index=AVAILABLE_ROUTES.index("Comparación CNN vs YOLO") if "Comparación CNN vs YOLO" in AVAILABLE_ROUTES else 0,
        captions=[ROUTES[name]["caption"] for name in AVAILABLE_ROUTES],
        label_visibility="collapsed",
        help="Define la historia del experimento; las opciones no disponibles en este entorno están ocultas.",
    )

    streaming_mode = False
    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Entrada</div>', unsafe_allow_html=True)
    source_kind = st.selectbox(
        "Fuente de frames",
        SOURCE_OPTIONS,
        index=SOURCE_OPTIONS.index(DEFAULT_SOURCE) if route == "YOLO en vivo" and CAPABILITIES.webcam else SOURCE_OPTIONS.index("Demo persona/perro/fruta"),
        label_visibility="collapsed",
        help="De dónde salen los frames que se miden.",
    )
    if CAPABILITIES.webcam and source_kind == "Webcam local (OpenCV)":
        if CAPABILITIES.streaming and route == "YOLO en vivo":
            streaming_mode = st.checkbox("Modo en vivo", value=True, help="Procesa los frames de la cámara en tiempo real.")
        st.number_input("Índice de cámara", min_value=0, max_value=5, value=0)

    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Ejecución</div>', unsafe_allow_html=True)
    st.selectbox("Dispositivo", list(CAPABILITIES.device_options), label_visibility="collapsed", help="Dónde se ejecutan los modelos.")
    with st.expander("Avanzado"):
        st.select_slider("Resolución", options=[320, 416, 512, 640], value=416, help="Más resolución procesa más píxeles: n = H × W crece.")
        st.number_input("Frames de calentamiento", min_value=0, max_value=30, value=3, help="Estabilizan cachés antes de medir; no se reportan.")
        st.number_input("Frames medidos", min_value=1, max_value=300, value=20, help="Los frames que entran en latencia y FPS.")
        st.slider("Confianza mínima", 0.05, 0.95, 0.25, 0.05, help="Descarta detecciones débiles.")
        st.slider("IoU para NMS", 0.10, 0.95, 0.45, 0.05, help="Umbral para decidir si dos cajas son el mismo objeto.")
        st.checkbox("Estimar GFLOPs", value=True, help="Pasada extra para contar operaciones; puede demorar.")

    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    environment_note()

# --- Encabezado ---------------------------------------------------------------

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

st.markdown('<div class="run-note">Prototipo · medición local del 2026-09-05</div>', unsafe_allow_html=True)

# --- Entrada interactiva: elegís el modelo, montás tu foto -------------------

section_head("Entrada", dim="416 × 416 · conf 0,25 · IoU 0,45")
overlay = st.pills(
    "Modelo",
    ["YOLO11n", "SSDlite", "Faster R-CNN", "Sin cajas"],
    selection_mode="single",
    default="YOLO11n",
    label_visibility="collapsed",
)

uploaded = st.file_uploader(
    "Montá tu propia foto",
    type=["jpg", "jpeg", "png", "webp"],
    help="Se procesa en vivo con el modelo elegido y la misma configuración.",
)

upload_error = None
if uploaded is not None:
    with st.spinner(f"Corriendo {overlay}…"):
        try:
            live_dets, live_ms = detect_live(overlay, uploaded.getvalue())
            array = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
            frame = cv2.cvtColor(cv2.imdecode(array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGR)
            scale = min(1.0, 1000 / frame.shape[1])
            if scale < 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            if overlay != "Sin cajas":
                frame = draw_detections(frame, live_dets, MODEL_COLOR[overlay])
            png_b64 = encode_png(frame)
            source_note = "tu foto"
        except Exception as exc:  # pesos, memoria o formato
            upload_error = str(exc)
            png_b64 = demo_variant(overlay)
            source_note = "demo"
            live_dets, live_ms = [], None
else:
    png_b64 = demo_variant(overlay)
    source_note = "demo"
    live_dets, live_ms = [], None

if upload_error:
    st.warning(f"No se pudo procesar tu foto: {upload_error[:120]}. Mostrando la demo.")

frame_color = "var(--hair)" if overlay == "Sin cajas" else MODEL_COLOR[overlay]
st.markdown(
    f'<div class="input-frame" style="border-color:{frame_color}">'
    f'<img class="input-img" src="data:image/png;base64,{png_b64}" alt="Entrada con detecciones de {overlay}"/></div>',
    unsafe_allow_html=True,
)

if overlay != "Sin cajas":
    dot = MODEL_COLOR[overlay]
    if uploaded is not None and live_ms is not None:
        badge = (
            f'<div class="input-badge"><span class="dot" style="background:{dot}"></span>'
            f"<strong>{overlay}</strong> · {detections_summary(live_dets)} · <strong>{es_num(live_ms)} ms</strong> en CPU</div>"
        )
    else:
        demo_dets = DEMO_DETECTIONS[overlay]
        badge = (
            f'<div class="input-badge"><span class="dot" style="background:{dot}"></span>'
            f"<strong>{overlay}</strong> · {detections_summary(demo_dets)}</div>"
        )
    st.markdown(badge, unsafe_allow_html=True)

# Acción primaria: grande, ancha, inmediatamente debajo de la imagen
st.button("Ejecutar benchmark", type="primary", use_container_width=True)

# --- Medición -----------------------------------------------------------------

section_head("Medición")
kpi_strip(
    [
        ("Latencia media", f"{es_num(38.2)} ms", "YOLO11n · 50% menos que el siguiente", "#FF6600"),
        ("FPS efectivo", es_num(26.2), "YOLO11n · 2,0× el two-stage", "#2563eb"),
        ("Menor costo", f"{es_num(1.17, 2)} G", "SSDlite · 57% menos que el siguiente", "#10b981"),
        ("Entrada n = H×W", f"{es_num(173056, 0)} px", "416 × 416, igual para todos", "#7c3aed"),
    ]
)
st.markdown(
    '<div class="conclusion-panel">'
    "<strong>YOLO11n</strong> es el más rápido con esta entrada: 38,2 ms por frame y <strong>26,2 FPS</strong>, "
    "el doble del baseline two-stage. <strong>SSDlite</strong> es el más barato de correr, con 1,17 G. "
    "Las conclusiones están acotadas a esta entrada, dispositivo y configuración.</div>",
    unsafe_allow_html=True,
)

# --- Resultados ---------------------------------------------------------------

section_head("Resultados", dim="ordenados por latencia · en azul, el mejor de cada columna")
min_table(
    ["Modelo", "Familia", "p50 (ms)", "p95 (ms)", "FPS", "GFLOPs", "Parámetros (M)", "Tamaño (MB)"],
    [
        [r.model, r.family, es_num(r.p50), es_num(r.p95), es_num(r.fps), es_num(r.gflops, 2), es_num(r.params_m, 2), es_num(r.size_mb)]
        for r in MODELS.itertuples()
    ],
    numeric_cols={2, 3, 4, 5, 6, 7},
    best_cols={2: BEST["p50"], 3: BEST["p95"], 4: BEST["fps"], 5: BEST["gflops"], 6: BEST["params_m"], 7: BEST["size_mb"]},
)

# --- Gráficas -----------------------------------------------------------------

section_head("Tiempos", dim="por modelo · misma entrada y configuración")
left, right = st.columns(2)
with left:
    st.markdown('<div class="chart-label">Latencia media <span class="dim">· menor es mejor</span></div>', unsafe_allow_html=True)
    st.plotly_chart(hbar_chart("p50", lambda v: f"{es_num(v)} ms", higher_better=False, with_legend=True), width="stretch", config={"displayModeBar": False})
with right:
    st.markdown('<div class="chart-label">FPS efectivo <span class="dim">· mayor es mejor</span></div>', unsafe_allow_html=True)
    st.plotly_chart(hbar_chart("fps", lambda v: f"{es_num(v)} FPS", higher_better=True, with_legend=False), width="stretch", config={"displayModeBar": False})

section_head("Costo vs latencia", dim="burbuja = parámetros · arriba a la izquierda es mejor")
st.plotly_chart(scatter_chart(), width="stretch", config={"displayModeBar": False})

# --- Reconocimiento -----------------------------------------------------------

section_head("Reconocimiento", dim="de la imagen demo · no reemplaza mAP")
min_table(
    ["Modelo", "Detectado", "Confianza media"],
    [[r.model, r.detected, es_num(r.conf, 2)] for r in MODELS.itertuples()],
    numeric_cols={2},
)

with st.expander("Configuración de la corrida"):
    st.caption("Calentamiento: 3 frames · medidos: 20 · GFLOPs activado · dispositivo: cpu · torch 2.12.0+cpu")

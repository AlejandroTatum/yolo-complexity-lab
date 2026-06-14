from __future__ import annotations

import base64
import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import streamlit as st

from yolo_complexity_lab.benchmark import BenchmarkConfig, benchmark_model, run_frame
from yolo_complexity_lab.catalog import MODEL_CATALOG, catalog_rows
from yolo_complexity_lab.exporting import write_results_csv
from yolo_complexity_lab.loaders import load_model
from yolo_complexity_lab.paths import default_export_dir
from yolo_complexity_lab.sources import (
    read_image_file,
    repeat_frame,
    sample_coco_frame,
)
from yolo_complexity_lab.system_info import system_info_dict

st.set_page_config(
    page_title="YOLO Complexity Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Opciones de fuente: en deploy, solo Demo e Imagen (no webcam)
SOURCE_HELP = {
    "Demo persona/perro/fruta": "Usa una lámina local con una persona, un perro y una banana para comparar reconocimiento y falsos positivos.",
    "Subir imagen": "Repite una imagen propia varias veces para medir latencia sin depender de un video.",
}

# Opciones de ruta: en deploy, solo comparación (no YOLO en vivo)
PRESET_MODELS = {
    "Comparación CNN vs YOLO": [
        "fasterrcnn_mobilenet_fpn",
        "ssdlite_mobilenet_v3",
        "yolo11n",
    ],
}

PRESET_HELP = {
    "Comparación CNN vs YOLO": "Comparar dos etapas, one-stage CNN y YOLO para probar tiempo y complejidad.",
}

DEVICE_HELP = {
    "auto": "Usa GPU si PyTorch detecta CUDA; si no, usa CPU.",
    "cpu": "Fuerza ejecución en procesador. Más comparable entre máquinas, pero más lento.",
    "cuda:0": "Fuerza la primera GPU NVIDIA disponible. Si no existe, el loader cae a CPU.",
}

METRIC_EXPLANATIONS = {
    "Latencia": "Tiempo que tarda el modelo en procesar un frame. Menor es mejor.",
    "FPS": "Frames por segundo efectivos. Mayor es mejor para tiempo real.",
    "GFLOPs": "Operaciones aproximadas por frame (1 MAC ≈ 2 FLOPs). Sirve como proxy de complejidad computacional.",
    "Parámetros": "Cantidad de pesos aprendidos. Afecta tamaño, memoria y capacidad del modelo.",
    "Big-O": "Describe cómo crece el costo cuando suben resolución, capas, canales o cajas candidatas.",
}


def get_default_device() -> str:
    """Detecta si hay GPU disponible y retorna el dispositivo apropiado."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "auto"


@st.cache_resource(show_spinner=False)
def cached_load_model(spec_key: str, device: str):
    return load_model(spec_key, device)


def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --page: #f6f8fb;
  --surface: #ffffff;
  --surface-soft: #eef5ff;
  --ink: #172033;
  --muted: #64748b;
  --line: #d8e0ec;
  --blue: #2563eb;
  --cyan: #0891b2;
  --green: #059669;
  --amber: #d97706;
  --violet: #7c3aed;
  --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}

.stApp {
  background:
    radial-gradient(circle at 12% 0%, rgba(37, 99, 235, 0.10), transparent 28%),
    linear-gradient(180deg, #f8fbff 0%, var(--page) 58%, #eef3f9 100%);
  color: var(--ink);
}

.block-container {
  padding-top: 1.35rem;
  padding-bottom: 2.4rem;
  max-width: 1240px;
}

[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--line);
  box-shadow: 10px 0 30px rgba(15, 23, 42, 0.05);
}

[data-testid="stSidebar"] * {
  color: var(--ink) !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: var(--muted);
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  background: #f8fafc !important;
  color: var(--ink) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
  color: var(--ink) !important;
}

[data-baseweb="radio"] > div {
  gap: 0.35rem;
}

h1, h2, h3 {
  color: var(--ink) !important;
  letter-spacing: -0.035em;
}

p, li, span, label {
  color: var(--ink) !important;
}

/* Asegurar que todo texto nativo de Streamlit sea oscuro sobre fondo claro */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
.stApp p,
.stApp li,
.stApp span,
.stApp label,
.stApp .stTextInput label,
.stApp .stNumberInput label,
.stApp .stSlider label,
.stApp .stSelectbox label,
.stApp .stMultiselect label,
.stApp .stRadio label,
.stApp .stCheckbox label,
.stApp .stTextArea label {
  color: var(--ink) !important;
}

.stApp .stCaption {
  color: var(--muted) !important;
}

.stApp .stInfo,
.stApp .stWarning,
.stApp .stSuccess,
.stApp .stError {
  color: var(--ink) !important;
}

.stApp .stInfo p,
.stApp .stWarning p,
.stApp .stSuccess p,
.stApp .stError p {
  color: var(--ink) !important;
}

.hero {
  padding: 1.35rem 1.55rem;
  border: 1px solid #cfe0f5;
  border-radius: 26px;
  background: linear-gradient(135deg, #ffffff 0%, #f1f7ff 100%);
  box-shadow: var(--shadow);
  margin-bottom: 0.9rem;
}

.hero-kicker {
  color: var(--blue);
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}

.hero-title {
  font-size: clamp(2rem, 4.2vw, 3.7rem);
  line-height: 0.98;
  font-weight: 850;
  color: #111827;
  margin-bottom: 0.45rem;
}

.hero-subtitle {
  max-width: 860px;
  color: #475569;
  font-size: 1.05rem;
  line-height: 1.45;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.95rem;
}

.pill {
  border: 1px solid #bfdbfe;
  color: #1d4ed8;
  background: #eff6ff;
  padding: 0.42rem 0.68rem;
  border-radius: 999px;
  font-size: 0.84rem;
  font-weight: 750;
}

.glass-card {
  border: 1px solid var(--line);
  border-radius: 20px;
  background: var(--surface);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
  padding: 1rem 1.05rem;
  height: 100%;
}

.glass-card h3 {
  margin: 0 0 0.35rem 0;
  font-size: 1rem;
  color: var(--ink);
}

.glass-card p {
  color: var(--muted);
  font-size: 0.93rem;
  line-height: 1.5;
  margin: 0;
}

.card-accent-blue { border-left: 4px solid var(--blue); }
.card-accent-green { border-left: 4px solid var(--green); }
.card-accent-amber { border-left: 4px solid var(--amber); }
.card-accent-violet { border-left: 4px solid var(--violet); }

.small-label {
  display: inline-block;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--blue);
  margin-bottom: 0.45rem;
  font-weight: 800;
}

.metric-note {
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.45;
}

.section-title {
  margin: 1rem 0 0.5rem 0;
  color: var(--ink);
  font-size: clamp(1.35rem, 2.3vw, 2rem);
}

[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: #ffffff;
  padding: 0.95rem;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
  color: var(--ink) !important;
}

.stButton > button,
.stDownloadButton > button {
  width: auto;
  border: 1px solid #2563eb;
  border-radius: 999px;
  background: linear-gradient(135deg, #2563eb, #0891b2);
  color: white;
  font-weight: 800;
  min-height: 2.75rem;
  padding: 0 1.25rem;
  box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
  transition: all 150ms ease;
}

.stButton > button *,
.stDownloadButton > button * {
  color: #ffffff !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: #1d4ed8;
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(37, 99, 235, 0.26);
}

[data-testid="stTabs"] button {
  border-radius: 999px;
  color: var(--muted);
}

[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--blue);
  background: #eff6ff;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  overflow: hidden;
  background: white;
}

code, pre {
  color: #1d4ed8 !important;
  background: #eff6ff !important;
}

hr {
  border-color: var(--line);
}

[data-testid="stDecoration"],
.stDeployButton,
[data-testid="stToolbar"] {
  display: none !important;
}

[data-testid="stHeader"] {
  background: transparent;
}

/* Ocultar botón de toggle de la sidebar para mantenerla siempre fija */
[data-testid="stSidebarNav"] {
  display: none !important;
}

[data-testid="stSidebar"] {
  transform: none !important;
  transition: none !important;
}

.preview-frame {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: #ffffff;
  padding: 0.65rem;
  box-shadow: var(--shadow);
}

.preview-frame img {
  width: 100%;
  max-height: 340px;
  object-fit: contain;
  display: block;
  border-radius: 16px;
  background: #f8fafc;
}

.preview-caption {
  text-align: center;
  color: var(--muted);
  font-size: 0.82rem;
  margin-top: 0.45rem;
}

div[data-testid="stAlert"] {
  border-radius: 16px;
}

@media (max-width: 760px) {
  .block-container { padding-top: 0.8rem; }
  .hero { padding: 1rem; border-radius: 20px; }
  .hero-title { font-size: 2.1rem; }
  .hero-subtitle { font-size: 0.92rem; }
  .hero-actions { gap: 0.38rem; }
  .pill { font-size: 0.72rem; padding: 0.34rem 0.55rem; }
  .preview-frame img { max-height: 260px; }
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
            "Faltan dependencias para ejecutar inferencia: "
            + ", ".join(missing)
            + ". Instala con `python -m pip install -r requirements.txt` dentro del entorno `.venv`."
        )


def render_hero(presentation_mode: bool = False) -> None:
    subtitle = (
        "Evolución de detectores, reconocimiento y tiempo real en una demo local."
        if presentation_mode
        else "Compara detectores antiguos y YOLO por reconocimiento, latencia, FPS y costo computacional."
    )
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-kicker">Laboratorio local de complejidad computacional</div>
  <div class="hero-title">YOLO Complexity Lab</div>
  <div class="hero-subtitle">
    {subtitle}
  </div>
  <div class="hero-actions">
    <span class="pill">Latencia por frame</span>
    <span class="pill">FPS efectivo</span>
    <span class="pill">Qué reconoce</span>
    <span class="pill">MACs y GFLOPs</span>
    <span class="pill">Big-O por modelo</span>
    <span class="pill">CPU o GPU local</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
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
        uploaded = st.file_uploader(
            "Archivo de imagen",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
            help="La app repetirá esta imagen para medir varios frames con el mismo input.",
        )
        if uploaded is None:
            st.info("Sube una imagen o cambia a 'Demo persona/perro/fruta' para una prueba rápida.")
            return [], None
        frame = read_image_file(uploaded)
        return repeat_frame(frame, total_needed), frame

    frame = sample_coco_frame()
    return repeat_frame(frame, total_needed), frame


def render_preview_image(frame: object, caption: str = "Vista previa del input") -> None:
    """Render a compact full-image preview with object-fit containment."""
    try:
        import cv2
        import numpy as np

        # Convertir RGB a BGR para cv2.imencode (OpenCV usa BGR por defecto)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        success, encoded = cv2.imencode(".png", frame_bgr)
        if not success:
            raise ValueError("No se pudo codificar la imagen de preview.")
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        st.markdown(
            f"""
<div class="preview-frame">
  <img src="data:image/png;base64,{b64}" alt="{caption}" />
  <div class="preview-caption">{caption}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.image(frame, caption=caption, channels="RGB", width="stretch")


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
            "recognized_classes": "Qué reconoció",
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
        cost_value = f"{lowest_cost['gflops_approx']} G"
        cost_delta = lowest_cost["model"]
    else:
        lowest_cost = df.sort_values("parameters_millions", na_position="last").iloc[0]
        cost_label = "Menos parámetros"
        cost_value = f"{lowest_cost['parameters_millions']} M"
        cost_delta = lowest_cost["model"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Menor latencia media", f"{fastest['latency_mean_ms']} ms", fastest["model"])
    c2.metric("Mayor FPS efectivo", f"{most_fps['fps_effective']} FPS", most_fps["model"])
    c3.metric(cost_label, cost_value, cost_delta)
    st.markdown(
        "<p class='metric-note'>Estas tarjetas resumen el resultado medido en tu hardware local. No son valores universales: cambian con CPU, GPU, resolución y número de frames.</p>",
        unsafe_allow_html=True,
    )


def plot_results(df: pd.DataFrame) -> list[tuple[str, object]]:
    plots = []
    if df.empty:
        return plots

    color_map = {
        "YOLO": "#7dd3fc",
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
            title="Complejidad computacional vs tiempo real",
            labels={
                "gflops_approx": "GFLOPs aproximados",
                "latency_mean_ms": "Latencia media (ms)",
                "parameters_millions": "Parámetros (M)",
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
            "1. YOLO actual",
            "Arrancá con webcam: una pasada por frame para mostrar tiempo real.",
            "blue",
        )
    with c2:
        render_card(
            "2. Comparación",
            "Luego medí Faster R-CNN, SSDlite y YOLO con la misma imagen.",
            "green",
        )
    with c3:
        render_card(
            "3. Conclusión",
            "Usá latencia, FPS y n = H×W para explicar complejidad.",
            "violet",
        )


def render_theory_bridge() -> None:
    st.markdown("<h2 class='section-title'>De la teoría a la demo</h2>", unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        (
            "Problema",
            "Detectar implica localizar y clasificar varios objetos. Los pipelines antiguos repetían trabajo y aumentaban latencia.",
            "amber",
        ),
        (
            "Idea YOLO",
            "Convertir detección en una regresión única: una pasada hacia adelante sobre la imagen completa.",
            "blue",
        ),
        (
            "Costo dominante",
            "Las convoluciones explican el crecimiento principal: resolución, capas, canales y kernel elevan operaciones.",
            "violet",
        ),
        (
            "Evidencia local",
            "La app cruza Big-O con GFLOPs, parámetros, RAM, latencia, FPS y clases reconocidas.",
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
        ("Tiempo", "Latencia media/p95 y FPS efectivo."),
        ("Complejidad", "n = H×W, GFLOPs y parámetros."),
        ("Reconocimiento", "Sirve como apoyo visual, no como mAP formal."),
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
    model_names = ", ".join(MODEL_CATALOG[key].display_name for key in selected_models) if selected_models else "Sin modelos"
    if presentation_mode:
        st.caption(f"Modelos: {model_names} | Fuente: {source_kind} | Dispositivo: {device} | {imgsz}×{imgsz} | Warmup: {warmup_frames} | Medidos: {measure_frames}")
    else:
        st.markdown(
            f"""
<div class="glass-card card-accent-violet">
  <span class="small-label">Configuración actual</span>
  <p>Modelos: {model_names} | Fuente: {source_kind} | Dispositivo: {device} | {imgsz}×{imgsz} | Warmup: {warmup_frames} | Medidos: {measure_frames}</p>
</div>
            """,
            unsafe_allow_html=True,
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
        complexity_sentence = "No se calculó GFLOPs en esta corrida."

    theory_sentence = (
        "Con el mismo n = H×W, la evidencia principal es latencia/FPS: "
        "si YOLO procesa más rápido, se observa el beneficio de la pasada única y las optimizaciones de arquitectura."
    )

    if presentation_mode:
        bullets = [
            f"<li>Menor latencia: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms por frame.</li>",
            f"<li>Mayor FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</li>",
            f"<li>Más detecciones en esta entrada: <strong>{most_detections['model']}</strong> — {most_detections.get('recognized_classes', 'sin detalle')}.</li>",
            f"<li>{complexity_sentence} GFLOPs es proxy; tiempo real se decide con latencia/FPS.</li>",
            f"<li>{theory_sentence}</li>",
        ]
        body = "<ul>" + "".join(bullets) + "</ul>"
    else:
        body = (
            f"<p>Menor latencia: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms por frame.</p>\n"
            f"<p>Mayor FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</p>\n"
            f"<p>Más detecciones en esta entrada: <strong>{most_detections['model']}</strong> — {most_detections.get('recognized_classes', 'sin detalle')}.</p>\n"
            f"<p>{complexity_sentence} GFLOPs es proxy; tiempo real se decide con latencia/FPS.</p>\n"
            f"<p>{theory_sentence}</p>"
        )

    st.markdown(
        f"""
<div class="glass-card card-accent-green">
  <span class="small-label">Interpretación para explicar en clase</span>
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
        "Esto no reemplaza mAP: es una lectura cualitativa de la imagen usada en el benchmark. "
        "Sirve para explicar falsos negativos, detecciones débiles y diferencias entre familias."
    )
    cols = st.columns(min(3, len(df)))
    for col, (_, row) in zip(cols, df.iterrows(), strict=False):
        with col:
            confidence = row.get("avg_confidence")
            confidence_text = "—" if pd.isna(confidence) else f"{float(confidence):.2f}"
            render_card(
                str(row.get("model", "Modelo")),
                (
                    f"<strong>Detectó:</strong> {row.get('recognized_classes', 'Sin detecciones')}<br>"
                    f"<strong>Principal:</strong> {row.get('top_detection', '—')}<br>"
                    f"<strong>Confianza media:</strong> {confidence_text}"
                ),
                "green" if row.get("family") == "CNN one-stage" else "blue" if row.get("family") == "YOLO" else "amber",
            )


def render_benchmark_results(df: pd.DataFrame, csv_path: str | None = None, presentation_mode: bool = False) -> None:
    """Render persisted benchmark results and export actions.

    Streamlit reruns the script whenever a button, checkbox or download action is
    used. Keeping rendering in this helper lets us show the last benchmark from
    st.session_state instead of losing it after export actions.
    """
    if df.empty:
        return

    metric_cards(df, presentation_mode)
    render_result_interpretation(df, presentation_mode)
    render_detection_summary(df)

    st.markdown("<h3 class='section-title'>Resumen comparativo</h3>", unsafe_allow_html=True)
    st.dataframe(compact_results_table(df), width="stretch", hide_index=True)

    with st.expander("Ver tabla técnica completa"):
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("<h3 class='section-title'>Gráficos</h3>", unsafe_allow_html=True)
    plots = plot_results(df)

    if csv_path:
        csv_name = Path(csv_path).name
        st.success(f"CSV generado: {csv_name}")
        st.caption(f"Ruta local: {csv_path}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar resultados CSV",
        csv_bytes,
        file_name="benchmark_yolo_complexity.csv",
        mime="text/csv",
        key="download_results_csv",
        on_click="ignore",
    )

    # Exportación HTML removida a pedido del usuario

    # --- Vista de detecciones: mostrar el último frame anotado por cada modelo ---
    st.markdown("<h3 class='section-title'>Detección visual (último frame medido)</h3>", unsafe_allow_html=True)
    st.caption("Estas imágenes muestran qué objetos detectó cada modelo en el último frame medido. Sirve para analizar precisión y falsos positivos.")
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
    else:
        st.info("No hay imágenes anotadas disponibles. Ejecuta el benchmark con modelos YOLO para generarlas.")


def render_controls_guide() -> None:
    st.markdown("<h2 class='section-title'>Guía de controles</h2>", unsafe_allow_html=True)
    st.write(
        "Esta sección explica para qué sirve cada opción del panel lateral y cómo cambia la lectura del benchmark."
    )

    rows = [
        ("Ruta de comparación", "Elige la historia del experimento: antiguo → YOLO, escalado YOLO, demo gestos o selección manual."),
        ("Modelos a comparar", "En modo personalizado elige detectores manualmente. Para clase, compara al menos un baseline antiguo con YOLO."),
        ("Fuente de frames", "Define de dónde salen las imágenes. La demo persona/perro/fruta sirve para reconocimiento; la sintética solo para tiempo."),
        ("Dispositivo", "Permite medir en CPU o GPU. Sirve para separar el diseño del modelo del hardware usado."),
        ("Perfil de prueba", "Configura valores iniciales. 'Rápida' valida que todo funcione; 'Presentación' es equilibrada; 'Completa' mide con más estabilidad."),
        ("Resolución", "Aumenta o reduce el tamaño de entrada. Sirve para ver cómo crece n = H × W en la complejidad."),
        ("Frames de calentamiento", "Se ejecutan antes de medir. Sirven para estabilizar caché, GPU y carga inicial."),
        ("Frames medidos", "Son los frames que sí entran en latencia y FPS. Más frames dan una medición más estable."),
        ("Confianza mínima", "Filtra detecciones débiles. Sirve para reducir falsas detecciones y puede bajar cajas candidatas."),
        ("IoU para NMS", "Controla cuándo dos cajas se consideran solapadas. Sirve para regular el postprocesamiento O(B²)."),
        ("MACs/GFLOPs", "Activa el cálculo aproximado de operaciones. Sirve para conectar la fórmula Big-O con una medida numérica."),
        ("Ejecutar benchmark", "Carga los modelos, mide los frames y genera tabla, gráficos y CSV."),
    ]

    for i in range(0, len(rows), 2):
        cols = st.columns(2)
        for col, item in zip(cols, rows[i:i + 2], strict=False):
            with col:
                render_card(item[0], item[1], "blue" if i % 4 == 0 else "green")


inject_css()
dependency_warning()
render_hero(True)

with st.sidebar:
    st.markdown("### Configuración del benchmark")
    
    comparison_route = st.radio(
        "Ruta de comparación",
        options=list(PRESET_MODELS.keys()),
        help="Comparar detectores por tiempo y precisión.",
    )
    st.caption(PRESET_HELP[comparison_route])

    selected_models = PRESET_MODELS[comparison_route]
    st.markdown("**Modelos de la ruta:**")
    for index, key in enumerate(selected_models, start=1):
        st.caption(f"{index}. {MODEL_CATALOG[key].display_name}")

    source_options = list(SOURCE_HELP.keys())
    default_source = "Demo persona/perro/fruta"
    source_kind = st.selectbox(
        "Fuente de frames",
        source_options,
        index=source_options.index(default_source),
        help="Define de dónde salen los frames usados en el benchmark.",
    )

    with st.expander("Configuración avanzada"):
        device = st.selectbox("Dispositivo de ejecución", list(DEVICE_HELP.keys()), help="Controla si se usa CPU o GPU.")

        imgsz = st.select_slider(
            "Resolución cuadrada",
            options=[320, 416, 512, 640],
            value=416,
            help="Mayor resolución procesa más píxeles. Eso sube el costo aproximado n = H × W.",
        )
        warmup_frames = st.number_input(
            "Frames de calentamiento",
            min_value=0,
            max_value=30,
            value=3,
            help="No se reportan. Sirven para estabilizar carga de modelo, cachés y GPU.",
        )
        measure_frames = st.number_input(
            "Frames medidos",
            min_value=1,
            max_value=300,
            value=20,
            help="Estos frames sí entran en latencia, FPS y estadísticas finales.",
        )
        confidence = st.slider(
            "Confianza mínima",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Sirve para aceptar solo detecciones con probabilidad suficiente. Más alto = menos cajas, pero podés perder objetos.",
        )
        iou = st.slider(
            "IoU para NMS",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05,
            help="Sirve para decidir cuándo dos cajas se solapan demasiado y deben fusionarse/eliminarse en NMS.",
        )
        include_complexity = st.checkbox(
            "Calcular MACs/GFLOPs aproximados",
            value=True,
            help="Activa un forward adicional para estimar operaciones de Conv2d y Linear. Puede tardar un poco más.",
        )

        pass

inicio_tab, benchmark_tab = st.tabs(
    ["Inicio", "Benchmark"]
)

with inicio_tab:
    st.markdown("<h2 class='section-title'>Ruta de comparación</h2>", unsafe_allow_html=True)
    render_model_overview()
    st.markdown("<h2 class='section-title'>Uso rápido</h2>", unsafe_allow_html=True)
    render_explanation_flow()
    st.write("")
    with st.expander("Ver glosario de métricas"):
        render_metric_glossary()

with benchmark_tab:
    st.markdown("<h2 class='section-title'>Benchmark de tiempo y complejidad</h2>", unsafe_allow_html=True)
    st.caption("¿Cuánto tarda por frame y cómo crece el costo cuando aumenta n = H×W?")
    
    total_needed = int(warmup_frames + measure_frames)
    frames_to_load = 1
    frames, preview = source_frames(source_kind, frames_to_load, imgsz)

    preview_col = st.columns(1)[0]
    with preview_col:
        if preview is not None:
            render_preview_image(preview, "Vista previa del input")
    
    st.write("")
    
    # Botón de ejecución debajo de la imagen
    run = st.button("Ejecutar comparación", type="primary")
    
    if run:
        # Recargar frames para el benchmark completo
        frames_to_load = total_needed
        frames, preview = source_frames(source_kind, frames_to_load, imgsz)

    if run:
        if not selected_models:
            st.error("Selecciona al menos un modelo para iniciar el benchmark.")
            st.stop()
        
        if not frames:
            st.error("No hay frames disponibles para medir. Revisa la fuente seleccionada.")
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
            status.markdown(f"## Cargando: Modelo {index} de {len(selected_models)}")
            try:
                loaded = cached_load_model(model_key, device)
                row = benchmark_model(loaded, frames, config, include_complexity=include_complexity)
                # Guardar frame anotado aparte (no va al DataFrame)
                annotated_frame = row.pop("last_annotated_frame", None)
                if annotated_frame is not None:
                    if "annotated_frames" not in st.session_state:
                        st.session_state["annotated_frames"] = {}
                    st.session_state["annotated_frames"][model_key] = annotated_frame
                rows.append(row)
            except Exception as exc:
                st.error(f"Falló {spec.display_name}: {exc}")
            progress.progress(index / len(selected_models))

        status.empty()
        progress.empty()

        if rows:
            df = pd.DataFrame(rows)
            export_path = write_results_csv(df)
            st.session_state["last_benchmark_df"] = df
            st.session_state["last_benchmark_csv_path"] = str(export_path)
            st.session_state.pop("last_html_zip", None)
            st.session_state.pop("last_html_paths", None)
            render_benchmark_results(df, str(export_path), True)
        else:
            st.warning("No se pudo medir ningún modelo. Revisa dependencias, conexión o disponibilidad de pesos.")
    elif "pending_streaming_results" in st.session_state:
        st.success("Streaming finalizado. Generando gráficas del rendimiento en vivo...")
        
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
        
        render_benchmark_results(df, str(export_path), True)

    elif "last_benchmark_df" in st.session_state:
        st.info("Mostrando el último benchmark ejecutado. Podés descargar CSV sin volver a medir.")
        render_benchmark_results(
            st.session_state["last_benchmark_df"],
            st.session_state.get("last_benchmark_csv_path"),
            True,
        )
    else:
        st.info("Ejecutá la comparación para generar tabla, gráficos y CSV.")

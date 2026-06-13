from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import streamlit as st

from yolo_complexity_lab.benchmark import BenchmarkConfig, benchmark_model
from yolo_complexity_lab.catalog import MODEL_CATALOG, catalog_rows
from yolo_complexity_lab.exporting import build_plot_html_zip, write_plot_html, write_results_csv
from yolo_complexity_lab.loaders import load_model
from yolo_complexity_lab.paths import default_export_dir
from yolo_complexity_lab.sources import (
    demo_frame,
    frames_from_video,
    frames_from_webcam,
    read_image_file,
    repeat_frame,
)
from yolo_complexity_lab.system_info import system_info_dict

st.set_page_config(
    page_title="YOLO Complexity Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

SOURCE_HELP = {
    "Imagen demo": "Usa una imagen generada por la app. Es la opción más estable para comparar modelos rápidamente.",
    "Subir imagen": "Repite una imagen propia varias veces para medir latencia sin depender de un video.",
    "Subir video": "Extrae frames de un video y mide el rendimiento sobre una secuencia más realista.",
    "Webcam OpenCV local": "Captura frames desde la cámara local. Útil para demo en vivo, pero depende de la cámara y luz.",
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


@st.cache_resource(show_spinner=False)
def cached_load_model(spec_key: str, device: str):
    return load_model(spec_key, device)


def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --bg-0: #0f172a;
  --bg-1: #0b1220;
  --glass: rgba(15, 23, 42, 0.72);
  --glass-soft: rgba(255, 255, 255, 0.055);
  --border: rgba(148, 163, 184, 0.22);
  --border-strong: rgba(125, 211, 252, 0.42);
  --text: #e5edf7;
  --muted: #9ca9ba;
  --blue: #7dd3fc;
  --cyan: #22d3ee;
  --violet: #a78bfa;
  --green: #34d399;
  --amber: #fbbf24;
  --red: #fb7185;
}

.stApp {
  background:
    radial-gradient(circle at 15% 15%, rgba(34, 211, 238, 0.13), transparent 32%),
    radial-gradient(circle at 85% 0%, rgba(167, 139, 250, 0.14), transparent 34%),
    linear-gradient(145deg, var(--bg-0) 0%, var(--bg-1) 58%, #050816 100%);
  color: var(--text);
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1380px;
}

[data-testid="stSidebar"] {
  background: rgba(3, 7, 18, 0.82);
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
  color: var(--muted);
}

h1, h2, h3 {
  letter-spacing: -0.035em;
}

.hero {
  padding: 2rem;
  border: 1px solid var(--border-strong);
  border-radius: 28px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(15, 23, 42, 0.44));
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255,255,255,0.08);
  backdrop-filter: blur(18px);
  margin-bottom: 1.15rem;
}

.hero-kicker {
  color: var(--blue);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
}

.hero-title {
  font-size: clamp(2.05rem, 5vw, 4.4rem);
  line-height: 0.98;
  font-weight: 850;
  color: #f8fafc;
  margin-bottom: 0.85rem;
}

.hero-subtitle {
  max-width: 860px;
  color: #b6c2d2;
  font-size: 1.4rem;
  line-height: 1.65;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  margin-top: 1.25rem;
}

.pill {
  border: 1px solid rgba(125, 211, 252, 0.35);
  color: #dff7ff;
  background: rgba(14, 165, 233, 0.10);
  padding: 0.48rem 0.8rem;
  border-radius: 999px;
  font-size: 1rem;
}

.glass-card {
  border: 1px solid var(--border);
  border-radius: 22px;
  background: var(--glass);
  box-shadow: 0 16px 48px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.07);
  backdrop-filter: blur(14px);
  padding: 1.1rem 1.15rem;
  height: 100%;
}

.glass-card h3 {
  margin: 0 0 0.35rem 0;
  font-size: 1.05rem;
  color: #f8fafc;
}

.glass-card p {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.55;
  margin: 0;
}

.card-accent-blue { border-top: 3px solid var(--blue); }
.card-accent-green { border-top: 3px solid var(--green); }
.card-accent-amber { border-top: 3px solid var(--amber); }
.card-accent-violet { border-top: 3px solid var(--violet); }

.small-label {
  display: inline-block;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--blue);
  margin-bottom: 0.45rem;
  font-weight: 750;
}

.metric-note {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.55;
}

.section-title {
  margin: 1.2rem 0 0.55rem 0;
  color: #f8fafc;
}

[data-testid="stMetric"] {
  border: 1px solid var(--border);
  border-radius: 20px;
  background: rgba(255,255,255,0.045);
  padding: 1rem;
  box-shadow: 0 12px 34px rgba(0,0,0,0.22);
}

.stButton > button {
  width: 100%;
  border: 1px solid rgba(125, 211, 252, 0.42);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(125, 211, 252, 0.20), rgba(167, 139, 250, 0.14));
  color: #eef9ff;
  font-weight: 800;
  min-height: 3.05rem;
  box-shadow: 0 16px 42px rgba(34, 211, 238, 0.16), inset 0 1px 0 rgba(255,255,255,0.13);
  backdrop-filter: blur(16px);
  transition: all 160ms ease;
}

.stButton > button:hover {
  border-color: rgba(125, 211, 252, 0.78);
  transform: translateY(-1px);
  box-shadow: 0 22px 54px rgba(34, 211, 238, 0.23), inset 0 1px 0 rgba(255,255,255,0.18);
}

.stButton > button:active {
  transform: translateY(0px) scale(0.99);
}

[data-testid="stTabs"] button {
  border-radius: 999px;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow: hidden;
}

code {
  color: #bae6fd !important;
  font-size: 1.05rem;
}

hr {
  border-color: rgba(148, 163, 184, 0.18);
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
        "Compara detectores YOLO en tu hardware local."
        if presentation_mode
        else "Compara detectores YOLO por latencia, FPS y costo computacional en tu hardware local."
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
    if source_kind == "Imagen demo":
        frame = demo_frame(imgsz)
        return repeat_frame(frame, total_needed), frame

    if source_kind == "Subir imagen":
        uploaded = st.file_uploader(
            "Archivo de imagen",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
            help="La app repetirá esta imagen para medir varios frames con el mismo input.",
        )
        if uploaded is None:
            st.info("Sube una imagen o cambia a 'Imagen demo' para una prueba rápida.")
            return [], None
        frame = read_image_file(uploaded)
        return repeat_frame(frame, total_needed), frame

    if source_kind == "Subir video":
        uploaded_video = st.file_uploader(
            "Archivo de video",
            type=["mp4", "mov", "avi", "mkv"],
            key="video_upload",
            help="La app extrae frames del video hasta completar el benchmark configurado.",
        )
        if uploaded_video is None:
            st.info("Sube un video para medir frames reales o cambia a 'Imagen demo'.")
            return [], None
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = Path(tmp.name)
        frames = frames_from_video(tmp_path, limit=total_needed, stride=1)
        preview = frames[0] if frames else None
        return frames, preview

    if source_kind == "Webcam OpenCV local":
        camera_index = int(st.session_state.get("camera_index", 0))
        frames = frames_from_webcam(camera_index, limit=total_needed)
        preview = frames[0] if frames else None
        return frames, preview

    frame = demo_frame(imgsz)
    return repeat_frame(frame, total_needed), frame


def metric_cards(df: pd.DataFrame, presentation_mode: bool = False) -> None:
    if df.empty:
        return
    if presentation_mode:
        st.markdown(
            "<style>[data-testid=\"stMetric\"] { padding: 1.8rem; }</style>",
            unsafe_allow_html=True,
        )
    fastest = df.sort_values("latency_mean_ms").iloc[0]
    lightest = df.sort_values("parameters_millions", na_position="last").iloc[0]
    most_fps = df.sort_values("fps_effective", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Menor latencia media", f"{fastest['latency_mean_ms']} ms", fastest["model"])
    c2.metric("Mayor FPS efectivo", f"{most_fps['fps_effective']} FPS", most_fps["model"])
    c3.metric("Menos parámetros", f"{lightest['parameters_millions']} M", lightest["model"])
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
    template = "plotly_dark"

    latency_fig = px.bar(
        df,
        x="model",
        y="latency_mean_ms",
        color="family",
        color_discrete_map=color_map,
        template=template,
        title="Latencia por modelo",
        labels={"latency_mean_ms": "Latencia media (ms)", "model": "Modelo"},
    )
    latency_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-18)
    st.plotly_chart(latency_fig, width="stretch")
    plots.append(("latencia_media", latency_fig))

    fps_fig = px.bar(
        df,
        x="model",
        y="fps_effective",
        color="family",
        color_discrete_map=color_map,
        template=template,
        title="FPS efectivo",
        labels={"fps_effective": "FPS", "model": "Modelo"},
    )
    fps_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-18)
    st.plotly_chart(fps_fig, width="stretch")
    plots.append(("fps_efectivo", fps_fig))

    complexity_fig = px.scatter(
        df,
        x="gflops_approx",
        y="latency_mean_ms",
        size="parameters_millions",
        color="family",
        color_discrete_map=color_map,
        hover_name="model",
        template=template,
        title="Complejidad vs tiempo real",
        labels={"gflops_approx": "GFLOPs aproximados", "latency_mean_ms": "Latencia media (ms)"},
    )
    complexity_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(complexity_fig, width="stretch")
    plots.append(("gflops_vs_latencia", complexity_fig))
    return plots


def render_model_overview() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card(
            "YOLO",
            "Detector de una etapa. Predice cajas y clases en una sola pasada; por eso suele tener baja latencia.",
            "blue",
        )
    with c2:
        render_card(
            "SSDlite",
            "También es one-stage, pero usa una familia CNN ligera basada en MobileNet. Sirve como comparación rápida.",
            "green",
        )
    with c3:
        render_card(
            "Faster R-CNN",
            "Detector de dos etapas. Agrega propuestas de regiones, lo que suele aumentar costo y latencia.",
            "amber",
        )


def render_metric_glossary() -> None:
    cols = st.columns(len(METRIC_EXPLANATIONS))
    accents = ["blue", "green", "violet", "amber", "blue"]
    for col, (name, description), accent in zip(cols, METRIC_EXPLANATIONS.items(), accents, strict=False):
        with col:
            render_card(name, description, accent)


def render_explanation_flow() -> None:
    steps = st.columns(4)
    content = [
        ("1. Entrada", "Todos los modelos reciben el mismo frame y la misma resolución. Así la comparación no depende del input."),
        ("2. Modelo", "YOLO, SSDlite y Faster R-CNN representan distintas familias de detectores de objetos."),
        ("3. Medición", "Se separan calentamiento y frames medidos para reportar latencia, FPS y costo aproximado."),
        ("4. Lectura", "Se conecta la fórmula Big-O con los datos reales del hardware local."),
    ]
    accents = ["blue", "green", "violet", "amber"]
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
    complexity_df = df.dropna(subset=["gflops_approx"])
    if not complexity_df.empty:
        heaviest = complexity_df.sort_values("gflops_approx", ascending=False).iloc[0]
        complexity_sentence = (
            f"El mayor costo aproximado lo tiene <strong>{heaviest['model']}</strong> "
            f"con {heaviest['gflops_approx']} GFLOPs."
        )
    else:
        complexity_sentence = "No se calculó GFLOPs en esta corrida."

    if presentation_mode:
        bullets = [
            f"<li>Menor latencia: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms por frame.</li>",
            f"<li>Mayor FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</li>",
            f"<li>{complexity_sentence} La conclusión cruza teoría, GFLOPs y tiempo real.</li>",
        ]
        body = "<ul>" + "".join(bullets) + "</ul>"
    else:
        body = (
            f"<p>Menor latencia: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms por frame.</p>\n"
            f"<p>Mayor FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</p>\n"
            f"<p>{complexity_sentence} La conclusión cruza teoría, GFLOPs y tiempo real.</p>"
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

    st.markdown("<h3 class='section-title'>Tabla completa</h3>", unsafe_allow_html=True)
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("<h3 class='section-title'>Gráficos</h3>", unsafe_allow_html=True)
    plots = plot_results(df)

    if csv_path:
        st.success(f"CSV exportado en: {csv_path}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar resultados CSV",
        csv_bytes,
        file_name="benchmark_yolo_complexity.csv",
        mime="text/csv",
        key="download_results_csv",
        on_click="ignore",
    )

    st.markdown("<h3 class='section-title'>Exportación HTML</h3>", unsafe_allow_html=True)
    st.caption("Sirve para guardar los gráficos interactivos y compartirlos sin volver a ejecutar el benchmark.")

    if st.button("Preparar gráficos HTML", key="prepare_html_export"):
        exported_paths = [write_plot_html(fig, name) for name, fig in plots]
        st.session_state["last_html_zip"] = build_plot_html_zip(plots)
        st.session_state["last_html_paths"] = [str(path) for path in exported_paths]

    if st.session_state.get("last_html_zip"):
        st.download_button(
            "Descargar gráficos HTML (.zip)",
            st.session_state["last_html_zip"],
            file_name="graficos_yolo_complexity_lab.zip",
            mime="application/zip",
            key="download_html_zip",
            on_click="ignore",
        )
        pass


def render_controls_guide() -> None:
    st.markdown("<h2 class='section-title'>Guía de controles</h2>", unsafe_allow_html=True)
    st.write(
        "Esta sección explica para qué sirve cada opción del panel lateral y cómo cambia la lectura del benchmark."
    )

    rows = [
        ("Modelos a comparar", "Elige qué detectores se miden. Sirve para contrastar familias: YOLO, SSDlite y Faster R-CNN."),
        ("Fuente de frames", "Define de dónde salen las imágenes: demo, imagen, video o webcam. Sirve para controlar el escenario de prueba."),
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

    default_models = [key for key, spec in MODEL_CATALOG.items() if spec.default_enabled]
    selected_models = st.multiselect(
        "Modelos a comparar",
        options=list(MODEL_CATALOG.keys()),
        default=default_models,
        format_func=lambda key: MODEL_CATALOG[key].display_name,
        help="Selecciona al menos dos modelos para comparar tiempo y complejidad. YOLO11n + SSDlite es una prueba rápida.",
    )

    source_kind = st.selectbox(
        "Fuente de frames",
        list(SOURCE_HELP.keys()),
        help="Define de dónde salen los frames usados en el benchmark.",
    )

    if source_kind == "Webcam OpenCV local":
        st.number_input("Índice de cámara", min_value=0, max_value=5, value=0, key="camera_index")

    device = st.selectbox("Dispositivo de ejecución", list(DEVICE_HELP.keys()), help="Controla si se usa CPU o GPU.")

    imgsz = st.select_slider(
        "Resolución cuadrada",
        options=[320, 416, 512, 640],
        value=416,
        help="Mayor resolución procesa más píxeles. Eso sube el costo aproximado n = H × W.",
    )

    with st.expander("Configuración avanzada"):
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

inicio_tab, benchmark_tab, teoria_tab = st.tabs(
    ["Inicio", "Benchmark", "Teoría"]
)

with inicio_tab:
    st.markdown("<h2 class='section-title'>Qué hace este recurso</h2>", unsafe_allow_html=True)
    render_model_overview()
    st.markdown("<h2 class='section-title'>Flujo de uso</h2>", unsafe_allow_html=True)
    render_explanation_flow()
    st.markdown("<h2 class='section-title'>Cómo leer los resultados</h2>", unsafe_allow_html=True)
    render_metric_glossary()

with teoria_tab:
    st.markdown("<h2 class='section-title'>Parámetro específico de complejidad</h2>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            r"""
El costo dominante en YOLO y en detectores CNN suele venir de las convoluciones:

```text
O(Σ_l H_l × W_l × C_in_l × C_out_l × K_l²)
```

Versión didáctica si definimos `n = H × W`:

```text
O(L × n × C_in × C_out × K²)
```

La lectura práctica es directa: si sube la resolución, sube `n`; si suben canales o capas, sube el costo por frame.
            """
        )
    with right:
        render_card(
            "Postprocesamiento",
            "Con NMS tradicional, comparar y filtrar cajas candidatas puede crecer como O(B²). B representa las cajas candidatas antes del filtrado final.",
            "violet",
        )
        st.write("")
        render_card(
            "Detectores two-stage",
            "Faster R-CNN agrega regiones propuestas: O(Σ conv + R × C_roi + NMS). R es el número de regiones evaluadas.",
            "amber",
        )
    st.info(
        "Big-O explica crecimiento teórico. Latencia, FPS y GFLOPs muestran lo que realmente ocurre en esta máquina."
    )

    st.markdown("<h2 class='section-title'>Catálogo de modelos</h2>", unsafe_allow_html=True)
    with st.expander("Guía de controles", expanded=False):
        render_controls_guide()

    with st.expander("Sistema y exportación", expanded=False):
        st.markdown("<h2 class='section-title'>Hardware y entorno local</h2>", unsafe_allow_html=True)
        st.write("Estos datos importan porque el benchmark no es universal: depende del equipo donde se ejecuta.")
        st.json(system_info_dict())
        st.write(f"Directorio de exportación: `{default_export_dir()}`")

with benchmark_tab:
    st.markdown("<h2 class='section-title'>Ejecución del benchmark</h2>", unsafe_allow_html=True)
    render_config_summary(selected_models, source_kind, device, int(imgsz), int(warmup_frames), int(measure_frames), include_complexity, True)
    run = st.button("Ejecutar benchmark", type="primary")
    total_needed = int(warmup_frames + measure_frames)

    if source_kind == "Webcam OpenCV local" and not run:
        frames, preview = [], None
        st.info("La webcam local se leerá recién cuando ejecutes el benchmark para evitar capturas innecesarias.")
    else:
        frames_to_load = total_needed if run else 1
        frames, preview = source_frames(source_kind, frames_to_load, imgsz)

    preview_col = st.columns(1)[0]
    with preview_col:
        if preview is not None:
            st.image(preview, caption="Vista previa del input", channels="RGB", width="stretch")

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

        for index, model_key in enumerate(selected_models, start=1):
            spec = MODEL_CATALOG[model_key]
            status.markdown(f"## Cargando: Modelo {index} de {len(selected_models)}")
            try:
                loaded = cached_load_model(model_key, device)
                row = benchmark_model(loaded, frames, config, include_complexity=include_complexity)
                rows.append(row)
            except Exception as exc:
                st.error(f"Falló {spec.display_name}: {exc}")
            progress.progress(index / len(selected_models))

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
    elif "last_benchmark_df" in st.session_state:
        st.info("Mostrando el último benchmark ejecutado. Puedes descargar CSV/HTML sin volver a medir.")
        render_benchmark_results(
            st.session_state["last_benchmark_df"],
            st.session_state.get("last_benchmark_csv_path"),
            True,
        )
    else:
        st.info("Ejecuta el benchmark para generar tabla, gráficos y descargas.")

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import cv2


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

st.set_page_config(
    page_title="YOLO Complexity Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

SOURCE_HELP_ALL = {
    "Demo image": "Use a local image with a person, dog, and banana to compare recognition and false positives.",
    "Upload image": "Repeat one image to measure latency without depending on a video stream.",
    "Local OpenCV webcam": "Capture frames from a local camera for a live demo; camera access and lighting are required.",
}

SOURCE_HELP = {
    key: value
    for key, value in SOURCE_HELP_ALL.items()
    if CAPABILITIES.webcam or key != "Local OpenCV webcam"
}

# Opciones de ruta: en cloud, no funciona YOLO en vivo (necesita webcam)
PRESET_MODELS_ALL = {
    "Live YOLO": ["yolo11n"],
    "CNN vs YOLO comparison": [
        "fasterrcnn_mobilenet_fpn",
        "ssdlite_mobilenet_v3",
        "yolo11n",
    ],
    "Custom weights": ["yolov8_gestures"],
}

PRESET_HELP_ALL = {
    "YOLO actual en vivo": "Mostrar YOLO11n funcionando en tiempo real con webcam local.",
    "Comparación CNN vs YOLO": "Comparar dos etapas, one-stage CNN y YOLO para probar tiempo y complejidad.",
}

PRESET_HELP_ALL["Custom weights"] = "Run the optional root best.pt weights when they are available."
PRESET_MODELS = {
    key: value
    for key, value in PRESET_MODELS_ALL.items()
    if (CAPABILITIES.streaming or key != "Live YOLO")
    and (CAPABILITIES.custom_weights or key != "Custom weights")
}
PRESET_HELP = {
    key: value
    for key, value in PRESET_HELP_ALL.items()
    if key in PRESET_MODELS
}

DEVICE_HELP = {
    "auto": "Use a GPU when PyTorch detects CUDA; otherwise use the CPU.",
    "cpu": "Force CPU execution for more comparable, usually slower measurements.",
    "cuda:0": "Use the first NVIDIA GPU; the loader falls back to CPU when unavailable.",
}
DEVICE_OPTIONS = CAPABILITIES.device_options

NAVIGATION_TABS = ("Overview", "Benchmark", "About")
AUTHOR_NAME = "Alejandro Padilla"
METRIC_EXPLANATIONS = {
    "Latency": "Time required to process one frame. Lower is better.",
    "FPS": "Effective frames per second. Higher is better for real-time use.",
    "GFLOPs": "Approximate operations per frame (1 MAC ≈ 2 FLOPs), a computational-cost proxy.",
    "Parameters": "Learned weights that affect model size, memory use, and capacity.",
    "Big-O": "Describes how cost grows with resolution, layers, channels, or candidate boxes.",
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
  --orange: #FF6600;
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
.card-accent-orange { border-left: 4px solid var(--orange); }
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

.preview-frame { border: 1px solid var(--line); border-radius: 22px; background: #ffffff; padding: 0.65rem; box-shadow: var(--shadow); }
.preview-frame img { width: 100%; max-height: 340px; object-fit: contain; display: block; border-radius: 16px; background: #f8fafc; }
.preview-caption { text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 0.45rem; }

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
            "Inference dependencies are missing: "
            + ", ".join(missing)
            + ". Install them with `python -m pip install -r requirements.txt` inside `.venv`."
        )


def render_evidence_path(df: pd.DataFrame | None = None) -> None:
    if df is None or df.empty:
        st.info("Evidence is unavailable until a benchmark runs. Use the Benchmark tab to select a source, measure the models, inspect the results, and download the CSV.")
        return
    st.success(f"Evidence available for {len(df)} model(s): measured input → runtime metrics → visual detections → CSV export.")


def render_hero(presentation_mode: bool = False) -> None:
    subtitle = (
        "A measured comparison of detector recognition, latency, FPS, and computational cost."
        if presentation_mode
        else "Compare classic detectors and YOLO through recognition, latency, FPS, and computational cost."
    )
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-kicker">A computer-vision complexity lab</div>
  <div class="hero-title">YOLO Complexity Lab</div>
  <div class="hero-subtitle">Built by {AUTHOR_NAME}</div>
  <div class="hero-subtitle">
    {subtitle}
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
    if source_kind == "Demo image":
        frame = sample_coco_frame()
        return repeat_frame(frame, total_needed), frame

    if source_kind == "Upload image":
        # Only show file uploader on first call (preview), reuse on benchmark
        if total_needed == 1:
            uploaded = st.file_uploader(
                "Image file",
                type=["jpg", "jpeg", "png", "webp"],
                key="image_upload",
                help="The app repeats this image to measure multiple frames with the same input.",
            )
            if uploaded is None:
                st.info("Upload an image or switch to Demo image for a quick test.")
                return [], None
            # Cache the decoded frame for the benchmark run
            frame = read_image_file(uploaded)
            st.session_state["uploaded_image_frame"] = frame
            return repeat_frame(frame, total_needed), frame
        else:
            # Reuse cached frame from preview call
            frame = st.session_state.get("uploaded_image_frame")
            if frame is None:
                st.error("Upload an image in the preview before running the benchmark.")
                return [], None
            return repeat_frame(frame, total_needed), frame

    if source_kind == "Local OpenCV webcam":
        if not CAPABILITIES.webcam:
            st.error("Webcam capture is unavailable in this environment. Choose Demo or Image upload.")
            return [], None
        camera_index = int(st.session_state.get("camera_index", 0))
        try:
            frames = frames_from_webcam(camera_index, limit=total_needed)
        except Exception as exc:
            st.error(
                f"Unable to access webcam index {camera_index}: {exc}. "
                "Check permissions, connect a camera, or choose another source."
            )
            return [], None
        if not frames:
            st.error(
                f"Webcam index {camera_index} returned no frames. "
                "Check permissions, connection, and camera availability."
            )
        preview = frames[0] if frames else None
        return frames, preview

    frame = sample_coco_frame()
    return repeat_frame(frame, total_needed), frame


def render_preview_image(frame: object, caption: str = "Input preview") -> None:
    """Render a compact full-image preview with object-fit containment."""
    try:
        import cv2
        import numpy as np

        # Convert RGB to BGR for cv2.imencode (OpenCV uses BGR by default).
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


def run_webcam_benchmark_streaming(loaded, imgsz: int, confidence: float, iou: float, device: str, camera_index: int, measure_frames: int | None = None) -> dict[str, list]:
    """Ejecuta streaming en tiempo real desde webcam recopilando métricas de benchmarking."""
    if not CAPABILITIES.streaming:
        st.error("Live streaming is unavailable in this environment. Choose a supported image source.")
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
            f"Unable to open webcam index {camera_index}. "
            "Check permissions, connection, and camera availability."
        )
        return {}
    
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

    # Inicializar flag de parada en session_state y renderizar botón una sola vez
    if "stream_stop_requested" not in st.session_state:
        st.session_state["stream_stop_requested"] = False

    stop_col = stop_placeholder.columns([4, 1])[1]
    if stop_col.button("Stop", key="stop_btn_stream"):
        st.session_state["stream_stop_requested"] = True

    try:
        # Streaming en vivo: si measure_frames es None, iteramos hasta que el usuario pare
        while True:
            # Si el usuario solicitó parar, salimos del loop
            if st.session_state.get("stream_stop_requested", False):
                break
            if measure_frames is not None and frame_count >= measure_frames:
                break
            
            success, frame = cap.read()
            if not success:
                continue
            
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
                    
                    # --- NUEVO: GUARDADO CONTINUO ---
                    # Hacemos una copia de seguridad en cada frame. Si el botón 'Parar' mata 
                    # el proceso, los datos ya están a salvo para graficarlos.
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
                            col_img.image(annotated_rgb, caption=f"Live frame {frame_count}/{measure_frames}", channels="RGB", width="stretch")
                        with col_info:
                            st.metric("Detections", detections_count)
                    
                            # Mostrar estadísticas en tiempo real
                            if frame_count > 0 and timings["total"]:
                                with placeholder_stats.container():
                                    col1, col2, col3, col4 = st.columns(4)
                            
                                    with col1:
                                        avg_latency = statistics.mean(timings["total"])
                                        st.metric("Mean latency", f"{avg_latency:.2f} ms")
                            
                                    with col2:
                                        if statistics.mean(timings["total"]) > 0:
                                            avg_fps = 1000 / statistics.mean(timings["total"])
                                        else:
                                            avg_fps = 0
                                        st.metric("Mean FPS", f"{avg_fps:.1f}")
                            
                                    with col3:
                                        st.metric("Captured frames", frame_count)
                            
                                    with col4:
                                        avg_detections = statistics.mean(timings["detections"])
                                        st.metric("Mean detections", f"{avg_detections:.1f}")

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
                                    "Download partial CSV",
                                    csv_bytes,
                                    file_name="streaming_partial_results.csv",
                                    mime="text/csv",
                                    key=f"download_stream_csv_{frame_count}",
                                )
                            except Exception:
                                pass
                else:
                    st.error("The streaming benchmark currently supports YOLO only.")
                    break
                    
            except Exception as e:
                st.warning(f"Frame {frame_count}: {str(e)[:100]}")
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
            "model": "Model",
            "family": "Family",
            "latency_mean_ms": "Mean latency (ms)",
            "latency_p95_ms": "p95 (ms)",
            "fps_effective": "FPS",
            "input_pixels_n": "n = H×W",
            "gflops_approx": "Approx. GFLOPs",
            "parameters_millions": "Parameters (M)",
            "recognized_classes": "Recognized classes",
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
        cost_label = "Lowest approx. GFLOPs"
        cost_value = f"{lowest_cost['gflops_approx']} G"
        cost_delta = lowest_cost["model"]
    else:
        lowest_cost = df.sort_values("parameters_millions", na_position="last").iloc[0]
        cost_label = "Fewest parameters"
        cost_value = f"{lowest_cost['parameters_millions']} M"
        cost_delta = lowest_cost["model"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Lowest mean latency", f"{fastest['latency_mean_ms']} ms", fastest["model"])
    c2.metric("Highest effective FPS", f"{most_fps['fps_effective']} FPS", most_fps["model"])
    c3.metric(cost_label, cost_value, cost_delta)


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
        title="Latency by model",
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
        title="Effective FPS",
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
            title="Computational complexity vs runtime",
            labels={
                "gflops_approx": "Approx. GFLOPs",
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
            "1. Live YOLO",
            "Start with a local webcam: one forward pass per frame makes the real-time trade-off visible.",
            "orange",
        )
    with c2:
        render_card(
            "2. Comparison",
            "Measure Faster R-CNN, SSDlite, and YOLO with the same input and settings.",
            "green",
        )
    with c3:
        render_card(
            "3. Conclusion",
            "Use latency, FPS, and n = H×W to explain the measured complexity trade-off.",
            "violet",
        )


def render_theory_bridge() -> None:
    st.markdown("<h2 class='section-title'>From theory to measured evidence</h2>", unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        (
            "Question",
            "Object detection must locate and classify several objects. The YOLO-versus-CNN question is whether a single pass reduces runtime cost.",
            "amber",
        ),
        (
            "Method",
            "Run the same input through selected detectors and compare runtime, effective FPS, model cost, and recognized classes.",
            "orange",
        ),
        (
            "Cost model",
            "Convolutions dominate growth: resolution, layers, channels, and kernel size increase the operation count.",
            "violet",
        ),
        (
            "Evidence",
            "The benchmark connects Big-O with GFLOPs, parameters, latency, FPS, and recognized classes, then exports the measured rows to CSV.",
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
        ("Runtime", "Mean and p95 latency plus effective FPS."),
        ("Complexity", "n = H×W, GFLOPs, and parameter count."),
        ("Recognition", "A visual aid for this input, not a formal mAP evaluation."),
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
    model_names = ", ".join(MODEL_CATALOG[key].display_name for key in selected_models) if selected_models else "No models"
    if presentation_mode:
        st.caption(f"Models: {model_names} | Source: {source_kind} | Device: {device} | {imgsz}×{imgsz} | Warmup: {warmup_frames} | Measured: {measure_frames}")
    else:
        st.markdown(
            f"""
<div class="glass-card card-accent-violet">
  <span class="small-label">Current configuration</span>
  <p>Models: {model_names} | Source: {source_kind} | Device: {device} | {imgsz}×{imgsz} | Warmup: {warmup_frames} | Measured: {measure_frames}</p>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_benchmark_focus(imgsz: int, streaming_mode: bool, comparison_route: str) -> None:
    n_pixels = imgsz * imgsz
    baseline_n = 320 * 320
    growth = n_pixels / baseline_n
    mode_title = "Live YOLO" if streaming_mode else "Measured comparison"
    st.markdown("<h3 class='section-title'>What to look for</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card(
            "Runtime",
            "Lower latency means higher FPS. At 30 FPS, each frame should be at or below 33 ms.",
            "blue",
        )
    with c2:
        render_card(
            "Complexity n",
            f"n = H×W = {imgsz}×{imgsz} = {n_pixels:,} pixels. Compared with 320², this processes {growth:.2f}× more pixels.",
            "violet",
        )
    with c3:
        render_card(
            mode_title,
            "YOLO processes the image in one pass; two-stage models add region proposals and more cost.",
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
            f"Lowest approximate GFLOPs: <strong>{lowest_gflops['model']}</strong> "
            f"with {lowest_gflops['gflops_approx']} G."
        )
    else:
        complexity_sentence = "GFLOPs was not calculated for this run."

    theory_sentence = (
        "With the same n = H×W, the primary evidence is latency/FPS: "
        "if YOLO runs faster, the benefit of its one-pass architecture and optimizations is visible."
    )

    if presentation_mode:
        bullets = [
            f"<li>Lowest latency: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms per frame.</li>",
            f"<li>Highest FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</li>",
            f"<li>Most detections for this input: <strong>{most_detections['model']}</strong> — {most_detections.get('recognized_classes', 'no details')}.</li>",
            f"<li>{complexity_sentence} GFLOPs is a proxy; real-time performance is determined by latency/FPS.</li>",
            f"<li>{theory_sentence}</li>",
        ]
        body = "<ul>" + "".join(bullets) + "</ul>"
    else:
        body = (
            f"<p>Lowest latency: <strong>{fastest['model']}</strong> — {fastest['latency_mean_ms']} ms per frame.</p>\n"
            f"<p>Highest FPS: <strong>{highest_fps['model']}</strong> — {highest_fps['fps_effective']} FPS.</p>\n"
            f"<p>Most detections for this input: <strong>{most_detections['model']}</strong> — {most_detections.get('recognized_classes', 'no details')}.</p>\n"
            f"<p>{complexity_sentence} GFLOPs is a proxy; real-time performance is determined by latency/FPS.</p>\n"
            f"<p>{theory_sentence}</p>"
        )

    st.markdown(
        f"""
<div class="glass-card card-accent-green">
  <span class="small-label">Interpretation for teaching</span>
  {body}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_summary(df: pd.DataFrame) -> None:
    if df.empty or "recognized_classes" not in df.columns:
        return
    st.markdown("<h3 class='section-title'>Recognition by model</h3>", unsafe_allow_html=True)
    st.caption(
        "This does not replace mAP: it is a qualitative reading of the image used in the benchmark. "
        "Use it to inspect false negatives, weak detections, and differences between model families."
    )
    cols = st.columns(min(3, len(df)))
    for col, (_, row) in zip(cols, df.iterrows(), strict=False):
        with col:
            confidence = row.get("avg_confidence")
            confidence_text = "—" if pd.isna(confidence) else f"{float(confidence):.2f}"
            render_card(
                str(row.get("model", "Model")),
                (
                    f"<strong>Recognized:</strong> {row.get('recognized_classes', 'No detections')}<br>"
                    f"<strong>Top detection:</strong> {row.get('top_detection', '—')}<br>"
                    f"<strong>Mean confidence:</strong> {confidence_text}"
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
    st.markdown("<h3 class='section-title'>Practical live YOLO summary</h3>", unsafe_allow_html=True)
    st.caption(
        "This section does not compare models. It summarizes how YOLO11n performed on the local webcam: "
        "time per frame, FPS, detections, and approximate model cost."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean latency", _display_value(row, "latency_mean_ms", " ms"))
    c2.metric("Effective FPS", _display_value(row, "fps_effective", " FPS"))
    c3.metric("Processed frames", _display_value(row, "frames_measured", decimals=0))
    c4.metric("Mean detections", _display_value(row, "detections_mean", decimals=1))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("p95 latency", _display_value(row, "latency_p95_ms", " ms"))
    c6.metric("Mean inference", _display_value(row, "inference_mean_ms", " ms"))
    c7.metric("Approx. GFLOPs", _display_value(row, "gflops_approx", " G", decimals=4))
    c8.metric("Parameters", _display_value(row, "parameters_millions", " M"))


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
    with st.expander("View technical run data"):
        st.dataframe(df[available], width="stretch", hide_index=True)

    if csv_path:
        csv_name = Path(csv_path).name
        st.success(f"CSV generated: {csv_name}")
        st.caption(f"Local path: {csv_path}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download summary CSV",
        csv_bytes,
        file_name="live_yolo_summary.csv",
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
        return

    if is_streaming:
        render_live_yolo_results(df, csv_path, presentation_mode)
        return

    metric_cards(df, presentation_mode)

    render_detection_summary(df)

    st.markdown("<h3 class='section-title'>Comparison summary</h3>", unsafe_allow_html=True)
    st.dataframe(compact_results_table(df), width="stretch", hide_index=True)

    with st.expander("View full technical table"):
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("<h3 class='section-title'>Charts</h3>", unsafe_allow_html=True)
    plots = plot_results(df)

    if csv_path:
        csv_name = Path(csv_path).name
        st.success(f"CSV ready: {csv_name}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download benchmark CSV",
        csv_bytes,
        file_name="benchmark_yolo_complexity.csv",
        mime="text/csv",
        key="download_results_csv",
        on_click="ignore",
    )

    # Exportación HTML removida a pedido del usuario

    # --- Vista de detecciones: mostrar el último frame anotado por cada modelo ---
    st.markdown("<h3 class='section-title'>Visual detection (last measured frame)</h3>", unsafe_allow_html=True)
    st.caption("These images show what each model recognized in the last measured frame. Use them to inspect visual recognition, false positives, and false negatives; they do not replace mAP.")
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

with st.sidebar:
    comparison_route = st.radio(
        "Comparison route",
        options=list(PRESET_MODELS.keys()),
        help="Choose the experiment route and its model set.",
    )

    previous_route = st.session_state.get("comparison_route", None)

    if previous_route is not None and previous_route != comparison_route:
        st.session_state.pop("last_benchmark_df", None)
        st.session_state.pop("last_benchmark_csv_path", None)
        st.session_state.pop("annotated_frames", None)
        st.session_state.pop("pending_streaming_results", None)
    
    st.session_state["comparison_route"] = comparison_route

    selected_models = PRESET_MODELS[comparison_route]

    source_options = list(SOURCE_HELP.keys())
    default_source = "Demo image" if CAPABILITIES.is_cloud else ("Local OpenCV webcam" if comparison_route == "Live YOLO" else "Demo image")
    source_kind = st.selectbox(
        "Frame source",
        source_options,
        index=source_options.index(default_source),
        help="Choose the input source used by the benchmark.",
    )

    streaming_mode = False
    if CAPABILITIES.streaming and source_kind == "Local OpenCV webcam":
        streaming_mode = st.checkbox(
            "Live streaming mode",
            value=comparison_route == "Live YOLO",
            help="Process local camera frames in real time.",
        )

    with st.expander("Advanced settings"):
        if CAPABILITIES.webcam and source_kind == "Local OpenCV webcam":
            st.number_input("Camera index", min_value=0, max_value=5, value=0, key="camera_index")

        device = st.selectbox(
            "Execution device",
            options=list(DEVICE_OPTIONS),
            index=list(DEVICE_OPTIONS).index(CAPABILITIES.device_default),
            format_func=lambda option: f"{option}: {DEVICE_HELP[option]}",
            help="Choose CPU or GPU according to the detected environment.",
        )

        imgsz = st.select_slider(
            "Square resolution",
            options=[320, 416, 512, 640],
            value=416,
            help="Higher resolution processes more pixels and increases approximate n = H × W cost.",
        )
        warmup_frames = st.number_input(
            "Warmup frames",
            min_value=0,
            max_value=30,
            value=3,
            help="Excluded from results; used to stabilize model loading, caches, and GPU behavior.",
            disabled=streaming_mode,
        )
        measure_frames = st.number_input(
            "Measured frames",
            min_value=1,
            max_value=300,
            value=20,
            help="Included in latency, FPS, and final statistics.",
            disabled=streaming_mode,
        )
        confidence = st.slider(
            "Minimum confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Accept detections above this confidence. Higher values reduce boxes but may miss objects.",
            disabled=streaming_mode,
        )
        iou = st.slider(
            "IoU for NMS",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05,
            help="Control when overlapping boxes are merged or removed during NMS.",
            disabled=streaming_mode,
        )
        include_complexity = st.checkbox(
            "Calculate approximate MACs/GFLOPs",
            value=True,
            help="Estimate Conv2d and Linear operations with an additional forward pass.",
            disabled=streaming_mode,
        )

        pass

overview_tab, benchmark_tab, about_tab = st.tabs(NAVIGATION_TABS)

with overview_tab:
    render_hero(True)
    render_model_overview()
    render_evidence_path(st.session_state.get("last_benchmark_df"))
    render_theory_bridge()
    st.markdown("<h2 class='section-title'>How to read the benchmark</h2>", unsafe_allow_html=True)
    render_explanation_flow()
    st.write("")
    with st.expander("View metric glossary"):
        render_metric_glossary()

with about_tab:
    st.markdown(f"<h2 class='section-title'>About this project</h2><p>Portfolio experiment by <strong>{AUTHOR_NAME}</strong> about the YOLO-versus-CNN complexity question. It uses the same input and measured settings to compare runtime, computational cost, and visual recognition.</p><p>Open Benchmark to choose a source, run the measurement, inspect the evidence, and export the session results as CSV.</p>", unsafe_allow_html=True)

with benchmark_tab:
    title = "Live YOLO" if streaming_mode else "Runtime and complexity benchmark"
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    st.caption("How long does each frame take, and how does cost grow as n = H×W increases?")
    
    total_needed = int(warmup_frames + measure_frames)
    frames, preview = (
        ([], None)
        if source_kind == "Local OpenCV webcam"
        else source_frames(source_kind, 1, imgsz)
    )

    preview_col = st.columns(1)[0]
    with preview_col:
        if preview is not None:
            render_preview_image(preview, "Input preview")
    
    st.write("")
    
    # Botón de ejecución debajo de la imagen
    run = st.button("Start live YOLO" if streaming_mode else "Run comparison", type="primary")
    
    if source_kind == "Local OpenCV webcam" and not run:
        frames, preview = [], None
        st.info("The local webcam is accessed only after you start the benchmark.")
    elif run:
        # Recargar frames para el benchmark completo
        if streaming_mode:
            frames, preview = [], None
        else:
            frames, preview = source_frames(source_kind, total_needed, imgsz)

    if run:
        if not selected_models:
            st.error("Select at least one model before starting the benchmark.")
            st.stop()
        
        # Modo streaming con webcam
        if source_kind == "Local OpenCV webcam" and streaming_mode:
            if len(selected_models) > 1:
                st.warning("The streaming benchmark supports one model at a time. The first selected model will be used.")
            
            model_key = selected_models[0]
            spec = MODEL_CATALOG[model_key]
            
            st.markdown(f"<h3>Live streaming: {spec.display_name}</h3>", unsafe_allow_html=True)
            st.info(f"Capturing real-time frames from camera {st.session_state.get('camera_index', 0)}. Press 'Stop' to finish and view the summary.")
            
            try:
                st.session_state.streaming_active = True
                loaded = cached_load_model(model_key, device)
                
                streaming_results = run_webcam_benchmark_streaming(
                    loaded,
                    imgsz=int(imgsz),
                    confidence=float(confidence),
                    iou=float(iou),
                    device=device,
                    camera_index=int(st.session_state.get('camera_index', 0)),
                    measure_frames=None,  # <--- Esto lo hace infinito
                )
                
                if streaming_results:
                    st.success("Streaming finished. Performance summary:")
                    
                    # Mostrar resumen simple del streaming (no tabla de comparación)
                    from yolo_complexity_lab.complexity import estimate_for_loaded_model
                    complexity = estimate_for_loaded_model(loaded, int(imgsz)) if include_complexity else None
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean latency", f"{round(streaming_results['latency_mean_ms'], 1)} ms")
                        st.metric("Effective FPS", f"{round(streaming_results['fps_effective'], 1)}")
                    with col2:
                        st.metric("Measured frames", streaming_results["frames_measured"])
                        st.metric("Mean detections", f"{round(streaming_results['detections_mean'], 1)}")
                    with col3:
                        st.metric("Preprocessing", f"{round(streaming_results['preprocess_mean_ms'], 1)} ms")
                        st.metric("Inference", f"{round(streaming_results['inference_mean_ms'], 1)} ms")
                    
                    if complexity:
                        st.write(f"**Complexity:** {complexity.gflops_approx} GFLOPs | {complexity.gmacs} GMACs | {complexity.conv_layers} Conv layers")
                    
                    st.write(f"**Model:** {spec.display_name} | **Device:** {device} | **Resolution:** {imgsz}px")
                    
                    # Solo guardar para referencia, no mostrar como comparación
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
                        "complexity_note": complexity.note if complexity else "Not calculated.",
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
                st.error(f"Streaming error: {exc}")
        
        # Modo benchmark estándar
        else:
            if not frames:
                st.error(
                    "No frames are available for the selected source. "
                    "Choose another source or verify the webcam and permissions."
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
                status.markdown(f"## Loading model {index} of {len(selected_models)}")
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
                    st.error(f"{spec.display_name} failed: {exc}")
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
                st.warning("No model could be measured. Check dependencies, connectivity, or weight availability.")
    elif "pending_streaming_results" in st.session_state:
        st.success("Streaming finished. Showing the practical live YOLO summary...")
        
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
            "complexity_note": complexity.note if complexity else "Not calculated.",
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
        st.info("Showing the latest benchmark. You can download its CSV without measuring again.")
        render_benchmark_results(
            st.session_state["last_benchmark_df"],
            st.session_state.get("last_benchmark_csv_path"),
            True,
            is_streaming=streaming_mode,
        )
    else:
        if streaming_mode:
            st.info("Start live YOLO to view latency, FPS, and detections from the camera.")
        else:
            st.info("Run the comparison to generate the table, charts, and CSV.")

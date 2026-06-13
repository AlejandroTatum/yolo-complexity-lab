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
from yolo_complexity_lab.exporting import write_plot_html, write_results_csv
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
    page_icon="📦",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_load_model(spec_key: str, device: str):
    return load_model(spec_key, device)


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
            + ". Instalá con `pip install -r requirements.txt` dentro del proyecto."
        )


def source_frames(source_kind: str, total_needed: int, imgsz: int) -> tuple[list[object], object | None]:
    if source_kind == "Imagen demo":
        frame = demo_frame(imgsz)
        return repeat_frame(frame, total_needed), frame

    if source_kind == "Subir imagen":
        uploaded = st.file_uploader("Imagen", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
        if uploaded is None:
            st.info("Subí una imagen o cambiá a imagen demo.")
            return [], None
        frame = read_image_file(uploaded)
        return repeat_frame(frame, total_needed), frame

    if source_kind == "Subir video":
        uploaded_video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
        if uploaded_video is None:
            st.info("Subí un video para medir frames reales.")
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


def metric_cards(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fastest = df.sort_values("latency_mean_ms").iloc[0]
    lightest = df.sort_values("parameters_millions", na_position="last").iloc[0]
    most_fps = df.sort_values("fps_effective", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Menor latencia media", f"{fastest['latency_mean_ms']} ms", fastest["model"])
    c2.metric("Mayor FPS efectivo", f"{most_fps['fps_effective']} FPS", most_fps["model"])
    c3.metric("Menos parámetros", f"{lightest['parameters_millions']} M", lightest["model"])


def plot_results(df: pd.DataFrame) -> list[tuple[str, object]]:
    plots = []
    if df.empty:
        return plots

    latency_fig = px.bar(
        df,
        x="model",
        y="latency_mean_ms",
        color="family",
        title="Latencia media por modelo (menor es mejor)",
        labels={"latency_mean_ms": "Latencia media (ms)", "model": "Modelo"},
    )
    st.plotly_chart(latency_fig, use_container_width=True)
    plots.append(("latencia_media", latency_fig))

    fps_fig = px.bar(
        df,
        x="model",
        y="fps_effective",
        color="family",
        title="FPS efectivo por modelo (mayor es mejor)",
        labels={"fps_effective": "FPS", "model": "Modelo"},
    )
    st.plotly_chart(fps_fig, use_container_width=True)
    plots.append(("fps_efectivo", fps_fig))

    complexity_fig = px.scatter(
        df,
        x="gflops_approx",
        y="latency_mean_ms",
        size="parameters_millions",
        color="family",
        hover_name="model",
        title="Complejidad aproximada vs tiempo real local",
        labels={"gflops_approx": "GFLOPs aprox.", "latency_mean_ms": "Latencia media (ms)"},
    )
    st.plotly_chart(complexity_fig, use_container_width=True)
    plots.append(("gflops_vs_latencia", complexity_fig))
    return plots


st.title("📦 YOLO Complexity Lab")
st.caption("Recurso local para comparar tiempo de ejecución y complejidad computacional en detectores de objetos.")
dependency_warning()

with st.sidebar:
    st.header("Configuración")
    default_models = [key for key, spec in MODEL_CATALOG.items() if spec.default_enabled]
    selected_models = st.multiselect(
        "Modelos a comparar",
        options=list(MODEL_CATALOG.keys()),
        default=default_models,
        format_func=lambda key: MODEL_CATALOG[key].display_name,
    )
    source_kind = st.selectbox("Fuente", ["Imagen demo", "Subir imagen", "Subir video", "Webcam OpenCV local"])
    if source_kind == "Webcam OpenCV local":
        st.number_input("Índice de cámara", min_value=0, max_value=5, value=0, key="camera_index")
    device = st.selectbox("Dispositivo", ["auto", "cpu", "cuda:0"])
    imgsz = st.select_slider("Resolución cuadrada", options=[320, 416, 512, 640], value=640)
    warmup_frames = st.number_input("Warmup frames", min_value=0, max_value=30, value=5)
    measure_frames = st.number_input("Frames medidos", min_value=1, max_value=300, value=50)
    confidence = st.slider("Confianza mínima", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = st.slider("IoU NMS", min_value=0.10, max_value=0.95, value=0.45, step=0.05)
    include_complexity = st.checkbox("Calcular MACs/GFLOPs aproximados", value=True)
    run = st.button("Ejecutar benchmark", type="primary")

intro_tab, benchmark_tab, theory_tab, system_tab = st.tabs([
    "Resumen",
    "Benchmark",
    "Big-O explicado",
    "Sistema",
])

with intro_tab:
    st.subheader("Objetivo")
    st.write(
        "Este recurso compara detectores por **tiempo** y **complejidad computacional**, "
        "no por mAP. La idea es ver cómo el diseño YOLO reduce latencia al hacer detección "
        "en una sola pasada y cómo eso se refleja en FPS, parámetros y GFLOPs aproximados."
    )
    st.dataframe(pd.DataFrame(catalog_rows()), use_container_width=True, hide_index=True)

with theory_tab:
    st.subheader("Parámetro específico de complejidad")
    st.markdown(
        r"""
Para una CNN/detector tipo YOLO, el costo dominante por frame suele venir de las convoluciones:

```text
O(Σ_l H_l × W_l × C_in_l × C_out_l × K_l²)
```

Versión didáctica si definimos `n = H × W`:

```text
O(L × n × C_in × C_out × K²)
```

Donde:
- `L`: cantidad de capas.
- `n = H × W`: cantidad de posiciones espaciales procesadas.
- `C_in` y `C_out`: canales de entrada y salida.
- `K²`: tamaño del filtro convolucional.

Para postprocesamiento:

```text
NMS tradicional: O(B²)
```

`B` es el número de cajas candidatas. Por eso, reducir cajas o evitar NMS ayuda a bajar latencia.
        """
    )
    st.info(
        "Lectura correcta: Big-O explica crecimiento teórico; la evidencia local se mide con latencia, FPS y GFLOPs aproximados."
    )

with system_tab:
    st.subheader("Hardware detectado")
    st.json(system_info_dict())
    st.write(f"Directorio de exportación: `{default_export_dir()}`")

with benchmark_tab:
    st.subheader("Ejecución")
    total_needed = int(warmup_frames + measure_frames)

    # Avoid reading an entire video/webcam stream on every Streamlit rerun.
    # Before pressing the button we only build a cheap preview frame.
    if source_kind == "Webcam OpenCV local" and not run:
        frames, preview = [], None
        st.info("La webcam local se leerá recién cuando ejecutes el benchmark.")
    else:
        frames_to_load = total_needed if run else 1
        frames, preview = source_frames(source_kind, frames_to_load, imgsz)

    if preview is not None:
        st.image(preview, caption="Preview de entrada", channels="RGB", use_container_width=False)

    if run:
        if not selected_models:
            st.error("Seleccioná al menos un modelo.")
            st.stop()
        if not frames:
            st.error("No hay frames disponibles para el benchmark.")
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
            status.write(f"Cargando y midiendo: **{spec.display_name}**")
            try:
                loaded = cached_load_model(model_key, device)
                row = benchmark_model(loaded, frames, config, include_complexity=include_complexity)
                rows.append(row)
            except Exception as exc:
                st.error(f"Falló {spec.display_name}: {exc}")
            progress.progress(index / len(selected_models))

        if rows:
            df = pd.DataFrame(rows)
            metric_cards(df)
            st.dataframe(df, use_container_width=True, hide_index=True)
            plots = plot_results(df)
            export_path = write_results_csv(df)
            st.success(f"CSV exportado en: {export_path}")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv_bytes, file_name=export_path.name, mime="text/csv")
            if st.checkbox("Exportar gráficos como HTML", value=False):
                exported = [write_plot_html(fig, name) for name, fig in plots]
                for path in exported:
                    st.write(f"Gráfico exportado: `{path}`")
        else:
            st.warning("No se pudo medir ningún modelo.")

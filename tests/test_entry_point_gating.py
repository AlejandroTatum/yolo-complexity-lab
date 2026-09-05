from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import streamlit
import numpy as np
import pandas as pd

from yolo_complexity_lab import environment as environment_module
from yolo_complexity_lab.environment import detect_capabilities


ROOT = Path(__file__).parents[1]


def load_app(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: SimpleNamespace,
    run: bool = False,
    source_kind: str | None = None,
    session_state: dict | None = None,
):
    widgets: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(environment_module, "detect_capabilities", lambda *_: capabilities)
    state = session_state if session_state is not None else type("State", (dict,), {"__getattr__": dict.__getitem__, "__setattr__": dict.__setitem__})()
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(streamlit, "cache_resource", lambda **_: lambda function: function)

    def radio(label: str, options, **_: object):
        values = tuple(options)
        widgets.append((label, values))
        return values[0]

    def selectbox(label: str, options, index: int = 0, **_: object):
        values = tuple(options)
        widgets.append((label, values))
        if label == "Fuente de frames" and source_kind is not None:
            return source_kind
        return values[index]

    monkeypatch.setattr(streamlit, "radio", radio)
    monkeypatch.setattr(streamlit, "selectbox", selectbox)
    monkeypatch.setattr(streamlit, "button", lambda *_args, **_kwargs: run)

    name = f"app_behavior_{id(capabilities)}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, widgets


def test_empty_standard_source_does_not_consume_notice_before_real_first_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yolo_complexity_lab import benchmark as benchmark_module, catalog as catalog_module, exporting as exporting_module, loaders as loaders_module, sources as sources_module

    capabilities = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu")
    state = type("State", (dict,), {"__getattr__": dict.__getitem__, "__setattr__": dict.__setitem__})()
    events: list[tuple[str, str]] = []
    load_calls: list[str] = []

    class BenchmarkStopped(Exception):
        pass

    monkeypatch.setattr(streamlit, "file_uploader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(streamlit, "stop", lambda: (_ for _ in ()).throw(BenchmarkStopped()))
    monkeypatch.setattr(streamlit, "info", lambda message: events.append(("info", str(message))))
    monkeypatch.setattr(streamlit, "error", lambda message: events.append(("error", str(message))))
    monkeypatch.setattr(streamlit, "warning", lambda message: events.append(("warning", str(message))))

    with pytest.raises(BenchmarkStopped):
        load_app(monkeypatch, capabilities, run=True, source_kind="Subir imagen", session_state=state)

    assert "cold_start_notice_shown" not in state
    assert not load_calls

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    def fake_load(model_key: str, device: str):
        load_calls.append(model_key)
        events.append(("load", model_key))
        return SimpleNamespace(spec=catalog_module.MODEL_CATALOG[model_key], device=device)

    monkeypatch.setattr(loaders_module, "load_model", fake_load)
    monkeypatch.setattr(benchmark_module, "benchmark_model", lambda loaded, frames, config, include_complexity=True: {"model": loaded.spec.display_name, "family": loaded.spec.family, "latency_mean_ms": 10.0, "fps_effective": 100.0, "gflops_approx": 1.0, "parameters_millions": 2.0, "detections_mean": 1.0})
    monkeypatch.setattr(sources_module, "sample_coco_frame", lambda: frame)
    monkeypatch.setattr(sources_module, "repeat_frame", lambda value, count: [value] * count)
    monkeypatch.setattr(exporting_module, "write_results_csv", lambda _df: tmp_path / "results.csv")

    load_app(monkeypatch, capabilities, run=True, source_kind="Demo persona/perro/fruta", session_state=state)

    cold_start_events = [event for event in events if "se descargan los pesos" in event[1]]
    assert len(cold_start_events) == 1
    assert load_calls
    assert events.index(cold_start_events[0]) < next(index for index, event in enumerate(events) if event[0] == "load")


@pytest.mark.parametrize(
    ("environment", "is_cloud", "webcam", "streaming", "devices"),
    [
        ("local", False, True, True, ("auto", "cpu")),
        ("cloud", True, False, False, ("cpu",)),
    ],
)
def test_entry_point_capability_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    is_cloud: bool,
    webcam: bool,
    streaming: bool,
    devices: tuple[str, ...],
) -> None:
    monkeypatch.setattr(environment_module, "_cuda_available", lambda: False)
    (tmp_path / "best.pt").touch()

    capabilities = detect_capabilities(tmp_path, {"YOLOLAB_ENV": environment})

    assert capabilities.is_cloud is is_cloud
    assert capabilities.webcam is webcam
    assert capabilities.streaming is streaming
    assert capabilities.custom_weights
    assert capabilities.device_options == devices
    assert capabilities.device_default == "cpu"


def test_app_renders_environment_gated_routes_sources_and_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = SimpleNamespace(is_cloud=False, webcam=True, streaming=True, custom_weights=True, device_options=("auto", "cpu"), device_default="cpu")
    cloud = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu")

    local_app, local_widgets = load_app(monkeypatch, local)
    cloud_app, cloud_widgets = load_app(monkeypatch, cloud)

    assert set(local_app.PRESET_MODELS) == {"YOLO en vivo", "Comparación CNN vs YOLO", "Pesos personalizados"}
    assert set(cloud_app.PRESET_MODELS) == {"Comparación CNN vs YOLO"}
    local_source = next(options for label, options in local_widgets if label == "Fuente de frames")
    cloud_source = next(options for label, options in cloud_widgets if label == "Fuente de frames")
    assert "Webcam local (OpenCV)" in local_source
    assert "Webcam local (OpenCV)" not in cloud_source
    assert next(options for label, options in local_widgets if label == "Dispositivo") == ("auto", "cpu")
    assert next(options for label, options in cloud_widgets if label == "Dispositivo") == ("cpu",)


def test_failed_webcam_open_releases_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("app_under_test", ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    app_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, app_module)
    spec.loader.exec_module(app_module)

    class FakeCapture:
        def __init__(self) -> None:
            self.release_calls = 0

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            self.release_calls += 1

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(VideoCapture=lambda _: capture)

    monkeypatch.setattr(app_module, "CAPABILITIES", SimpleNamespace(streaming=True))
    errors: list[str] = []
    monkeypatch.setattr(app_module.st, "error", errors.append)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    result = app_module.run_webcam_benchmark_streaming(
        loaded=object(),
        imgsz=32,
        confidence=0.25,
        iou=0.45,
        device="cpu",
        camera_index=7,
        measure_frames=1,
    )

    assert result == {}
    assert capture.release_calls == 1
    assert errors == ["No se pudo abrir la webcam índice 7. Revisá permisos, conexión y disponibilidad de la cámara."]


def test_operational_states_are_actionable_and_cold_start_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu")
    app_module, _ = load_app(monkeypatch, capabilities)
    fake_streamlit = MagicMock()
    fake_streamlit.session_state = {}
    monkeypatch.setattr(app_module, "st", fake_streamlit)

    app_module.render_cold_start_notice()
    app_module.render_cold_start_notice()

    assert fake_streamlit.info.call_count == 1
    assert "se descargan los pesos" in fake_streamlit.info.call_args.args[0]
    app_module.render_benchmark_results(pd.DataFrame())
    assert "Todavía no hay evidencia de benchmark" in fake_streamlit.info.call_args.args[0]
    for state, renderer in (("empty", fake_streamlit.info), ("partial", fake_streamlit.warning), ("failure", fake_streamlit.error), ("active_stream", fake_streamlit.info)):
        app_module.render_operational_state(state, "Recovery detail.")
        assert "Recovery detail." in renderer.call_args.args[0]


def test_streaming_read_failure_is_bounded_and_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("app_stream_failure", ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    app_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, app_module)
    spec.loader.exec_module(app_module)

    class FakeCapture:
        def __init__(self) -> None:
            self.release_calls = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, object]:
            return False, None

        def release(self) -> None:
            self.release_calls += 1

    capture = FakeCapture()
    stop_column = MagicMock()
    stop_column.button.return_value = False
    stop_placeholder = MagicMock()
    stop_placeholder.columns.return_value = [MagicMock(), stop_column]
    fake_streamlit = MagicMock()
    fake_streamlit.session_state = {}
    fake_streamlit.empty.side_effect = [MagicMock(), MagicMock(), stop_placeholder]
    errors: list[str] = []
    fake_streamlit.error.side_effect = errors.append
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    monkeypatch.setattr(app_module, "CAPABILITIES", SimpleNamespace(streaming=True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=lambda _: capture))

    result = app_module.run_webcam_benchmark_streaming(object(), 32, 0.25, 0.45, "cpu", 3, 1)

    assert result == {}
    assert capture.release_calls == 1
    assert any("dejó de entregar frames" in message for message in errors)


def test_streaming_predict_failures_are_bounded_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("app_stream_predict_failure", ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    app_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, app_module)
    spec.loader.exec_module(app_module)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self) -> None:
            self.read_calls = 0
            self.release_calls = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            self.read_calls += 1
            return True, frame

        def release(self) -> None:
            self.release_calls += 1

    class FailingModel:
        def __init__(self) -> None:
            self.predict_calls = 0

        def predict(self, **_: object) -> list[object]:
            self.predict_calls += 1
            raise RuntimeError("predict failed")

    capture = FakeCapture()
    model = FailingModel()
    stop_column = MagicMock()
    stop_column.button.return_value = False
    stop_placeholder = MagicMock()
    stop_placeholder.columns.return_value = [MagicMock(), stop_column]
    fake_streamlit = MagicMock()
    fake_streamlit.session_state = {}
    fake_streamlit.empty.side_effect = [MagicMock(), MagicMock(), stop_placeholder]
    errors: list[str] = []
    warnings: list[str] = []
    fake_streamlit.error.side_effect = errors.append
    fake_streamlit.warning.side_effect = warnings.append
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    monkeypatch.setattr(app_module, "CAPABILITIES", SimpleNamespace(streaming=True))
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda _: capture,
            resize=lambda *_args, **_kwargs: frame,
            INTER_LINEAR=0,
        ),
    )

    loaded = SimpleNamespace(
        spec=SimpleNamespace(backend="ultralytics"),
        model=model,
    )
    result = app_module.run_webcam_benchmark_streaming(
        loaded, 32, 0.25, 0.45, "cpu", 3, measure_frames=None
    )

    assert result == {}
    assert model.predict_calls == 3
    assert capture.read_calls == 3
    assert capture.release_calls == 1
    assert any("errores repetidos de frame" in message for message in errors)
    assert len(warnings) == 3
    assert fake_streamlit.session_state["streaming_active"] is False
    assert fake_streamlit.session_state["stream_stop_requested"] is False


def test_download_labels_identify_complete_and_live_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu")
    app_module, _ = load_app(monkeypatch, capabilities)
    fake_streamlit = MagicMock()
    fake_streamlit.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]
    fake_streamlit.session_state = {"annotated_frames": {}}
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    results = pd.DataFrame([{"model": "YOLO", "family": "YOLO", "latency_mean_ms": 20.0, "fps_effective": 50.0, "gflops_approx": 2.0, "parameters_millions": 1.5, "detections_mean": 2.0, "frames_measured": 2, "inference_mean_ms": 12.0, "latency_p95_ms": 25.0}])

    app_module.render_benchmark_results(results, "/tmp/benchmark.csv", presentation_mode=True)
    app_module.render_live_yolo_results(results, "/tmp/live.csv", presentation_mode=True)

    labels = [call.args[0] for call in fake_streamlit.download_button.call_args_list]
    assert "Descargar CSV de resultados" in labels
    assert "Descargar CSV del resumen" in labels
    assert fake_streamlit.download_button.call_args_list[-1].kwargs["file_name"] == "yolo_live_summary.csv"


def test_partial_benchmark_preserves_successes_and_explains_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yolo_complexity_lab import benchmark as benchmark_module, exporting as exporting_module, loaders as loaders_module, sources as sources_module
    capabilities = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    failed_key = "fasterrcnn_mobilenet_fpn"

    def fake_load(model_key: str, device: str):
        if model_key == failed_key:
            raise RuntimeError("weights unavailable")
        return SimpleNamespace(spec=type("Spec", (), {"key": model_key, "display_name": model_key, "family": "YOLO"})(), device=device)

    def fake_benchmark(loaded, frames, config, include_complexity=True):
        return {"model": loaded.spec.display_name, "family": loaded.spec.family, "latency_mean_ms": 10.0, "fps_effective": 100.0, "gflops_approx": 1.0, "parameters_millions": 2.0, "detections_mean": 1.0}

    monkeypatch.setattr(loaders_module, "load_model", fake_load)
    monkeypatch.setattr(benchmark_module, "benchmark_model", fake_benchmark)
    monkeypatch.setattr(sources_module, "sample_coco_frame", lambda: frame)
    monkeypatch.setattr(sources_module, "repeat_frame", lambda value, count: [value] * count)
    monkeypatch.setattr(exporting_module, "write_results_csv", lambda _df: tmp_path / "partial.csv")
    errors: list[str] = []
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(streamlit, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(streamlit, "info", lambda message: infos.append(str(message)))
    monkeypatch.setattr(streamlit, "warning", lambda message: warnings.append(str(message)))

    app_module, _ = load_app(monkeypatch, capabilities, run=True)

    assert app_module.st.session_state["last_benchmark_df"].shape[0] == 2
    assert any("weights unavailable" in message and "reintentá" in message for message in errors)
    assert any("resultados parciales disponibles" in message and "2 de 3" in message for message in warnings)
    assert sum("se descargan los pesos" in message for message in infos) == 1


def test_reachable_live_results_render_spanish_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu"); app_module, _ = load_app(monkeypatch, capabilities)
    fake_streamlit = MagicMock(); fake_streamlit.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]; monkeypatch.setattr(app_module, "st", fake_streamlit)
    results = pd.DataFrame([{"model": "YOLO", "latency_mean_ms": 20.0, "fps_effective": 50.0, "detections_mean": 2.0, "gflops_approx": 2.0, "parameters_millions": 1.5, "frames_measured": 4, "inference_mean_ms": 12.0, "latency_p95_ms": 25.0}]); app_module.render_live_yolo_results(results, "/tmp/benchmark.csv")
    rendered = " ".join(str(call.args[0]) for call in fake_streamlit.markdown.call_args_list); assert "Resumen en vivo de YOLO11n" in rendered; assert "Practical YOLO live summary" not in rendered

def test_streaming_success_preserves_result_export_and_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from yolo_complexity_lab import exporting as exporting_module, loaders as loaders_module; capabilities = SimpleNamespace(is_cloud=False, webcam=True, streaming=True, custom_weights=False, device_options=("cpu",), device_default="cpu"); frame = np.zeros((2, 2, 3), dtype=np.uint8); releases: list[bool] = []; visible: list[str] = []
    capture = SimpleNamespace(isOpened=lambda: True, read=lambda: (True, frame), release=lambda: releases.append(True)); fake_cv2 = SimpleNamespace(VideoCapture=lambda _: capture, resize=lambda *_args, **_kwargs: frame, cvtColor=lambda *_: frame, INTER_LINEAR=0, COLOR_BGR2RGB=0)
    prediction = SimpleNamespace(boxes=[], names={}, speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0}, plot=lambda: frame); loaded = SimpleNamespace(spec=SimpleNamespace(key="yolo11n", backend="ultralytics"), model=SimpleNamespace(predict=lambda **_: [prediction]), device="cpu", parameter_count=1000, model_size_mb=1.0, size_note="fake")
    writer_path = tmp_path / "stream.csv"; written: list[pd.DataFrame] = []; writer = lambda df: (written.append(df.copy()), df.to_csv(writer_path, index=False), writer_path)[-1]
    placeholders = [MagicMock() for _ in range(3)]; placeholders[2].columns.return_value = [MagicMock(), MagicMock()]; placeholders[2].columns.return_value[1].button.return_value = False; placeholders[1].download_button.side_effect = lambda label, *_args, **_kwargs: (visible.append(str(label)), streamlit.session_state.__setitem__("stream_stop_requested", True))[-1]
    for name in ("info", "success", "warning", "error", "markdown", "write", "metric"): monkeypatch.setattr(streamlit, name, lambda *args, **_kwargs: visible.extend(map(str, args)))
    monkeypatch.setattr(streamlit, "checkbox", lambda label, **_: label == "Modo en vivo"); monkeypatch.setattr(streamlit, "empty", MagicMock(side_effect=placeholders)); monkeypatch.setitem(sys.modules, "cv2", fake_cv2); monkeypatch.setattr(loaders_module, "load_model", lambda *_: loaded); monkeypatch.setattr(exporting_module, "write_results_csv", writer)
    app_module, _ = load_app(monkeypatch, capabilities, run=True); result = app_module.st.session_state["last_benchmark_df"]; assert releases == [True]; assert result.iloc[0]["frames_measured"] == 1; assert result.iloc[0]["model"] == "YOLO11n — ligero"; assert written[0].equals(result); assert writer_path.read_text().startswith("model_key,"); assert app_module.st.session_state["last_benchmark_csv_path"] == str(writer_path); assert any("En vivo: YOLO11n" in text for text in visible); assert "Streaming finalizado. Resumen de rendimiento:" in visible; assert "Descargar CSV parcial" in visible; assert all(label in visible for label in ("Latencia media", "FPS efectivo", "Frames medidos", "Detecciones promedio", "Preprocesamiento", "Inferencia")); assert not any("Performance summary" in text for text in visible)
def test_standard_benchmark_results_render_spanish_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu"); app_module, _ = load_app(monkeypatch, capabilities)
    fake_streamlit = MagicMock(); fake_streamlit.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]; fake_streamlit.session_state = {}; monkeypatch.setattr(app_module, "st", fake_streamlit)
    results = pd.DataFrame([{"model": "YOLO", "family": "YOLO", "latency_mean_ms": 20.0, "fps_effective": 50.0, "gflops_approx": 2.0, "parameters_millions": 1.5, "detections_mean": 2.0}]); app_module.render_benchmark_results(results, presentation_mode=True)
    rendered = " ".join(str(call.args[0]) for call in fake_streamlit.markdown.call_args_list); assert "Resumen comparativo" in rendered; assert fake_streamlit.plotly_chart.call_args.args[0].layout.xaxis.title.text == "GFLOPs aproximados"
    app_module.render_config_summary(["yolo11n"], "Person/dog/fruit demo", "cpu", 416, 3, 20, True); app_module.render_benchmark_focus(416, True, "YOLO live"); app_module.render_result_interpretation(results); rendered = " ".join(str(call.args[0]) for call in fake_streamlit.markdown.call_args_list)
    assert "Configuración actual" in rendered and "Interpretación" in rendered and "Qué observar" in rendered and "YOLO en vivo" in rendered and "What to watch" not in rendered


def test_cloud_benchmark_preserves_session_and_csv_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yolo_complexity_lab import benchmark as benchmark_module, catalog as catalog_module, exporting as exporting_module, loaders as loaders_module, sources as sources_module

    cloud = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    calls: list[tuple[str, tuple[object, ...], object]] = []

    def fake_load(model_key: str, device: str):
        return SimpleNamespace(spec=catalog_module.MODEL_CATALOG[model_key], device=device)

    def fake_benchmark(loaded, frames, config, include_complexity=True):
        calls.append((loaded.spec.key, tuple(frames), config))
        return {"model": loaded.spec.display_name, "family": loaded.spec.family, "latency_mean_ms": 10.0, "latency_p95_ms": 12.0, "fps_effective": 100.0, "gflops_approx": 1.0, "parameters_millions": 2.0, "recognized_classes": "person", "avg_confidence": 0.9, "input_size_px": config.imgsz, "detections_mean": 1.0}

    monkeypatch.setattr(loaders_module, "load_model", fake_load)
    monkeypatch.setattr(benchmark_module, "benchmark_model", fake_benchmark)
    monkeypatch.setattr(sources_module, "sample_coco_frame", lambda: frame)
    monkeypatch.setattr(sources_module, "repeat_frame", lambda value, count: [value] * count)
    monkeypatch.setattr(exporting_module, "write_results_csv", lambda _df: tmp_path / "results.csv")

    app_module, _ = load_app(monkeypatch, cloud, run=True)

    assert [key for key, _, _ in calls] == app_module.PRESET_MODELS["Comparación CNN vs YOLO"]
    assert all(frames for _, frames, _ in calls)
    assert all(config.measure_frames == 20 and config.warmup_frames == 3 for _, _, config in calls)
    assert app_module.st.session_state["last_benchmark_df"].shape[0] == len(calls)
    assert app_module.st.session_state["last_benchmark_csv_path"] == str(tmp_path / "results.csv")
    assert list(app_module.st.session_state["last_benchmark_df"]["model"]) == [
        streamlit_app_model.display_name for streamlit_app_model in [
            catalog_module.MODEL_CATALOG[key] for key in app_module.PRESET_MODELS["Comparación CNN vs YOLO"]
        ]
    ]


def test_comparison_winner_deltas_preserve_benchmark_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = SimpleNamespace(
        is_cloud=True,
        webcam=False,
        streaming=False,
        custom_weights=False,
        device_options=("cpu",),
        device_default="cpu",
    )
    app_module, _ = load_app(monkeypatch, capabilities)
    results = pd.DataFrame(
        [
            {"model": "Classic CNN", "latency_mean_ms": 40.0, "fps_effective": 25.0, "gflops_approx": 4.0},
            {"model": "YOLO", "latency_mean_ms": 20.0, "fps_effective": 50.0, "gflops_approx": 2.0},
        ]
    )
    before = results.copy(deep=True)

    winners = app_module.comparison_winner_deltas(results)

    assert winners["latency_mean_ms"]["model"] == "YOLO"
    assert winners["latency_mean_ms"]["delta_pct"] == 50.0
    assert winners["fps_effective"]["model"] == "YOLO"
    assert winners["gflops_approx"]["model"] == "YOLO"
    pd.testing.assert_frame_equal(results, before)

    fake_streamlit = MagicMock()
    metric_columns = [MagicMock() for _ in range(3)]
    fake_streamlit.columns.return_value = metric_columns
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    app_module.render_comparison_presentation(results)

    rendered = " ".join(str(call.args[0]) for call in fake_streamlit.markdown.call_args_list)
    assert "Comparación medida" in rendered
    assert "Conclusión medida" in rendered
    assert metric_columns[0].metric.call_args.args[2] == "50.0% menor que el siguiente modelo"
    assert metric_columns[1].metric.call_args.args[2] == "100.0% mayor que el siguiente modelo"
    assert metric_columns[2].metric.call_args.args[2] == "50.0% menor que el siguiente modelo"


def test_single_model_conclusion_does_not_claim_comparison_lead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = SimpleNamespace(
        is_cloud=True,
        webcam=False,
        streaming=False,
        custom_weights=False,
        device_options=("cpu",),
        device_default="cpu",
    )
    app_module, _ = load_app(monkeypatch, capabilities)
    results = pd.DataFrame(
        [{"model": "YOLO", "latency_mean_ms": 20.0, "fps_effective": 50.0, "gflops_approx": 2.0}]
    )

    fake_streamlit = MagicMock()
    fake_streamlit.columns.return_value = [MagicMock() for _ in range(3)]
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    app_module.render_comparison_presentation(results)

    rendered = " ".join(str(call.args[0]) for call in fake_streamlit.markdown.call_args_list)
    assert "Solo se midió un modelo: YOLO." in rendered
    assert "leads this measured comparison" not in rendered


def test_navigation_shell_orders_tabs_and_exposes_author_and_evidence_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = SimpleNamespace(
        is_cloud=True,
        webcam=False,
        streaming=False,
        custom_weights=False,
        device_options=("cpu",),
        device_default="cpu",
    )
    app_module, _ = load_app(monkeypatch, capabilities)

    tab_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        app_module.st,
        "tabs",
        lambda labels: tab_calls.append(tuple(labels)) or ("overview", "benchmark", "about"),
    )
    assert app_module.render_navigation_tabs() == ("overview", "benchmark", "about")
    assert tab_calls == [("Resumen", "Benchmark", "Acerca de")]

    fake_streamlit = MagicMock()
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    app_module.render_commercial_header()
    app_module.render_evidence_path()

    rendered = " ".join(str(call.args[0]) for call in fake_streamlit.markdown.call_args_list)
    assert "YOLO Complexity Lab" in rendered and app_module.AUTHOR_NAME == "Alejandro Padilla"
    assert "La evidencia no está disponible" in str(fake_streamlit.info.call_args.args[0])

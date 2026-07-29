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


def load_app(monkeypatch: pytest.MonkeyPatch, capabilities: SimpleNamespace, run: bool = False):
    widgets: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(environment_module, "detect_capabilities", lambda *_: capabilities)
    monkeypatch.setattr(streamlit, "session_state", {})
    monkeypatch.setattr(streamlit, "cache_resource", lambda **_: lambda function: function)

    def radio(label: str, options, **_: object):
        values = tuple(options)
        widgets.append((label, values))
        return values[0]

    def selectbox(label: str, options, index: int = 0, **_: object):
        values = tuple(options)
        widgets.append((label, values))
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

    assert set(local_app.PRESET_MODELS) == {"Live YOLO", "CNN vs YOLO comparison", "Custom weights"} and set(cloud_app.PRESET_MODELS) == {"CNN vs YOLO comparison"}
    local_source = next(options for label, options in local_widgets if label == "Frame source")
    cloud_source = next(options for label, options in cloud_widgets if label == "Frame source")
    assert "Local OpenCV webcam" in local_source and "Local OpenCV webcam" not in cloud_source
    assert next(options for label, options in local_widgets if label == "Execution device") == ("auto", "cpu") and next(options for label, options in cloud_widgets if label == "Execution device") == ("cpu",)


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
    monkeypatch.setattr(app_module.st, "error", lambda _: None)
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

    assert [key for key, _, _ in calls] == app_module.PRESET_MODELS["CNN vs YOLO comparison"]
    assert all(frames for _, frames, _ in calls)
    assert all(config.measure_frames == 20 and config.warmup_frames == 3 for _, _, config in calls)
    assert app_module.st.session_state["last_benchmark_df"].shape[0] == len(calls)
    assert app_module.st.session_state["last_benchmark_csv_path"] == str(tmp_path / "results.csv")

CLOUD = SimpleNamespace(is_cloud=True, webcam=False, streaming=False, custom_weights=False, device_options=("cpu",), device_default="cpu")

def test_rendered_result_and_streaming_surfaces_use_english_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    app_module, _ = load_app(monkeypatch, CLOUD)
    st = app_module.st = MagicMock(); st.__enter__.return_value = st; st.__exit__.return_value = False; st.columns.side_effect = lambda spec: [st] * (len(spec) if isinstance(spec, list) else spec); st.expander.return_value = st; st.session_state = {"annotated_frames": {}}
    frame = {"model": "YOLO11n", "family": "YOLO", "latency_mean_ms": 10.0, "latency_p95_ms": 12.0, "fps_effective": 100.0, "gflops_approx": 1.0, "parameters_millions": 2.0, "recognized_classes": "person", "top_detection": "person", "avg_confidence": 0.9, "input_size_px": 32, "detections_mean": 1.0, "preprocess_mean_ms": 1.0, "inference_mean_ms": 2.0}; df = pd.DataFrame([frame])
    app_module.render_benchmark_results(df, "/tmp/results.csv", True); app_module.render_live_yolo_results(df, "/tmp/results.csv", True)
    labels = [call.args[0] for call in st.metric.call_args_list + st.download_button.call_args_list]; tables = [call.args[0] for call in st.dataframe.call_args_list]
    assert {"Model", "Family", "Mean latency (ms)", "Recognized classes"} <= set(tables[0].columns) and {"preprocess_mean_ms", "inference_mean_ms"} <= set(tables[-1].columns)
    assert {"Lowest mean latency", "Highest effective FPS", "Mean latency", "Mean inference", "Frames processed"} <= set(labels) and [call.args[0].layout.title.text for call in st.plotly_chart.call_args_list] == ["Latency by model", "Effective FPS", "Computational complexity vs runtime"]
    assert "Recognition by model" in " ".join(str(call.args[0]) for call in st.markdown.call_args_list) and "Recognized:" in " ".join(str(call.args[0]) for call in st.markdown.call_args_list)

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_complexity_lab import benchmark
from yolo_complexity_lab.benchmark import BenchmarkConfig, FrameTiming, benchmark_model


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "benchmark_baseline"
INTEGER_FIELDS = {
    "input_size_px",
    "frames_measured",
    "warmup_frames",
    "detections_total",
    "parameters",
    "macs",
    "conv_layers_counted",
    "linear_layers_counted",
}
FLOAT_FIELDS = {
    "latency_mean_ms",
    "latency_median_ms",
    "latency_min_ms",
    "latency_max_ms",
    "latency_p95_ms",
    "fps_effective",
    "preprocess_mean_ms",
    "inference_mean_ms",
    "postprocess_mean_ms",
    "detections_mean",
    "avg_confidence",
    "avg_confidence_all_frames",
    "parameters_millions",
    "model_size_mb",
    "gmacs_approx",
    "gflops_approx",
    "ram_delta_mb",
}
CSV_FIELDS = [
    "model_key",
    "model",
    "family",
    "backend",
    "device",
    "input_size_px",
    "frames_measured",
    "warmup_frames",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_min_ms",
    "latency_max_ms",
    "latency_p95_ms",
    "fps_effective",
    "preprocess_mean_ms",
    "inference_mean_ms",
    "postprocess_mean_ms",
    "detections_mean",
    "detections_total",
    "recognized_classes",
    "top_detection",
    "avg_confidence",
    "avg_confidence_all_frames",
    "parameters",
    "parameters_millions",
    "model_size_mb",
    "model_size_note",
    "macs",
    "gmacs_approx",
    "gflops_approx",
    "conv_layers_counted",
    "linear_layers_counted",
    "complexity_note",
    "big_o_inference",
    "big_o_didactic",
    "big_o_postprocess",
    "ram_delta_mb",
]


def _load_inputs() -> dict[str, object]:
    return json.loads((FIXTURE_DIR / "inputs.json").read_text(encoding="utf-8"))


def _fixture_model(data: dict[str, object]) -> SimpleNamespace:
    model = data["model"]
    assert isinstance(model, dict)
    spec = SimpleNamespace(
        key=model["key"],
        display_name=model["display_name"],
        family=model["family"],
        backend=model["backend"],
        inference_big_o="O(test)",
        didactic_big_o="O(test didactic)",
        postprocess_big_o="O(test postprocess)",
    )
    return SimpleNamespace(
        spec=spec,
        model=object(),
        device=model["device"],
        parameter_count=model["parameter_count"],
        model_size_mb=model["model_size_mb"],
        size_note=model["size_note"],
        class_names={0: "person", 1: "dog"},
    )


def test_benchmark_matches_frozen_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _load_inputs()
    config_data = data["config"]
    assert isinstance(config_data, dict)
    frames = data["frames"]
    assert isinstance(frames, list)

    timings = {
        1: FrameTiming(1.0, 10.0, 0.5, 12.0, 1, ("person",), (0.8,)),
        2: FrameTiming(2.0, 20.0, 1.0, 24.0, 2, ("dog", "person"), (0.7, 0.6)),
        3: FrameTiming(3.0, 30.0, 1.5, 36.0, 3, ("person",), (0.5,)),
    }

    monkeypatch.setattr(benchmark, "run_frame", lambda _loaded, frame, _config: (timings[frame], frame))
    monkeypatch.setattr(
        benchmark,
        "estimate_for_loaded_model",
        lambda _loaded, _input_size: SimpleNamespace(
            macs=123456,
            gmacs=0.1235,
            gflops_approx=0.2469,
            conv_layers=3,
            linear_layers=1,
            note="fixture complexity",
        ),
    )
    monkeypatch.setattr(benchmark.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1024)))

    result = benchmark_model(
        _fixture_model(data),
        frames,
        BenchmarkConfig(**config_data),
    )
    result.pop("last_annotated_frame")
    actual = {field: str(result[field]) for field in CSV_FIELDS}

    with (FIXTURE_DIR / "expected.csv").open(newline="", encoding="utf-8") as stream:
        expected_rows = list(csv.DictReader(stream))

    assert len(expected_rows) == 1
    expected = expected_rows[0]
    assert list(expected) == CSV_FIELDS
    assert set(actual) == set(expected)
    for field in CSV_FIELDS:
        if field in INTEGER_FIELDS:
            assert int(actual[field]) == int(expected[field]), field
        elif field in FLOAT_FIELDS:
            assert float(actual[field]) == pytest.approx(float(expected[field]), rel=1e-9, abs=1e-12), field
        else:
            assert actual[field] == expected[field], field

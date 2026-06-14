from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Callable

import psutil

from .complexity import ComplexityEstimate, estimate_for_loaded_model
from .loaders import LoadedModel


@dataclass(frozen=True)
class BenchmarkConfig:
    imgsz: int = 640
    warmup_frames: int = 5
    measure_frames: int = 50
    confidence: float = 0.25
    iou: float = 0.45


@dataclass(frozen=True)
class FrameTiming:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    detections: int


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def _sync_if_cuda(device: str) -> None:
    if "cuda" not in str(device):
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


def _resize_rgb(frame_rgb: object, imgsz: int) -> object:
    import cv2

    return cv2.resize(frame_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)


# MODIFICACIÓN: Ahora retorna una tupla (FrameTiming, imagen_anotada)
def run_yolo_frame(loaded: LoadedModel, frame_rgb: object, config: BenchmarkConfig) -> tuple[FrameTiming, object]:
    start_total = time.perf_counter()
    start_pre = time.perf_counter()
    resized = _resize_rgb(frame_rgb, config.imgsz)
    preprocess_ms = (time.perf_counter() - start_pre) * 1000

    _sync_if_cuda(loaded.device)
    start_infer = time.perf_counter()
    results = loaded.model.predict(
        source=resized,
        imgsz=config.imgsz,
        conf=config.confidence,
        iou=config.iou,
        device=loaded.device,
        verbose=False,
    )
    _sync_if_cuda(loaded.device)
    measured_model_ms = (time.perf_counter() - start_infer) * 1000

    # MODIFICACIÓN: Extraer la imagen anotada con las predicciones de YOLO
    annotated_frame = results[0].plot()

    # Ultralytics exposes internal stage timings. Prefer them when available.
    inference_ms = measured_model_ms
    postprocess_ms = 0.0
    detections = 0
    try:
        speed = getattr(results[0], "speed", {}) or {}
        preprocess_ms = float(speed.get("preprocess", preprocess_ms))
        inference_ms = float(speed.get("inference", measured_model_ms))
        postprocess_ms = float(speed.get("postprocess", 0.0))
        detections = len(getattr(results[0], "boxes", []) or [])
    except Exception:
        pass

    total_ms = (time.perf_counter() - start_total) * 1000
    return FrameTiming(preprocess_ms, inference_ms, postprocess_ms, total_ms, detections), annotated_frame


# MODIFICACIÓN: Retorna tupla para coincidir con la firma
def run_torchvision_frame(loaded: LoadedModel, frame_rgb: object, config: BenchmarkConfig) -> tuple[FrameTiming, object]:
    import torch

    start_total = time.perf_counter()
    start_pre = time.perf_counter()
    resized = _resize_rgb(frame_rgb, config.imgsz)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0).to(loaded.device)
    preprocess_ms = (time.perf_counter() - start_pre) * 1000

    _sync_if_cuda(loaded.device)
    start_infer = time.perf_counter()
    with torch.inference_mode():
        outputs = loaded.model([tensor])
    _sync_if_cuda(loaded.device)
    inference_ms = (time.perf_counter() - start_infer) * 1000

    start_post = time.perf_counter()
    detections = 0
    try:
        scores = outputs[0].get("scores", [])
        detections = int((scores >= config.confidence).sum().item())
    except Exception:
        pass
    postprocess_ms = (time.perf_counter() - start_post) * 1000
    total_ms = (time.perf_counter() - start_total) * 1000
    
    # Retornamos el frame original ya que torchvision no tiene un .plot() directo
    return FrameTiming(preprocess_ms, inference_ms, postprocess_ms, total_ms, detections), resized


def run_frame(loaded: LoadedModel, frame_rgb: object, config: BenchmarkConfig) -> tuple[FrameTiming, object]:
    if loaded.spec.backend == "ultralytics":
        return run_yolo_frame(loaded, frame_rgb, config)
    return run_torchvision_frame(loaded, frame_rgb, config)


# MODIFICACIÓN: Añadido el parámetro frame_callback
def benchmark_model(
    loaded: LoadedModel,
    frames: Iterable[object],
    config: BenchmarkConfig,
    include_complexity: bool = True,
    frame_callback: Callable[[object], None] | None = None,
) -> dict[str, object]:
    """Benchmark one loaded model with warmup and measured frames."""
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No hay frames para medir.")

    warmup_source = frame_list[: max(0, config.warmup_frames)] or [frame_list[0]]
    for frame in warmup_source:
        # En warmup no enviamos la imagen a la UI
        run_frame(loaded, frame, config)

    measured_source = frame_list[: max(1, config.measure_frames)]
    ram_before = psutil.Process().memory_info().rss / (1024**2)
    
    timings = []
    # MODIFICACIÓN: Ejecutar y enviar el frame al callback si existe
    for frame in measured_source:
        timing, annotated_frame = run_frame(loaded, frame, config)
        timings.append(timing)
        if frame_callback is not None:
            frame_callback(annotated_frame)
            
    ram_after = psutil.Process().memory_info().rss / (1024**2)

    totals = [t.total_ms for t in timings]
    preprocess = [t.preprocess_ms for t in timings]
    inference = [t.inference_ms for t in timings]
    postprocess = [t.postprocess_ms for t in timings]
    detections = [t.detections for t in timings]
    total_seconds = sum(totals) / 1000
    fps = len(totals) / total_seconds if total_seconds > 0 else None

    complexity: ComplexityEstimate | None = estimate_for_loaded_model(loaded, config.imgsz) if include_complexity else None

    return {
        "model_key": loaded.spec.key,
        "model": loaded.spec.display_name,
        "family": loaded.spec.family,
        "backend": loaded.spec.backend,
        "device": loaded.device,
        "input_size_px": config.imgsz,
        "frames_measured": len(totals),
        "warmup_frames": config.warmup_frames,
        "latency_mean_ms": round(statistics.mean(totals), 3),
        "latency_median_ms": round(statistics.median(totals), 3),
        "latency_min_ms": round(min(totals), 3),
        "latency_max_ms": round(max(totals), 3),
        "latency_p95_ms": round(_percentile(totals, 95) or 0, 3),
        "fps_effective": round(fps, 3) if fps is not None else None,
        "preprocess_mean_ms": round(statistics.mean(preprocess), 3),
        "inference_mean_ms": round(statistics.mean(inference), 3),
        "postprocess_mean_ms": round(statistics.mean(postprocess), 3),
        "detections_mean": round(statistics.mean(detections), 3),
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
        "big_o_inference": loaded.spec.inference_big_o,
        "big_o_didactic": loaded.spec.didactic_big_o,
        "big_o_postprocess": loaded.spec.postprocess_big_o,
        "ram_delta_mb": round(ram_after - ram_before, 3),
    }
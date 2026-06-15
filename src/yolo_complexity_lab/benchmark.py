from __future__ import annotations

import cv2
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Callable

import psutil

from .complexity import ComplexityEstimate, estimate_for_loaded_model
from .loaders import LoadedModel


DETECTION_ORANGE_BGR = (0, 102, 255)
DETECTION_TEXT_BGR = (255, 255, 255)


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
    detection_labels: tuple[str, ...] = ()
    detection_confidences: tuple[float, ...] = ()


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


def _class_name(loaded: LoadedModel, class_id: int) -> str:
    names = loaded.class_names
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, f"clase_{class_id}"))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
    except Exception:
        pass
    return f"clase_{class_id}"


def _summarize_detection_names(labels: list[str]) -> str:
    if not labels:
        return "Sin detecciones"
    counts = Counter(labels)
    return ", ".join(f"{label} ({count})" for label, count in counts.most_common())


def _draw_torchvision_detections(frame_rgb: object, outputs: dict, loaded: LoadedModel, confidence: float) -> object:
    """Draw torchvision detections and return a BGR image.

    The app converts this BGR result to RGB immediately before calling
    ``st.image``. Keeping one explicit convention prevents the blue tint caused
    by mixing OpenCV's BGR arrays with Streamlit's RGB rendering.
    """
    import cv2

    annotated_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    boxes = outputs.get("boxes", [])
    labels = outputs.get("labels", [])
    scores = outputs.get("scores", [])

    try:
        iterable = zip(boxes.detach().cpu().numpy(), labels.detach().cpu().numpy(), scores.detach().cpu().numpy(), strict=False)
    except Exception:
        iterable = []

    for box, class_id, score in iterable:
        if float(score) < confidence:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        name = _class_name(loaded, int(class_id))
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), DETECTION_ORANGE_BGR, 2)
        label = f"{name} {float(score):.2f}"
        cv2.rectangle(annotated_bgr, (x1, max(0, y1 - 22)), (x1 + min(220, 9 * len(label)), y1), DETECTION_ORANGE_BGR, -1)
        cv2.putText(
            annotated_bgr,
            label,
            (x1 + 4, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            DETECTION_TEXT_BGR,
            1,
            cv2.LINE_AA,
        )
    return annotated_bgr


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

    # Ultralytics exposes internal stage timings. Prefer them when available.
    inference_ms = measured_model_ms
    postprocess_ms = 0.0
    detections = 0
    labels: list[str] = []
    confidences: list[float] = []
    try:
        speed = getattr(results[0], "speed", {}) or {}
        preprocess_ms = float(speed.get("preprocess", preprocess_ms))
        inference_ms = float(speed.get("inference", measured_model_ms))
        postprocess_ms = float(speed.get("postprocess", 0.0))
        boxes = getattr(results[0], "boxes", None)
        detections = len(boxes) if boxes is not None else 0
        if boxes is not None:
            cls_values = getattr(boxes, "cls", [])
            conf_values = getattr(boxes, "conf", [])
            for class_id in cls_values:
                labels.append(_class_name(loaded, int(class_id.item() if hasattr(class_id, "item") else class_id)))
            for score in conf_values:
                confidences.append(float(score.item() if hasattr(score, "item") else score))
    except Exception:
        pass

    # Dibujar cajas manualmente con color naranja en BGR. No usamos
    # results[0].plot() porque aplica la paleta por defecto de Ultralytics.
    annotated_frame = cv2.cvtColor(resized.copy(), cv2.COLOR_RGB2BGR)
    try:
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            names = getattr(results[0], "names", {})
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                if len(xyxy) >= 4:
                    x1, y1, x2, y2 = xyxy[:4]
                    class_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                    conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                    name = names.get(class_id, "") if isinstance(names, dict) else str(class_id)
                    label = f"{name} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), DETECTION_ORANGE_BGR, 2)
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    label_top = max(0, y1 - text_h - 8)
                    cv2.rectangle(annotated_frame, (x1, label_top), (x1 + text_w + 4, y1), DETECTION_ORANGE_BGR, -1)
                    cv2.putText(annotated_frame, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DETECTION_TEXT_BGR, 1, cv2.LINE_AA)
    except Exception:
        pass

    total_ms = (time.perf_counter() - start_total) * 1000
    return FrameTiming(preprocess_ms, inference_ms, postprocess_ms, total_ms, detections, tuple(labels), tuple(confidences)), annotated_frame


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
    labels: list[str] = []
    confidences: list[float] = []
    try:
        output = outputs[0]
        scores = output.get("scores", [])
        class_ids = output.get("labels", [])
        keep = scores >= config.confidence
        for class_id in class_ids[keep].detach().cpu().tolist():
            labels.append(_class_name(loaded, int(class_id)))
        confidences = [float(v) for v in scores[keep].detach().cpu().tolist()]
        annotated_frame = _draw_torchvision_detections(resized, output, loaded, config.confidence)
    except Exception:
        import cv2

        annotated_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    return FrameTiming(preprocess_ms, inference_ms, postprocess_ms, total_ms, detections, tuple(labels), tuple(confidences)), annotated_frame


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
    last_annotated_frame = None
    # MODIFICACIÓN: Ejecutar y enviar el frame al callback si existe
    for frame in measured_source:
        timing, annotated_frame = run_frame(loaded, frame, config)
        timings.append(timing)
        last_annotated_frame = annotated_frame
        if frame_callback is not None:
            frame_callback(annotated_frame)
            
    ram_after = psutil.Process().memory_info().rss / (1024**2)

    totals = [t.total_ms for t in timings]
    preprocess = [t.preprocess_ms for t in timings]
    inference = [t.inference_ms for t in timings]
    postprocess = [t.postprocess_ms for t in timings]
    detections = [t.detections for t in timings]
    detection_labels = [label for timing in timings for label in timing.detection_labels]
    detection_confidences = [score for timing in timings for score in timing.detection_confidences]
    representative_labels = list(timings[-1].detection_labels) if timings else []
    representative_confidences = list(timings[-1].detection_confidences) if timings else []
    label_counts = Counter(representative_labels)
    top_detection = label_counts.most_common(1)[0][0] if label_counts else "Sin detecciones"
    recognized_classes = _summarize_detection_names(representative_labels)
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
        "detections_total": int(sum(detections)),
        "recognized_classes": recognized_classes,
        "top_detection": top_detection,
        "avg_confidence": round(statistics.mean(representative_confidences), 3) if representative_confidences else None,
        "avg_confidence_all_frames": round(statistics.mean(detection_confidences), 3) if detection_confidences else None,
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
        "last_annotated_frame": last_annotated_frame,
    }

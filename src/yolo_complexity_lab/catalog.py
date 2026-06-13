from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal["YOLO", "CNN one-stage", "CNN two-stage"]
Backend = Literal["ultralytics", "torchvision"]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata used by the UI and benchmark runner.

    The Big-O fields are intentionally explanatory. The measured evidence is the
    local latency/FPS/GFLOPs reported by the benchmark.
    """

    key: str
    display_name: str
    family: ModelFamily
    backend: Backend
    weight_name: str
    default_imgsz: int
    inference_big_o: str
    didactic_big_o: str
    postprocess_big_o: str
    complexity_note: str
    default_enabled: bool = True


CONVOLUTIONAL_BIG_O = "O(Σ_l H_l × W_l × C_in_l × C_out_l × K_l²)"
DIDACTIC_BIG_O = "O(L × n × C_in × C_out × K²), con n = H × W"
NMS_BIG_O = "O(B²) por NMS, con B = cajas candidatas"
NMS_FREE_BIG_O = "O(B) aproximado si el modelo evita NMS tradicional"
TWO_STAGE_BIG_O = "O(Σ conv + R × C_roi + NMS), con R = regiones propuestas"


MODEL_CATALOG: dict[str, ModelSpec] = {
    "yolo11n": ModelSpec(
        key="yolo11n",
        display_name="YOLO11n — ligero",
        family="YOLO",
        backend="ultralytics",
        weight_name="yolo11n.pt",
        default_imgsz=640,
        inference_big_o=CONVOLUTIONAL_BIG_O,
        didactic_big_o=DIDACTIC_BIG_O,
        postprocess_big_o=NMS_BIG_O,
        complexity_note=(
            "Detector de una etapa: predice cajas y clases en una sola pasada. "
            "El costo dominante viene de convoluciones; NMS agrega costo cuadrático "
            "sobre cajas candidatas."
        ),
    ),
    "yolo11s": ModelSpec(
        key="yolo11s",
        display_name="YOLO11s — mediano",
        family="YOLO",
        backend="ultralytics",
        weight_name="yolo11s.pt",
        default_imgsz=640,
        inference_big_o=CONVOLUTIONAL_BIG_O,
        didactic_big_o=DIDACTIC_BIG_O,
        postprocess_big_o=NMS_BIG_O,
        complexity_note=(
            "Misma familia que YOLO11n, pero con más capacidad. Normalmente suben "
            "parámetros/GFLOPs y puede subir la latencia."
        ),
        default_enabled=False,
    ),
    "ssdlite_mobilenet_v3": ModelSpec(
        key="ssdlite_mobilenet_v3",
        display_name="SSDlite MobileNetV3 — CNN one-stage",
        family="CNN one-stage",
        backend="torchvision",
        weight_name="ssdlite320_mobilenet_v3_large",
        default_imgsz=320,
        inference_big_o=CONVOLUTIONAL_BIG_O,
        didactic_big_o=DIDACTIC_BIG_O,
        postprocess_big_o=NMS_BIG_O,
        complexity_note=(
            "Detector CNN de una etapa con backbone ligero. Sirve como comparación "
            "contra YOLO porque también prioriza tiempo real, aunque suele ser menos "
            "robusto en escenas complejas."
        ),
    ),
    "fasterrcnn_mobilenet_fpn": ModelSpec(
        key="fasterrcnn_mobilenet_fpn",
        display_name="Faster R-CNN MobileNet/FPN — CNN two-stage",
        family="CNN two-stage",
        backend="torchvision",
        weight_name="fasterrcnn_mobilenet_v3_large_320_fpn",
        default_imgsz=320,
        inference_big_o=TWO_STAGE_BIG_O,
        didactic_big_o="O(L × n × C_in × C_out × K² + R × C_roi)",
        postprocess_big_o=NMS_BIG_O,
        complexity_note=(
            "Detector de dos etapas: primero propone regiones y luego clasifica/refina. "
            "Eso mejora flexibilidad, pero agrega costo por región R y suele penalizar "
            "la latencia frente a detectores one-stage."
        ),
        default_enabled=False,
    ),
}


def catalog_rows() -> list[dict[str, str]]:
    return [
        {
            "modelo": spec.display_name,
            "familia": spec.family,
            "backend": spec.backend,
            "pesos": spec.weight_name,
            "Big-O inferencia": spec.inference_big_o,
            "Big-O didáctico": spec.didactic_big_o,
            "Big-O postproceso": spec.postprocess_big_o,
            "nota": spec.complexity_note,
        }
        for spec in MODEL_CATALOG.values()
    ]

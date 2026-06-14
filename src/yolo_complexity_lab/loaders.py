from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .catalog import MODEL_CATALOG, ModelSpec


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    device: str
    parameter_count: int | None
    model_size_mb: float | None
    size_note: str
    class_names: Any | None = None


def _torch_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _count_parameters(model: Any) -> int | None:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _estimate_size_from_params(parameter_count: int | None) -> tuple[float | None, str]:
    if parameter_count is None:
        return None, "No se pudo estimar tamaño."
    return round(parameter_count * 4 / (1024**2), 3), "Estimado como parámetros × 4 bytes (FP32)."


def _file_size_mb(path: str | Path | None) -> float | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if p.exists() and p.is_file():
        return round(p.stat().st_size / (1024**2), 3)
    return None


def load_model(spec_key: str, requested_device: str = "auto") -> LoadedModel:
    """Load a supported detector. Downloads weights if the backend does so.

    The function intentionally does not train anything; it only loads pretrained
    weights for inference and benchmarking.
    """
    if spec_key not in MODEL_CATALOG:
        raise KeyError(f"Modelo no soportado: {spec_key}")

    import torch

    spec = MODEL_CATALOG[spec_key]
    device = _torch_device(requested_device)

    if spec.backend == "ultralytics":
        from ultralytics import YOLO

        # Keep downloaded YOLO weights outside the repository. Ultralytics
        # downloads known assets into the current working directory when given
        # a short name like yolo11n.pt, so temporarily switch to a cache dir.
        weights_dir = Path.home() / ".cache" / "yolo-complexity-lab" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        expected_weight_path = weights_dir / spec.weight_name
        previous_cwd = Path.cwd()
        try:
            os.chdir(weights_dir)
            yolo = YOLO(str(expected_weight_path if expected_weight_path.exists() else spec.weight_name))
        finally:
            os.chdir(previous_cwd)

        try:
            yolo.to(device)
        except Exception:
            # Ultralytics also accepts device during predict; keep model usable.
            pass
        inner = getattr(yolo, "model", None)
        params = _count_parameters(inner) if inner is not None else None
        ckpt_path = getattr(yolo, "ckpt_path", None) or expected_weight_path
        size = _file_size_mb(ckpt_path)
        size_note = f"Tamaño del archivo de pesos: {ckpt_path}" if size else "Tamaño estimado desde parámetros."
        if size is None:
            size, fallback_note = _estimate_size_from_params(params)
            size_note = fallback_note
        return LoadedModel(
            spec=spec,
            model=yolo,
            device=device,
            parameter_count=params,
            model_size_mb=size,
            size_note=size_note,
            class_names=getattr(yolo, "names", None),
        )

    if spec.backend == "torchvision":
        import torchvision

        if spec_key == "ssdlite_mobilenet_v3":
            weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        elif spec_key == "fasterrcnn_mobilenet_fpn":
            weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        else:
            raise KeyError(f"Modelo torchvision no soportado: {spec_key}")
        model.eval().to(torch.device(device))
        params = _count_parameters(model)
        size, size_note = _estimate_size_from_params(params)
        return LoadedModel(
            spec=spec,
            model=model,
            device=device,
            parameter_count=params,
            model_size_mb=size,
            size_note=size_note,
            class_names=weights.meta.get("categories"),
        )

    raise ValueError(f"Backend no soportado: {spec.backend}")

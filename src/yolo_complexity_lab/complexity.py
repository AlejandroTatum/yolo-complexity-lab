from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ComplexityEstimate:
    macs: int | None
    gmacs: float | None
    gflops_approx: float | None
    conv_layers: int
    linear_layers: int
    note: str


def _device_from_model(model: Any, fallback: str = "cpu") -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return fallback


def estimate_conv_linear_macs(
    model: Any,
    input_size: int,
    runner: Callable[[Any, Any], Any] | None = None,
    device: str | None = None,
) -> ComplexityEstimate:
    """Approximate MACs by counting Conv2d and Linear layers via forward hooks.

    This is intentionally local and transparent. It does not count every operation
    used by modern detectors (activation, normalization, attention, NMS, ROI ops),
    but it gives a useful comparative proxy for computational complexity.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        return ComplexityEstimate(None, None, None, 0, 0, f"PyTorch no disponible: {exc}")

    if model is None:
        return ComplexityEstimate(None, None, None, 0, 0, "Modelo no disponible.")

    model_device = device or _device_from_model(model)
    macs = 0
    conv_layers = 0
    linear_layers = 0
    hooks = []

    def conv_hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal macs, conv_layers
        try:
            if isinstance(output, (tuple, list)):
                output = output[0]
            batch_size = int(output.shape[0])
            out_channels = int(output.shape[1])
            out_h = int(output.shape[2])
            out_w = int(output.shape[3])
            kernel_h, kernel_w = module.kernel_size
            in_channels = int(module.in_channels)
            groups = int(module.groups)
            macs += batch_size * out_h * out_w * out_channels * (in_channels // groups) * kernel_h * kernel_w
            conv_layers += 1
        except Exception:
            pass

    def linear_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal macs, linear_layers
        try:
            batch_size = int(inputs[0].shape[0]) if hasattr(inputs[0], "shape") and len(inputs[0].shape) > 1 else 1
            macs += batch_size * int(module.in_features) * int(module.out_features)
            linear_layers += 1
        except Exception:
            pass

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
        dummy = torch.zeros(1, 3, input_size, input_size, device=torch.device(model_device))
        with torch.inference_mode():
            if runner is not None:
                runner(model, dummy)
            else:
                model(dummy)
    except Exception as exc:
        return ComplexityEstimate(
            None,
            None,
            None,
            conv_layers,
            linear_layers,
            f"No se pudo ejecutar forward de conteo MACs: {exc}",
        )
    finally:
        for hook in hooks:
            hook.remove()
        try:
            model.train(was_training)
        except Exception:
            pass

    gmacs = round(macs / 1e9, 4)
    # Common convention: 1 MAC ≈ 2 FLOPs (multiply + add).
    gflops = round((2 * macs) / 1e9, 4)
    return ComplexityEstimate(
        macs=int(macs),
        gmacs=gmacs,
        gflops_approx=gflops,
        conv_layers=conv_layers,
        linear_layers=linear_layers,
        note="Conteo aproximado por hooks Conv2d/Linear; no incluye NMS/ROI/activaciones con precisión completa.",
    )


def estimate_for_loaded_model(loaded: Any, input_size: int) -> ComplexityEstimate:
    """Route complexity estimation according to backend."""
    spec = loaded.spec
    if spec.backend == "ultralytics":
        inner = getattr(loaded.model, "model", None)
        return estimate_conv_linear_macs(inner, input_size=input_size, device=loaded.device)

    if spec.backend == "torchvision":
        def torchvision_runner(model: Any, dummy: Any) -> Any:
            # TorchVision detection models expect list[Tensor] with CxHxW.
            return model([dummy[0]])

        return estimate_conv_linear_macs(loaded.model, input_size=input_size, runner=torchvision_runner, device=loaded.device)

    return ComplexityEstimate(None, None, None, 0, 0, "Backend sin estimador.")

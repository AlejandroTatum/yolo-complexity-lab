"""System information probing.

State reporting is always coherent: ``cuda_available`` is ``True`` only when a
real CUDA device was successfully probed end-to-end (import, ``is_available``
and ``get_device_name``). Any probe failure fails closed to a safe value
(no CUDA / ``None``) and is surfaced both in the result (``probe_errors``) and
in the logs, so a benchmark comparison is never silently invalidated.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass

import psutil

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemInfo:
    python: str
    platform: str
    processor: str
    cpu_logical_cores: int | None
    cpu_physical_cores: int | None
    ram_total_gb: float | None
    torch_available: bool
    cuda_available: bool
    cuda_device: str
    probe_errors: tuple[str, ...]


@dataclass(frozen=True)
class _TorchProbe:
    torch_available: bool
    cuda_available: bool
    cuda_device: str
    errors: tuple[str, ...]


def _probe_failure(torch_available: bool, label: str, exc: Exception) -> _TorchProbe:
    message = f"{label} failed: {exc!r}"
    _logger.warning("system probe failed: %s", message, exc_info=True)
    return _TorchProbe(torch_available, False, "CPU", (message,))


def _probe_torch_cuda() -> _TorchProbe:
    """Import torch and probe CUDA. Fails closed: any probe failure means no CUDA."""
    try:
        import torch
    except Exception as exc:
        return _probe_failure(False, "import torch", exc)

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return _probe_failure(True, "torch.cuda.is_available()", exc)

    if not cuda_available:
        return _TorchProbe(True, False, "CPU", ())

    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        return _probe_failure(True, "torch.cuda.get_device_name(0)", exc)

    return _TorchProbe(True, True, device_name, ())


def _probe_cpu(logical: bool) -> tuple[int | None, tuple[str, ...]]:
    """Return (core count, errors). ``None`` means undetermined, never a fake 0."""
    try:
        count = psutil.cpu_count(logical=logical)
    except Exception as exc:
        message = f"psutil.cpu_count(logical={logical}) failed: {exc!r}"
        _logger.warning("system probe failed: %s", message, exc_info=True)
        return None, (message,)
    return count, ()


def _probe_memory() -> tuple[float | None, tuple[str, ...]]:
    """Return (total RAM in GiB, errors). ``None`` means the probe failed."""
    try:
        total = psutil.virtual_memory().total
    except Exception as exc:
        message = f"psutil.virtual_memory() failed: {exc!r}"
        _logger.warning("system probe failed: %s", message, exc_info=True)
        return None, (message,)
    return round(total / (1024**3), 2), ()


def collect_system_info() -> SystemInfo:
    torch_probe = _probe_torch_cuda()
    logical_cores, logical_errors = _probe_cpu(logical=True)
    physical_cores, physical_errors = _probe_cpu(logical=False)
    ram_gb, memory_errors = _probe_memory()

    return SystemInfo(
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        processor=platform.processor() or "unknown",
        cpu_logical_cores=logical_cores,
        cpu_physical_cores=physical_cores,
        ram_total_gb=ram_gb,
        torch_available=torch_probe.torch_available,
        cuda_available=torch_probe.cuda_available,
        cuda_device=torch_probe.cuda_device,
        probe_errors=(
            torch_probe.errors
            + logical_errors
            + physical_errors
            + memory_errors
        ),
    )


def system_info_dict() -> dict[str, object]:
    return asdict(collect_system_info())

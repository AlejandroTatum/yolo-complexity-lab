from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import psutil


@dataclass(frozen=True)
class SystemInfo:
    python: str
    platform: str
    processor: str
    cpu_logical_cores: int
    cpu_physical_cores: int | None
    ram_total_gb: float
    torch_available: bool
    cuda_available: bool
    cuda_device: str


def collect_system_info() -> SystemInfo:
    torch_available = False
    cuda_available = False
    cuda_device = "CPU"
    try:
        import torch

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return SystemInfo(
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        processor=platform.processor() or "unknown",
        cpu_logical_cores=psutil.cpu_count(logical=True) or 0,
        cpu_physical_cores=psutil.cpu_count(logical=False),
        ram_total_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        torch_available=torch_available,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
    )


def system_info_dict() -> dict[str, object]:
    return asdict(collect_system_info())

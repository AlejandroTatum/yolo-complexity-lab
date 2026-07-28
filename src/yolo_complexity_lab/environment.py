from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


@dataclass(frozen=True)
class Capabilities:
    """Runtime capabilities exposed to the application entry point."""

    is_cloud: bool
    webcam: bool
    streaming: bool
    custom_weights: bool
    device_options: tuple[str, ...]
    device_default: str


_LOCALHOST_NAMES = {"localhost", "127.0.0.1", "::1"}
_TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "on"}


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUTHY_VALUES


def _is_non_localhost_url(value: str) -> bool:
    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"//{candidate}"
    hostname = (urlparse(candidate).hostname or "").lower()
    return bool(hostname) and hostname not in _LOCALHOST_NAMES


def _environment_kind(environ: Mapping[str, str]) -> str:
    override = environ.get("YOLOLAB_ENV", "").strip().lower()
    if override in {"local", "cloud"}:
        return override

    if _is_non_localhost_url(environ.get("STREAMLIT_SERVER_URL", "")):
        return "cloud"
    if _is_truthy(environ.get("STREAMLIT_SHARING_MODE")):
        return "cloud"
    return "restricted"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_capabilities(root: Path, environ: Mapping[str, str] = os.environ) -> Capabilities:
    """Return fail-safe capabilities for a local, cloud, or unknown runtime.

    An environment is local only when explicitly configured as such. This keeps
    public deployments from rendering controls that cannot work there.
    """
    environment = _environment_kind(environ)
    is_local = environment == "local"
    is_cloud = not is_local
    cuda = is_local and _cuda_available()
    device_options = ("auto", "cpu", "cuda:0") if cuda else ("auto", "cpu") if is_local else ("cpu",)

    return Capabilities(
        is_cloud=is_cloud,
        webcam=is_local,
        streaming=is_local,
        custom_weights=(root / "best.pt").is_file(),
        device_options=device_options,
        device_default="auto" if is_local and cuda else "cpu",
    )

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_complexity_lab import environment
from yolo_complexity_lab.environment import detect_capabilities


def test_unknown_environment_is_restricted_and_preserves_root_weights(tmp_path: Path) -> None:
    (tmp_path / "best.pt").touch()

    capabilities = detect_capabilities(tmp_path, {})

    assert capabilities.is_cloud
    assert not capabilities.webcam
    assert not capabilities.streaming
    assert capabilities.custom_weights
    assert capabilities.device_options == ("cpu",)
    assert capabilities.device_default == "cpu"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("STREAMLIT_SERVER_URL", "https://portfolio.streamlit.app"),
        ("STREAMLIT_SHARING_MODE", "true"),
    ],
)
def test_cloud_signals_restrict_local_only_capabilities(tmp_path: Path, key: str, value: str) -> None:
    capabilities = detect_capabilities(tmp_path, {key: value})

    assert capabilities.is_cloud
    assert not capabilities.webcam
    assert capabilities.device_options == ("cpu",)


def test_local_override_unlocks_capabilities_and_cuda_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(environment, "_cuda_available", lambda: True)

    capabilities = detect_capabilities(
        tmp_path,
        {
            "YOLOLAB_ENV": "local",
            "STREAMLIT_SERVER_URL": "https://portfolio.streamlit.app",
        },
    )

    assert not capabilities.is_cloud
    assert capabilities.webcam
    assert capabilities.streaming
    assert capabilities.device_options == ("auto", "cpu", "cuda:0")
    assert capabilities.device_default == "auto"


def test_cloud_override_wins_over_local_signals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(environment, "_cuda_available", lambda: True)

    capabilities = detect_capabilities(
        tmp_path,
        {
            "YOLOLAB_ENV": "cloud",
            "STREAMLIT_SERVER_URL": "localhost:8501",
        },
    )

    assert capabilities.is_cloud
    assert not capabilities.webcam
    assert capabilities.device_options == ("cpu",)


def test_local_environment_without_cuda_uses_cpu_and_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(environment, "_cuda_available", lambda: False)

    capabilities = detect_capabilities(tmp_path, {"YOLOLAB_ENV": "local"})

    assert capabilities.device_options == ("auto", "cpu")
    assert capabilities.device_default == "cpu"

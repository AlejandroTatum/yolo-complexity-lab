"""Tests for yolo_complexity_lab.system_info.

Covers the failure paths the module used to swallow silently: missing torch,
CUDA probe exceptions, undetermined CPU counts and psutil errors. No real
hardware (or torch install) is required: torch is simulated with a fake module
and psutil with monkeypatches.
"""

from __future__ import annotations

import builtins
import logging
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from yolo_complexity_lab import system_info
from yolo_complexity_lab.system_info import (
    SystemInfo,
    _TorchProbe,
    collect_system_info,
    system_info_dict,
)


def fake_torch_module(
    *,
    is_available: bool = True,
    device_name: str = "NVIDIA Test GPU 0",
    fail_is_available: bool = False,
    fail_device_name: bool = False,
) -> ModuleType:
    mod = ModuleType("torch")
    cuda = MagicMock()
    cuda.is_available.return_value = is_available
    if fail_is_available:
        cuda.is_available.side_effect = RuntimeError("cuda driver error")
    cuda.get_device_name.return_value = device_name
    if fail_device_name:
        cuda.get_device_name.side_effect = RuntimeError("cuda query error")
    mod.cuda = cuda
    return mod


@pytest.fixture
def install_torch(monkeypatch: pytest.MonkeyPatch):
    def _install(mod: ModuleType) -> None:
        monkeypatch.setitem(sys.modules, "torch", mod)

    return _install


def assert_coherent(info: SystemInfo) -> None:
    """The reported device state must never contradict itself."""
    assert info.cuda_available is False or info.torch_available
    assert (info.cuda_available is True) == (info.cuda_device != "CPU")
    if not info.cuda_available:
        assert info.cuda_device == "CPU"


def assert_coherent_probe(probe: _TorchProbe) -> None:
    assert probe.cuda_available is False or probe.torch_available
    assert (probe.cuda_available is True) == (probe.cuda_device != "CPU")
    if not probe.cuda_available:
        assert probe.cuda_device == "CPU"


# --- _probe_torch_cuda ------------------------------------------------------


def test_probe_torch_cuda_happy_path(install_torch) -> None:
    install_torch(fake_torch_module(device_name="NVIDIA GeForce RTX 4090"))

    probe = system_info._probe_torch_cuda()

    assert probe == _TorchProbe(True, True, "NVIDIA GeForce RTX 4090", ())


def test_probe_torch_cuda_no_cuda_available(install_torch) -> None:
    install_torch(fake_torch_module(is_available=False))

    probe = system_info._probe_torch_cuda()

    assert probe == _TorchProbe(True, False, "CPU", ())
    assert_coherent_probe(probe)


def test_probe_torch_cuda_missing_torch(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    real_import = builtins.__import__

    def raise_for_torch(name: str, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raise_for_torch)

    with caplog.at_level(logging.WARNING):
        probe = system_info._probe_torch_cuda()

    assert probe.torch_available is False
    assert probe.cuda_available is False
    assert probe.cuda_device == "CPU"
    assert len(probe.errors) == 1
    assert "import torch failed" in probe.errors[0]
    assert any("system probe failed" in r.message for r in caplog.records)


def test_probe_torch_cuda_device_name_failure_fails_closed(install_torch, caplog: pytest.LogCaptureFixture) -> None:
    """Regression: get_device_name raising must not leave cuda_available=True with device 'CPU'."""
    install_torch(fake_torch_module(is_available=True, fail_device_name=True))

    with caplog.at_level(logging.WARNING):
        probe = system_info._probe_torch_cuda()

    assert probe.torch_available is True
    assert probe.cuda_available is False
    assert probe.cuda_device == "CPU"
    assert len(probe.errors) == 1
    assert "torch.cuda.get_device_name(0) failed" in probe.errors[0]
    assert any("system probe failed" in r.message for r in caplog.records)
    assert_coherent_probe(probe)


def test_probe_torch_cuda_is_available_failure(install_torch, caplog: pytest.LogCaptureFixture) -> None:
    install_torch(fake_torch_module(fail_is_available=True))

    with caplog.at_level(logging.WARNING):
        probe = system_info._probe_torch_cuda()

    assert probe == _TorchProbe(True, False, "CPU", ("torch.cuda.is_available() failed: RuntimeError('cuda driver error')",))
    assert any("system probe failed" in r.message for r in caplog.records)


# --- _probe_cpu / _probe_memory --------------------------------------------


def test_probe_cpu_none_is_reported_as_none_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_info.psutil, "cpu_count", lambda logical: None)

    logical, errors = system_info._probe_cpu(logical=True)

    assert logical is None
    assert errors == ()


def test_probe_cpu_returns_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_info.psutil, "cpu_count", lambda logical: 8)

    assert system_info._probe_cpu(logical=True) == (8, ())
    assert system_info._probe_cpu(logical=False) == (8, ())


def test_probe_cpu_exception_is_surfaced(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(logical):
        raise OSError("permission denied")

    monkeypatch.setattr(system_info.psutil, "cpu_count", boom)

    with caplog.at_level(logging.WARNING):
        count, errors = system_info._probe_cpu(logical=True)

    assert count is None
    assert len(errors) == 1
    assert "psutil.cpu_count(logical=True) failed" in errors[0]
    assert any("system probe failed" in r.message for r in caplog.records)


def test_probe_memory_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = MagicMock(total=16 * 1024**3)
    monkeypatch.setattr(system_info.psutil, "virtual_memory", lambda: memory)

    assert system_info._probe_memory() == (16.0, ())


def test_probe_memory_exception_is_surfaced(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom():
        raise OSError("vm stat failed")

    monkeypatch.setattr(system_info.psutil, "virtual_memory", boom)

    with caplog.at_level(logging.WARNING):
        total, errors = system_info._probe_memory()

    assert total is None
    assert len(errors) == 1
    assert "psutil.virtual_memory() failed" in errors[0]
    assert any("system probe failed" in r.message for r in caplog.records)


# --- collect_system_info ----------------------------------------------------


def test_collect_system_info_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_info,
        "_probe_torch_cuda",
        lambda: _TorchProbe(True, True, "NVIDIA GeForce RTX 4090", ()),
    )
    monkeypatch.setattr(system_info, "_probe_cpu", lambda logical: (8, ()))
    monkeypatch.setattr(system_info, "_probe_memory", lambda: (15.9, ()))

    info = collect_system_info()

    assert info.python
    assert info.platform
    assert info.processor
    assert info.cpu_logical_cores == 8
    assert info.cpu_physical_cores == 8
    assert info.ram_total_gb == 15.9
    assert info.torch_available is True
    assert info.cuda_available is True
    assert info.cuda_device == "NVIDIA GeForce RTX 4090"
    assert info.probe_errors == ()
    assert_coherent(info)


def test_collect_system_info_no_cuda_is_coherent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_info,
        "_probe_torch_cuda",
        lambda: _TorchProbe(True, False, "CPU", ()),
    )

    info = collect_system_info()

    assert info.torch_available is True
    assert info.cuda_available is False
    assert info.cuda_device == "CPU"
    assert_coherent(info)


def test_collect_system_info_undetermined_cpu_is_none_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_info,
        "_probe_torch_cuda",
        lambda: _TorchProbe(True, False, "CPU", ()),
    )
    monkeypatch.setattr(system_info, "_probe_cpu", lambda logical: (None, ()))

    info = collect_system_info()

    assert info.cpu_logical_cores is None
    assert info.cpu_physical_cores is None
    assert_coherent(info)


def test_collect_system_info_aggregates_all_probe_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_info,
        "_probe_torch_cuda",
        lambda: _TorchProbe(True, False, "CPU", ("cuda probe failed",)),
    )
    monkeypatch.setattr(system_info, "_probe_cpu", lambda logical: (None, ("cpu probe failed",)))
    monkeypatch.setattr(system_info, "_probe_memory", lambda: (None, ("memory probe failed",)))

    info = collect_system_info()

    assert info.cpu_logical_cores is None
    assert info.ram_total_gb is None
    assert info.cuda_available is False
    assert info.cuda_device == "CPU"
    assert info.probe_errors == ("cuda probe failed", "cpu probe failed", "cpu probe failed", "memory probe failed")
    assert_coherent(info)


def test_system_info_dict_contains_all_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_info,
        "_probe_torch_cuda",
        lambda: _TorchProbe(True, False, "CPU", ("boom",)),
    )

    data = system_info_dict()

    assert set(data) == {
        "python",
        "platform",
        "processor",
        "cpu_logical_cores",
        "cpu_physical_cores",
        "ram_total_gb",
        "torch_available",
        "cuda_available",
        "cuda_device",
        "probe_errors",
    }
    assert data["cuda_available"] is False
    assert data["cuda_device"] == "CPU"
    assert data["probe_errors"] == ("boom",)

from __future__ import annotations

from io import StringIO
from pathlib import Path
import socket
import sys
import threading

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_check


class FakeProcess:
    def __init__(self, logs: str = "", return_code: int | None = None) -> None:
        self.stdout = StringIO(logs)
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = 0

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.return_code or 0


def test_static_smoke_success() -> None:
    result = smoke_check.check_static(ROOT)

    assert result["capabilities"].device_options == ("cpu",)  # type: ignore[union-attr]


def test_static_smoke_reports_missing_resources(tmp_path: Path) -> None:
    with pytest.raises(smoke_check.SmokeCheckError, match="Missing required entry point"):
        smoke_check.check_static(tmp_path)


def test_server_success_probes_health_and_page(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess()
    probes: list[str] = []

    class Probe:
        def __init__(self, port: int) -> None:
            assert port == 8901

        def get(self, path: str) -> int:
            probes.append(path)
            return 200

    monkeypatch.setattr(smoke_check, "_start_server", lambda _root, _port: process)
    monkeypatch.setattr(smoke_check, "_LoopbackProbe", Probe)

    smoke_check.check_server(ROOT)

    assert probes == ["/healthz", "/"]
    assert process.terminated


def test_server_rejects_error_published_during_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess()
    reader_started = threading.Event()
    publish_logs = threading.Event()
    published = threading.Event()

    def delayed_reader(_process: FakeProcess, logs: list[str]) -> None:
        reader_started.set()
        publish_logs.wait(timeout=1)
        logs.append("Traceback (most recent call last):")
        published.set()

    def terminate(_process: FakeProcess) -> None:
        process.terminate()
        publish_logs.set()

    def probe(_probe: smoke_check._LoopbackProbe, _path: str) -> int:
        assert reader_started.wait(timeout=1)
        return 200

    monkeypatch.setattr(smoke_check, "_start_server", lambda _root, _port: process)
    monkeypatch.setattr(smoke_check, "_read_output", delayed_reader)
    monkeypatch.setattr(smoke_check, "_terminate_process", terminate)
    monkeypatch.setattr(smoke_check._LoopbackProbe, "get", probe)

    with pytest.raises(smoke_check.SmokeCheckError, match="runtime errors"):
        smoke_check.check_server(ROOT)

    assert published.is_set()
    assert process.terminated


def test_server_rejects_occupied_port_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    spawned = False

    def fail_if_spawned(_root: Path, _port: int) -> FakeProcess:
        nonlocal spawned
        spawned = True
        raise AssertionError("the Streamlit child must not spawn")

    monkeypatch.setattr(smoke_check, "_start_server", fail_if_spawned)
    try:
        with pytest.raises(smoke_check.SmokeCheckError, match=f"Port {port} .* already in use"):
            smoke_check.check_server(ROOT, port=port)
    finally:
        listener.close()

    assert not spawned


@pytest.mark.parametrize(
    ("statuses", "message"),
    [([503], "Health probe failed"), ([200, 503], "Primary page probe failed")],
)
def test_server_probe_failures_are_actionable_and_clean_up(
    monkeypatch: pytest.MonkeyPatch, statuses: list[int], message: str
) -> None:
    process = FakeProcess()
    responses = iter(statuses)
    monkeypatch.setattr(smoke_check, "_start_server", lambda _root, _port: process)
    monkeypatch.setattr(smoke_check._LoopbackProbe, "get", lambda _probe, _path: next(responses))

    with pytest.raises(smoke_check.SmokeCheckError, match=message):
        smoke_check.check_server(ROOT)

    assert process.terminated


def test_probe_treats_direct_timeout_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(_host: str, _port: int, timeout: float):
        class Connection:
            def request(self, _method: str, _path: str) -> None:
                raise TimeoutError("timed out")

            def close(self) -> None:
                pass

        return Connection()

    monkeypatch.setattr(smoke_check, "HTTPConnection", timeout)

    assert smoke_check._LoopbackProbe(8901).get("/healthz") is None


@pytest.mark.parametrize("port", [0, 65536, -1, True, "8901"])
def test_internal_ports_are_rejected(port: object) -> None:
    for function in (smoke_check._validate_port, smoke_check._require_port_available):
        with pytest.raises(smoke_check.SmokeCheckError, match="between 1 and 65535"):
            function(port)  # type: ignore[arg-type]
    with pytest.raises(smoke_check.SmokeCheckError, match="between 1 and 65535"):
        smoke_check._start_server(ROOT, port)  # type: ignore[arg-type]
    with pytest.raises(smoke_check.SmokeCheckError, match="between 1 and 65535"):
        smoke_check._LoopbackProbe(port)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["0", "65536", "not-a-port"])
def test_cli_rejects_invalid_ports(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        smoke_check.main(["--port", value])

    assert exc_info.value.code == 2
    assert "port must be an integer between 1 and 65535" in capsys.readouterr().err


def test_probe_rejects_arbitrary_paths() -> None:
    with pytest.raises(smoke_check.SmokeCheckError, match="Unsupported loopback probe path"):
        smoke_check._LoopbackProbe(8901).get("/not-allowed")


def test_probe_connection_timeout_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(_host: str, _port: int, timeout: float):
        raise TimeoutError("timed out")

    monkeypatch.setattr(smoke_check, "HTTPConnection", timeout)

    assert smoke_check._LoopbackProbe(8901).get("/healthz") is None


@pytest.mark.parametrize(
    ("logs", "return_code", "timeout", "message"),
    [
        ("", 1, 90, "exited before becoming ready"),
        ("", None, 0, "did not become ready within 0 seconds"),
        ("Traceback (most recent call last):", None, 90, "startup errors"),
        ("ERROR app failed to load", None, 90, "startup errors"),
    ],
)
def test_server_failure_is_actionable_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    logs: str,
    return_code: int | None,
    timeout: float,
    message: str,
) -> None:
    process = FakeProcess(logs, return_code)
    monkeypatch.setattr(smoke_check, "_start_server", lambda _root, _port: process)
    monkeypatch.setattr(smoke_check, "STARTUP_TIMEOUT_SECONDS", timeout)
    monkeypatch.setattr(smoke_check._LoopbackProbe, "get", lambda _probe, _path: None)

    with pytest.raises(smoke_check.SmokeCheckError, match=message):
        smoke_check.check_server(ROOT)

    assert process.terminated or process.return_code is not None

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from yolo_complexity_lab.catalog import MODEL_CATALOG, catalog_rows
from yolo_complexity_lab.environment import detect_capabilities
from yolo_complexity_lab.paths import default_export_dir, find_workspace_root
from yolo_complexity_lab.system_info import system_info_dict

STARTUP_TIMEOUT_SECONDS = 90.0
POLL_INTERVAL_SECONDS = 0.25
_LOG_ERROR = re.compile(r"\b(?:traceback|exception|error)\b", re.IGNORECASE)


class SmokeCheckError(RuntimeError):
    """An actionable deployment smoke-check failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeCheckError(message)


def check_static(root: Path = ROOT) -> dict[str, object]:
    """Validate files and contracts that do not require starting Streamlit."""
    _require((root / "app.py").is_file(), f"Missing required entry point: {root / 'app.py'}")
    for relative in (".streamlit/config.toml", "assets/demo_person_dog_fruit.jpg"):
        _require((root / relative).is_file(), f"Missing required resource: {root / relative}")

    try:
        config = tomllib.loads((root / ".streamlit/config.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SmokeCheckError(f"Cannot parse Streamlit configuration: {exc}") from exc
    _require(config.get("server", {}).get("headless") is True, "Streamlit config must enable headless mode.")

    _require(len(MODEL_CATALOG) == 5, "Model catalog must contain five planned models.")
    required = {"yolov8_gestures", "yolo11n", "yolo11s", "ssdlite_mobilenet_v3", "fasterrcnn_mobilenet_fpn"}
    _require(required.issubset(MODEL_CATALOG), "Model catalog is incomplete.")
    for row in catalog_rows():
        _require("O(" in row["Big-O inferencia"], f"Missing inference complexity in catalog row: {row}")
        _require("O(" in row["Big-O postproceso"], f"Missing postprocess complexity in catalog row: {row}")

    capabilities = detect_capabilities(root, {"YOLOLAB_ENV": "cloud"})
    _require(capabilities.is_cloud, "Cloud capability detection did not enter restricted mode.")
    _require(not capabilities.webcam and not capabilities.streaming, "Cloud mode exposed local-only capabilities.")
    _require(capabilities.device_options == ("cpu",), "Cloud mode must expose CPU as its only device option.")
    local = detect_capabilities(root, {"YOLOLAB_ENV": "local"})
    _require(local.webcam and local.streaming, "Local capability detection did not enable camera features.")
    _require("auto" in local.device_options and "cpu" in local.device_options, "Local device options are incomplete.")
    _require("python" in system_info_dict(), "Runtime system information is unavailable.")
    workspace = find_workspace_root(root)
    export_dir = default_export_dir()
    _require(workspace.exists(), f"Workspace root does not exist: {workspace}")
    _require("PROYECTO001_YOLO_COMPLEXITY_LAB_Alejandro_Padilla" in str(export_dir), "Export path contract changed.")
    return {"workspace": workspace, "export_dir": export_dir, "capabilities": capabilities}


def _start_server(root: Path, port: int) -> subprocess.Popen[str]:
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(root / "app.py"),
                "--server.headless",
                "true",
                "--server.port",
                str(port),
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise SmokeCheckError(f"Could not start Streamlit child process: {exc}") from exc


def _require_port_available(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            raise SmokeCheckError(f"Port {port} on {host} is already in use; choose an available port.")
    except SmokeCheckError:
        raise
    except OSError:
        return


def _read_output(process: subprocess.Popen[str], logs: list[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        logs.append(line.rstrip())


def _probe(url: str) -> int | None:
    try:
        with urlopen(url, timeout=2) as response:
            return response.status
    except HTTPError as exc:
        return exc.code
    except TimeoutError:
        return None
    except URLError:
        return None


def _log_errors(logs: list[str]) -> list[str]:
    return [line for line in logs if _LOG_ERROR.search(line)]


def _wait_for_server(process: subprocess.Popen[str], base_url: str, timeout: float, logs: list[str]) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        failures = _log_errors(logs)
        if failures:
            raise SmokeCheckError("Streamlit emitted startup errors: " + " | ".join(failures[-3:]))
        return_code = process.poll()
        if return_code is not None:
            raise SmokeCheckError(f"Streamlit exited before becoming ready (code {return_code}). Check its logs.")
        health = _probe(f"{base_url}/healthz")
        if health == 200:
            page = _probe(f"{base_url}/")
            if page == 200:
                failures = _log_errors(logs)
                if failures:
                    raise SmokeCheckError("Streamlit emitted runtime errors: " + " | ".join(failures[-3:]))
                return
            if page is not None:
                raise SmokeCheckError(f"Primary page probe failed with HTTP {page}; expected HTTP 200.")
        elif health is not None:
            raise SmokeCheckError(f"Health probe failed with HTTP {health}; expected HTTP 200.")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise SmokeCheckError(f"Streamlit did not become ready within {timeout:g} seconds.")


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def check_server(root: Path = ROOT, port: int = 8901) -> None:
    """Start, probe, and always terminate an isolated Streamlit child."""
    host = "127.0.0.1"
    _require_port_available(host, port)
    process = _start_server(root, port)
    logs: list[str] = []
    reader = threading.Thread(target=_read_output, args=(process, logs), daemon=True)
    reader.start()
    succeeded = False
    try:
        reader.join(timeout=0.01)
        _wait_for_server(process, f"http://{host}:{port}", STARTUP_TIMEOUT_SECONDS, logs)
        succeeded = True
    finally:
        _terminate_process(process)
        reader.join(timeout=1)
        if succeeded:
            if reader.is_alive():
                raise SmokeCheckError("Streamlit output reader did not finish after process cleanup.")
            failures = _log_errors(logs)
            if failures:
                raise SmokeCheckError("Streamlit emitted runtime errors: " + " | ".join(failures[-3:]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate static deployment requirements and Streamlit reachability.")
    parser.add_argument("--server", action="store_true", help="Start Streamlit and probe /healthz and /.")
    parser.add_argument("--port", type=int, default=8901, help="Port used by --server (default: 8901).")
    args = parser.parse_args(argv)
    try:
        result = check_static()
        if args.server:
            check_server(port=args.port)
        print("Smoke check OK")
        print(f"Workspace root: {result['workspace']}")
        print(f"Export dir: {result['export_dir']}")
        return 0
    except SmokeCheckError as exc:
        print(f"Smoke check FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

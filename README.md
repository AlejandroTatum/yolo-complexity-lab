# YOLO Complexity Lab

[![CI](https://github.com/AlejandroTatum/yolo-complexity-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/AlejandroTatum/yolo-complexity-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/AlejandroTatum/yolo-complexity-lab)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Built with Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)](https://streamlit.io)

**English** | [Español](README.es.md)

A local-first Streamlit lab that runs **YOLO in real time** and compares **runtime** and **computational complexity** against classic CNN detectors. Built as a portfolio experiment for a university computational-complexity course.

🔗 **Live demo:** [yolo-complexity-lab-unl.streamlit.app](https://yolo-complexity-lab-unl.streamlit.app/)

![YOLO Complexity Lab overview](docs/screenshots/overview.png)

## Why

Object detection has to locate and classify many objects per image. The YOLO-versus-CNN question is whether a **single pass** over the image reduces runtime cost. This lab turns that theory into a measurable experiment: same input, same settings, different architectures.

The lab does not try to demonstrate mAP. Recognition is a visual aid; the evidence is time and cost.

## What it measures

| Metric | Meaning |
| --- | --- |
| Mean / p95 latency | Time per frame — lower is better |
| Effective FPS | Real-time throughput — higher is better |
| `n = H × W` | Input size: the axis complexity grows on |
| MACs / GFLOPs | Approximate operations per frame (1 MAC ≈ 2 FLOPs) |
| Parameters | Learned weights: model size, memory, capacity |
| Recognized classes | Visual aid for the chosen input, not a formal accuracy metric |

## How it works

The sidebar offers three comparison routes; unsupported routes are hidden automatically depending on the environment (local vs cloud):

1. **YOLO en vivo** — stream a local webcam through YOLO11n and watch latency, FPS, and detections update per frame.
2. **Comparación CNN vs YOLO** — measure `fasterrcnn_mobilenet_v3_large_320_fpn` (two-stage), `ssdlite320_mobilenet_v3_large` (one-stage), and `yolo11n` on the same input, then read the winners chart and export a CSV.
3. **Pesos personalizados** — run an optional local `best.pt` (e.g. a gesture model) when the file exists at the repository root.

Frame sources: a bundled person/dog/fruit demo image, your own uploaded photo (processed live), or the local webcam. The Benchmark tab overlays each model's real detections on the input — pick a model chip, mount your own photo, and run.

![Benchmark comparison of the three detectors](docs/screenshots/benchmark.png)

## Quickstart

> Use a virtual environment; do not install with the system `pip`.

```bash
git clone https://github.com/AlejandroTatum/yolo-complexity-lab.git
cd yolo-complexity-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only setup, install PyTorch first to avoid pulling CUDA wheels:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run the lab locally (this exposes webcam, streaming, and CUDA choices):

```bash
YOLOLAB_ENV=local streamlit run app.py
```

Model weights download on first use into `~/.cache/yolo-complexity-lab/weights/`; they stay out of the repository.

## The Big-O behind the lab

Main convolutional cost:

```text
O(Σ_l H_l × W_l × C_in_l × C_out_l × K_l²)
```

Didactic form with `n = H × W`:

```text
O(L × n × C_in × C_out × K²)
```

NMS post-processing:

```text
O(B²)
```

Two-stage detectors add a per-region cost:

```text
O(Σ conv + R × C_roi + NMS)
```

## Project structure

```text
.
├── app.py                      # Streamlit UI: layout, tabs, charts, CSS
├── src/yolo_complexity_lab/    # benchmark, catalog, loaders, complexity, system info
├── scripts/smoke_check.py      # static + server smoke checks
├── tests/                      # pytest suite
├── assets/                     # demo image (person/dog/fruit)
├── docs/                       # visual guide + screenshots
└── .streamlit/config.toml      # global theme
```

## Testing and CI

```bash
.venv/bin/python scripts/smoke_check.py            # static contracts
.venv/bin/python scripts/smoke_check.py --server   # boots Streamlit, probes /healthz and /
.venv/bin/pytest -q                                # full test suite
```

GitHub Actions runs the smoke checks and the test suite on Python 3.11–3.13 for every pull request.

## Deployment

The app deploys to Streamlit Community Cloud with `app.py` as the main file. Unknown (unset `YOLOLAB_ENV`) and `YOLOLAB_ENV=cloud` environments use a restricted capability contract: webcam routes are hidden and only the CPU device is offered. See [DEPLOY.md](DEPLOY.md) for the full guide and recovery playbook.

## Roadmap

- [ ] Light/dark theme toggle for different rooms and projectors
- [ ] Small mAP-style quantitative evaluation on a fixed dataset
- [ ] Video-file source in addition to image and webcam
- [ ] Side-by-side multi-model live comparison

## Credits

The bundled demo image combines `zidane.jpg` (Ultralytics sample asset) and a dog-with-banana photo by Karsten Winegeart on Unsplash (`de5wBys0nok`). It is used purely as a didactic slide for comparing recognition, false positives, and timing.

## License

[MIT](LICENSE)

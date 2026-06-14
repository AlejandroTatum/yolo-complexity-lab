#!/usr/bin/env python3
"""
Benchmark exhaustivo sobre COCO val2017.

Mide latencia y detecciones por imagen real para comparar modelos
con datos de validación estándar de la comunidad.

Uso:
    python scripts/benchmark_coco_val.py --sample 100 --device auto

Autogenerado para pruebas exhaustivas del yolo-complexity-lab.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import pandas as pd

from yolo_complexity_lab.benchmark import BenchmarkConfig, benchmark_model
from yolo_complexity_lab.catalog import MODEL_CATALOG
from yolo_complexity_lab.loaders import load_model
from yolo_complexity_lab.sources import read_image_file


def load_image_paths(dataset_dir: Path, sample: int | None = None) -> list[Path]:
    """Cargar rutas de imágenes del dataset, opcionalmente muestreando."""
    image_dir = dataset_dir / "val2017"
    if not image_dir.exists():
        raise FileNotFoundError(f"No existe {image_dir}")
    
    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No se encontraron imágenes .jpg en {image_dir}")
    
    if sample and sample > 0:
        # Muestreo determinista para reproducibilidad
        import random
        rng = random.Random(42)
        paths = rng.sample(paths, min(sample, len(paths)))
        paths = sorted(paths)
    
    return paths


def run_single_image(model, image_path: Path, imgsz: int, conf: float, iou: float, device: str):
    """Ejecutar un modelo sobre una imagen y retornar métricas."""
    frame = read_image_file(image_path)
    
    # Redimensionar a imgsz si es necesario
    h, w = frame.shape[:2]
    if h != imgsz or w != imgsz:
        frame = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    
    start = time.perf_counter()
    
    if model.spec.backend == "ultralytics":
        results = model.model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        detections = len(getattr(results[0], "boxes", []) or [])
    elif model.spec.backend == "torchvision":
        import torch
        import torchvision.transforms as T
        from PIL import Image
        
        pil_img = Image.fromarray(frame)
        transform = T.Compose([T.ToTensor()])
        tensor = transform(pil_img).unsqueeze(0)
        
        # Detectar device real del modelo y mover tensor allí
        model_device = next(model.model.parameters()).device
        tensor = tensor.to(model_device)
        
        with torch.no_grad():
            outputs = model.model(tensor)
        
        if hasattr(outputs, "__getitem__"):
            out = outputs[0]
            # Filtrar por confianza para conteo real de detecciones
            if "scores" in out:
                scores = out["scores"]
                if hasattr(scores, "cpu"):
                    scores = scores.cpu()
                scores_np = scores.numpy() if hasattr(scores, "numpy") else scores
                detections = int((scores_np > conf).sum())
            elif "boxes" in out:
                detections = len(out["boxes"])
            else:
                detections = 0
        else:
            detections = 0
    else:
        detections = 0
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, detections


def benchmark_model_on_dataset(
    model_key: str,
    image_paths: list[Path],
    imgsz: int = 416,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
    warmup: int = 3,
):
    """Benchmark completo de un modelo sobre el dataset."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_key}")
    print(f"Device: {device} | imgsz: {imgsz} | images: {len(image_paths)}")
    print(f"{'='*60}")
    
    loaded = load_model(model_key, device)
    
    # Warmup
    if warmup > 0 and image_paths:
        for _ in range(warmup):
            run_single_image(loaded, image_paths[0], imgsz, conf, iou, device)
    
    timings = []
    detections = []
    
    for i, path in enumerate(image_paths):
        try:
            latency_ms, dets = run_single_image(loaded, path, imgsz, conf, iou, device)
            timings.append(latency_ms)
            detections.append(dets)
            
            if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
                print(f"  Procesadas: {i+1}/{len(image_paths)} | "
                      f"latencia media: {statistics.mean(timings):.2f} ms | "
                      f"detecciones media: {statistics.mean(detections):.1f}")
        except Exception as e:
            print(f"  Error en {path.name}: {e}")
            continue
    
    if not timings:
        return None
    
    return {
        "model": model_key,
        "display_name": MODEL_CATALOG.get(model_key, {}).display_name if model_key in MODEL_CATALOG else model_key,
        "family": MODEL_CATALOG.get(model_key, {}).family if model_key in MODEL_CATALOG else "unknown",
        "images_processed": len(timings),
        "latency_mean_ms": round(statistics.mean(timings), 2),
        "latency_median_ms": round(statistics.median(timings), 2),
        "latency_min_ms": round(min(timings), 2),
        "latency_max_ms": round(max(timings), 2),
        "latency_std_ms": round(statistics.stdev(timings) if len(timings) > 1 else 0, 2),
        "fps_effective": round(1000 / statistics.mean(timings), 2) if statistics.mean(timings) > 0 else 0,
        "detections_mean": round(statistics.mean(detections), 2),
        "detections_median": round(statistics.median(detections), 2),
        "detections_max": max(detections),
        "imgsz": imgsz,
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark exhaustivo sobre COCO val2017")
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/coco_val2017"),
                        help="Directorio con val2017/")
    parser.add_argument("--sample", type=int, default=100,
                        help="Número de imágenes a muestrear (0=todas)")
    parser.add_argument("--models", nargs="+", default=["yolo11n", "fasterrcnn_mobilenet_fpn", "ssdlite_mobilenet_v3"],
                        help="Modelos a comparar")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda:0")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Resolución de entrada")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "coco_val_benchmark.csv")
    args = parser.parse_args()
    
    image_paths = load_image_paths(args.dataset, args.sample if args.sample > 0 else None)
    print(f"Dataset: {args.dataset}")
    print(f"Imágenes a procesar: {len(image_paths)}")
    
    results = []
    for model_key in args.models:
        if model_key not in MODEL_CATALOG:
            print(f"Modelo {model_key} no encontrado en catálogo. Saltando.")
            continue
        
        result = benchmark_model_on_dataset(
            model_key, image_paths,
            imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=args.device, warmup=args.warmup,
        )
        if result:
            results.append(result)
    
    if not results:
        print("No se generaron resultados.")
        return
    
    df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(df[["display_name", "latency_mean_ms", "fps_effective", "detections_mean", "images_processed"]].to_string(index=False))
    print(f"\nCSV guardado en: {args.output}")


if __name__ == "__main__":
    main()

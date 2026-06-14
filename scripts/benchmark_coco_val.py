#!/usr/bin/env python3
"""
Benchmark exhaustivo sobre COCO val2017 con mAP.

Mide latencia, detecciones y precision (mAP) por imagen real para comparar modelos
con datos de validacion estandar de la comunidad.

Uso:
    python scripts/benchmark_coco_val.py --sample 100 --device auto

Autogenerado para pruebas exhaustivas del yolo-complexity-lab.
"""
from __future__ import annotations

import argparse
import json
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo_complexity_lab.catalog import MODEL_CATALOG
from yolo_complexity_lab.loaders import load_model
from yolo_complexity_lab.sources import read_image_file


def load_image_paths(dataset_dir: Path, sample: int | None = None) -> list[Path]:
    """Cargar rutas de imagenes del dataset, opcionalmente muestreando."""
    image_dir = dataset_dir / "val2017"
    if not image_dir.exists():
        raise FileNotFoundError(f"No existe {image_dir}")
    
    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No se encontraron imagenes .jpg en {image_dir}")
    
    if sample and sample > 0:
        import random
        rng = random.Random(42)
        paths = rng.sample(paths, min(sample, len(paths)))
        paths = sorted(paths)
    
    return paths


def get_image_id_from_filename(filename: str) -> int:
    """Extraer image_id de nombre de archivo COCO (000000123456.jpg -> 123456)."""
    stem = Path(filename).stem
    return int(stem)


# Mapping de YOLO class IDs (0-79) a COCO category IDs (1-90)
# Los COCO IDs no son secuenciales, algunos estan omitidos
YOLO_TO_COCO = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90,
}


def run_single_image(model, image_path: Path, imgsz: int, conf: float, iou: float, device: str):
    """Ejecutar un modelo sobre una imagen y retornar predicciones en formato COCO."""
    frame = read_image_file(image_path)
    orig_h, orig_w = frame.shape[:2]
    
    start = time.perf_counter()
    predictions = []
    
    if model.spec.backend == "ultralytics":
        # Ultralytics maneja letterboxing internamente, pasamos imagen original
        results = model.model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)
        
        if boxes is not None and len(boxes) > 0:
            # Ultralytics ya devuelve coordenadas en imagen original
            # Usa indices YOLO 0-79, mapeamos a COCO IDs
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                w = x2 - x1
                h = y2 - y1
                
                # Mapear YOLO class ID a COCO category ID
                coco_id = YOLO_TO_COCO.get(cls_id)
                if coco_id is None:
                    continue
                
                predictions.append({
                    "category_id": coco_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf_score),
                })
                
    elif model.spec.backend == "torchvision":
        import torch
        import torchvision.transforms as T
        from PIL import Image
        
        # Para torchvision necesitamos redimensionar manualmente
        # y luego escalar las coordenadas de vuelta
        if orig_h != imgsz or orig_w != imgsz:
            frame_resized = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame
        
        scale_x = orig_w / imgsz if imgsz > 0 else 1.0
        scale_y = orig_h / imgsz if imgsz > 0 else 1.0
        
        pil_img = Image.fromarray(frame_resized)
        transform = T.Compose([T.ToTensor()])
        tensor = transform(pil_img).unsqueeze(0)
        
        model_device = next(model.model.parameters()).device
        tensor = tensor.to(model_device)
        
        with torch.no_grad():
            outputs = model.model(tensor)
        
        if hasattr(outputs, "__getitem__"):
            out = outputs[0]
            if "boxes" in out and "scores" in out and "labels" in out:
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score < conf:
                        continue
                    x1, y1, x2, y2 = box
                    
                    # Escalar coordenadas a imagen original
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    # torchvision models para COCO usan indices 1-91
                    # Filtramos solo clases COCO validas (1-90)
                    if 1 <= label <= 90:
                        predictions.append({
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score),
                        })
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, predictions


def calculate_map(coco_gt: COCO, predictions: list, image_ids: list) -> dict:
    """Calcular mAP usando pycocotools."""
    if not predictions:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0, "mAP_75": 0.0}
    
    # Filtrar solo imagenes que existen en el ground truth
    valid_image_ids = set(coco_gt.getImgIds())
    filtered_preds = [p for p in predictions if p["image_id"] in valid_image_ids]
    filtered_image_ids = [iid for iid in image_ids if iid in valid_image_ids]
    
    if not filtered_preds or not filtered_image_ids:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0, "mAP_75": 0.0}
    
    # Crear COCO para predicciones
    coco_dt = coco_gt.loadRes(filtered_preds)
    
    # Evaluar
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = filtered_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extraer métricas
    # coco_eval.stats[0] = AP @ IoU=0.50:0.95
    # coco_eval.stats[1] = AP @ IoU=0.50
    # coco_eval.stats[2] = AP @ IoU=0.75
    return {
        "mAP_50_95": round(coco_eval.stats[0] * 100, 2),
        "mAP_50": round(coco_eval.stats[1] * 100, 2),
        "mAP_75": round(coco_eval.stats[2] * 100, 2),
    }


def benchmark_model_on_dataset(
    model_key: str,
    image_paths: list[Path],
    coco_gt: COCO,
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
    all_predictions = []
    image_ids_processed = []
    
    for i, path in enumerate(image_paths):
        try:
            latency_ms, predictions = run_single_image(loaded, path, imgsz, conf, iou, device)
            timings.append(latency_ms)
            
            image_id = get_image_id_from_filename(path.name)
            image_ids_processed.append(image_id)
            
            for pred in predictions:
                pred["image_id"] = image_id
                all_predictions.append(pred)
            
            if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
                print(f"  Procesadas: {i+1}/{len(image_paths)} | "
                      f"latencia media: {statistics.mean(timings):.2f} ms | "
                      f"detecciones: {len(predictions)}")
        except Exception as e:
            print(f"  Error en {path.name}: {e}")
            continue
    
    if not timings:
        return None
    
    # Calcular mAP
    print(f"  Calculando mAP...")
    map_metrics = calculate_map(coco_gt, all_predictions, image_ids_processed)
    
    detections_count = [len([p for p in all_predictions if p["image_id"] == iid]) for iid in image_ids_processed]
    
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
        "detections_mean": round(statistics.mean(detections_count), 2) if detections_count else 0,
        "detections_median": round(statistics.median(detections_count), 2) if detections_count else 0,
        "detections_max": max(detections_count) if detections_count else 0,
        "mAP_50": map_metrics["mAP_50"],
        "mAP_50_95": map_metrics["mAP_50_95"],
        "mAP_75": map_metrics["mAP_75"],
        "imgsz": imgsz,
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark exhaustivo sobre COCO val2017 con mAP")
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/coco_val2017"),
                        help="Directorio con val2017/")
    parser.add_argument("--annotations", type=Path, 
                        default=PROJECT_ROOT / "annotations" / "annotations" / "instances_val2017.json",
                        help="Archivo JSON de anotaciones COCO")
    parser.add_argument("--sample", type=int, default=100,
                        help="Numero de imagenes a muestrear (0=todas)")
    parser.add_argument("--models", nargs="+", default=["yolo11n", "fasterrcnn_mobilenet_fpn", "ssdlite_mobilenet_v3"],
                        help="Modelos a comparar")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda:0")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Resolucion de entrada")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "coco_val_benchmark_with_map.csv")
    args = parser.parse_args()
    
    # Cargar anotaciones COCO
    if not args.annotations.exists():
        print(f"ERROR: No se encontraron anotaciones en {args.annotations}")
        print("Descargalas con: wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        return
    
    print(f"Cargando anotaciones COCO desde: {args.annotations}")
    coco_gt = COCO(str(args.annotations))
    
    image_paths = load_image_paths(args.dataset, args.sample if args.sample > 0 else None)
    print(f"Dataset: {args.dataset}")
    print(f"Imagenes a procesar: {len(image_paths)}")
    
    results = []
    for model_key in args.models:
        if model_key not in MODEL_CATALOG:
            print(f"Modelo {model_key} no encontrado en catalogo. Saltando.")
            continue
        
        result = benchmark_model_on_dataset(
            model_key, image_paths, coco_gt,
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
    print("RESUMEN COMPARATIVO CON PRECISION")
    print(f"{'='*60}")
    print(df[["display_name", "latency_mean_ms", "fps_effective", "mAP_50", "mAP_50_95", "images_processed"]].to_string(index=False))
    print(f"\nCSV guardado en: {args.output}")


if __name__ == "__main__":
    main()

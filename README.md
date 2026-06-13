# YOLO Complexity Lab

App local en Streamlit para comparar **tiempo de ejecución** y **complejidad computacional** de YOLO frente a detectores CNN clásicos.

## Enfoque

El recurso no intenta demostrar mAP. Se centra en:

- Latencia media, mediana, mínimo, máximo y p95.
- FPS efectivo.
- Tiempo por etapa: preprocesamiento, inferencia y postprocesamiento.
- Parámetros y tamaño aproximado del modelo.
- MACs/GFLOPs aproximados por conteo de `Conv2d`/`Linear`.
- Big-O teórico por familia de detector.

## Modelos incluidos

- `yolo11n.pt` — YOLO ligero.
- `yolo11s.pt` — YOLO mediano.
- `ssdlite320_mobilenet_v3_large` — detector CNN one-stage.
- `fasterrcnn_mobilenet_v3_large_320_fpn` — detector CNN two-stage.


## Trabajo en equipo

Si vas a pulir la parte visual/front, empezá por:

- `app.py` para layout, tabs, textos, métricas y gráficos.
- `.streamlit/config.toml` para colores/tema global.
- `docs/VISUAL_GUIDE.md` para una guía rápida de mejoras visuales.

La UI ya evita emojis, usa estilo glass, captions explicativos y una pestaña de guía de controles para que cualquier alumno entienda para qué sirve cada opción.

No subas `.venv/`, pesos `.pt`, videos pesados ni resultados generados. Los pesos YOLO se guardan fuera del repo en `~/.cache/yolo-complexity-lab/weights/`.

## Instalación recomendada

> Nota: no instales con `pip` del sistema. Este proyecto ya usa `.venv` para evitar el error `externally-managed-environment`.

```bash
cd /home/alejandro/OpenCode/.projects/apps/yolo-complexity-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En este equipo se validó con `python3.14` dentro de `.venv` y `torch 2.12.0+cu130`.

## Ejecutar

```bash
cd /home/alejandro/OpenCode/.projects/apps/yolo-complexity-lab
source .venv/bin/activate
streamlit run app.py
```

También podés correr desde el entregable visible:

```bash
/home/alejandro/OpenCode/outputs/university/complejidad-computacional/PROYECTO001_YOLO_COMPLEXITY_LAB_Alejandro_Padilla/run_app.sh
```

## Big-O usado

Complejidad convolucional principal:

```text
O(Σ_l H_l × W_l × C_in_l × C_out_l × K_l²)
```

Versión didáctica con `n = H × W`:

```text
O(L × n × C_in × C_out × K²)
```

Postprocesamiento con NMS:

```text
O(B²)
```

Para detectores two-stage se agrega costo por regiones:

```text
O(Σ conv + R × C_roi + NMS)
```

## Exportación

Los resultados quedan persistidos en la sesión de Streamlit. Podés descargar CSV o preparar gráficos HTML sin perder la tabla ni repetir el benchmark.

Los CSV se guardan automáticamente en:

```text
outputs/university/complejidad-computacional/PROYECTO001_YOLO_COMPLEXITY_LAB_Alejandro_Padilla/results/
```

## Validación rápida sin dependencias pesadas

```bash
cd /home/alejandro/OpenCode/.projects/apps/yolo-complexity-lab
python3 scripts/smoke_check.py
```

# YOLO Complexity Lab

App local en Streamlit para explicar **YOLO en tiempo real** y comparar **tiempo de ejecución** + **complejidad computacional** frente a detectores CNN clásicos.

## Enfoque

El recurso no intenta demostrar mAP (Mean Average Precision). El foco de la exposición es:

1. Mostrar **YOLO actual en vivo** con webcam local.
2. Comparar **Faster R-CNN / SSDlite / YOLO11n** con el mismo input.
3. Explicar tiempo y complejidad usando `n = H × W`.

Métricas principales:

- Latencia media y p95.
- FPS efectivo.
- `n = H × W` como tamaño de entrada.
- MACs/GFLOPs aproximados.
- Parámetros del modelo.
- Clases reconocidas como apoyo visual, no como métrica formal de precisión.

## Modelos incluidos

- `yolo11n.pt` — YOLO ligero.
- `yolo11s.pt` — YOLO mediano.
- `best.pt` — demo local opcional de gestos, si el archivo existe en el repo.
- `ssdlite320_mobilenet_v3_large` — detector CNN one-stage.
- `fasterrcnn_mobilenet_v3_large_320_fpn` — detector CNN two-stage.

## Flujo recomendado para la exposición

1. En el panel lateral dejá seleccionada la ruta **YOLO actual en vivo**.
2. Abrí la pestaña **Benchmark** y usá **Iniciar YOLO en vivo** para mostrar tiempo real.
3. Cambiá la ruta a **Comparación CNN vs YOLO**.
4. Ejecutá la comparación con la imagen demo persona/perro/fruta.
5. Explicá la conclusión:
   - con el mismo `n = H×W`, YOLO suele ganar en latencia/FPS;
   - GFLOPs es un proxy de costo;
   - el reconocimiento visual ayuda a discutir falsos positivos/falsos negativos.

## Trabajo en equipo

Si vas a pulir la parte visual/front, empezá por:

- `app.py` para layout, tabs, textos, métricas y gráficos.
- `.streamlit/config.toml` para colores/tema global.
- `docs/VISUAL_GUIDE.md` si quieren documentar nuevas decisiones visuales.

La UI actual usa tema claro, pocas tarjetas, textos cortos y una ruta de benchmark pensada para exposición.

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

Los resultados quedan persistidos en la sesión de Streamlit. Podés descargar CSV sin perder la tabla ni repetir el benchmark.

Los CSV se guardan automáticamente en:

```text
outputs/university/complejidad-computacional/PROYECTO001_YOLO_COMPLEXITY_LAB_Alejandro_Padilla/results/
```

## Validación rápida sin dependencias pesadas

```bash
cd /home/alejandro/OpenCode/.projects/apps/yolo-complexity-lab
python3 scripts/smoke_check.py
```

## Assets de demo

La imagen local `assets/demo_person_dog_fruit.jpg` combina:

- `zidane.jpg`, asset de ejemplo incluido con Ultralytics.
- Foto de perro con banana de Karsten Winegeart en Unsplash (`de5wBys0nok`).

Se usa solo como lámina didáctica local para comparar reconocimiento, falsos positivos y tiempos.

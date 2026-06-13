# Guía visual para pulir el front en Streamlit

Este proyecto es Python, pero la parte visual vive principalmente en `app.py` y en `.streamlit/config.toml`.

## Archivos que vas a tocar más

```text
app.py                    # layout principal, tabs, textos, tablas y gráficos
.streamlit/config.toml    # tema global: colores, fondo, fuente
src/yolo_complexity_lab/catalog.py  # nombres y textos explicativos de modelos/Big-O
```

## Zonas visuales actuales

La app está organizada en tabs:

1. **Resumen**: objetivo + catálogo de modelos.
2. **Benchmark**: carga de input, ejecución y gráficos.
3. **Big-O explicado**: explicación académica de complejidad.
4. **Sistema**: hardware detectado y ruta de exportación.

## Ideas de mejora visual

- Agregar una portada con cards: `YOLO`, `SSDlite`, `Faster R-CNN`.
- Convertir la tabla de Big-O en tarjetas comparativas.
- Usar colores por familia:
  - YOLO: azul/celeste.
  - CNN one-stage: verde.
  - CNN two-stage: naranja/rojo.
- Agregar una sección tipo “lectura rápida”:
  - Menor latencia = mejor para tiempo real.
  - Más GFLOPs = más operaciones aproximadas.
  - `O(B²)` en NMS puede afectar cuando hay muchas cajas.
- Agregar imágenes/diagramas propios en `assets/` si quieren una presentación más académica.

## Regla importante

No cambies primero la lógica de benchmark. Para pulir visual, empezá por:

1. Textos y estructura de tabs.
2. Gráficos Plotly.
3. Cards/métricas de Streamlit.
4. Tema global en `.streamlit/config.toml`.

Así evitás romper la medición.

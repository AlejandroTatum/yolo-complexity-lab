# Guía visual para pulir el front en Streamlit

Este proyecto usa Streamlit, pero debe sentirse como un recurso académico profesional, no como una demo improvisada.

## Tema adoptado: "tailwind" (nativo, sin CSS de maqueta)

La identidad visual vive en `.streamlit/config.toml`: tema basado en [awesome-streamlit-themes](https://github.com/jmedia65/awesome-streamlit-themes) (MIT) — fondo blanco, **Inter** para todo (fuente local en `static/`, servida por Streamlit), azul `#3b82f6` primario, bordes slate `#e2e8f0`, radio 0.5rem, sidebar `#f8fafc` y JetBrains Mono para código. Requiere Streamlit ≥ 1.50 (`theme.fontFaces`).

`inject_css()` en `app.py` queda solo para los componentes propios del lab (KPI strip, tabla minimal, pills de familia, badge, anillos de foco) y el hack que conserva el control para reabrir la sidebar colapsada. No reestiliza widgets: eso lo hace el tema.

## Principios visuales

1. **UI en español** (rioplatense). Los tests fijan la copy: si cambiás textos, actualizá `tests/test_entry_point_gating.py`.
2. **Cada control explica para qué sirve** con `help` corto; las tres decisiones viven en la sidebar con rótulos: Experimento / Entrada / Ejecución.
3. **Una sola acción primaria** por vista: el botón grande "Ejecutar benchmark" debajo de la imagen de entrada.
4. **Color con semántica**: naranja `#FF6600` = YOLO, verde `#10b981` = one-stage, ámbar `#eab308` = two-stage; azul para acciones/ganadores. Los colores de cajas, pills y gráficas son los mismos.
5. **No tocar la lógica de benchmark si solo estás puliendo front.**

## Estructura actual de la app

- **Header comercial** (global): marca + propuesta de valor + ficha técnica de la corrida.
- **Resumen:** qué mide el recurso, puente teoría→evidencia, diagrama Backbone → Neck → Head, glosario.
- **Benchmark:** Entrada interactiva (pills por modelo, foto propia con inferencia en vivo, cajas superpuestas reales), botón grande, KPIs, resultados, gráficas, reconocimiento y export CSV.
- **Acerca de:** propósito/método/evidencia + panel Environment con el hardware real.

## Próximas mejoras posibles

- Modo oscuro (el theming nativo 1.50+ ya soporta tema claro/oscuro separado).
- Evaluación cuantitativa estilo mAP sobre dataset fijo.
- Fuente de video además de imagen y webcam.

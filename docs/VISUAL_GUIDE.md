# Guía visual para pulir el front en Streamlit

Este proyecto usa Streamlit, pero debe sentirse como un recurso académico profesional, no como una demo improvisada.

## Principios visuales

1. **Sin emojis en la UI principal.** El tono debe ser limpio y presentable para exposición universitaria.
2. **Cada control debe explicar para qué sirve.** Si un alumno no sabe qué es `confianza` o `NMS`, la interfaz debe darle contexto.
3. **La estética debe apoyar la explicación.** Los efectos glass y colores deben guiar la lectura, no decorar sin propósito.
4. **No tocar la lógica de benchmark si solo estás puliendo front.** Primero layout, textos, cards, colores y gráficos.

## Archivos visuales principales

```text
app.py                    # layout, cards, tabs, captions, gráficos y CSS embebido
.streamlit/config.toml    # tema global base
docs/VISUAL_GUIDE.md      # guía para colaboradores
```

## Estructura actual de la app

- **Resumen:** explica qué mide el recurso, familias de modelos y cómo leer métricas.
- **Benchmark:** ejecuta medición real con configuración visible y resultados exportables.
- **Complejidad Big-O:** conecta la fórmula con resolución, canales, capas y NMS.
- **Guía de controles:** explica para qué sirve cada selector, slider y botón.
- **Sistema:** muestra hardware local para contextualizar los resultados.

## Paleta actual

- Fondo: azul noche / casi negro.
- Principal: celeste técnico para YOLO y acciones.
- Verde: comparación one-stage ligera.
- Ámbar: modelos/costos más pesados.
- Violeta: información conceptual o configuración.

## Componentes visuales ya implementados

- Hero principal con estilo glass.
- Cards glass por familia de modelo.
- Botón principal con efecto flotado/glass.
- Métricas destacadas en tarjetas.
- Captions visibles debajo de controles críticos.
- Gráficos Plotly con paleta consistente.
- Tab dedicado para explicar controles.
- Descarga HTML persistente: exportar gráficos ya no borra el benchmark.

## Próximas mejoras posibles

- Agregar logo propio del grupo o de la materia.
- Agregar un diagrama simple Backbone → Neck → Head.
- Agregar modo claro/oscuro si el docente prefiere fondo blanco.
- Agregar screenshots de ejemplo en el README.
- Crear una página de “conclusiones” que se llene automáticamente después del benchmark.

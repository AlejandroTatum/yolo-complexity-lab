# Guía de Presentación - YOLO Complexity Lab

## Antes de la Expo (30 min antes)

1. **Abrir la app local** (para demo principal):
   ```bash
   cd /home/alejandro/OpenCode/.projects/apps/yolo-complexity-lab
   source .venv/bin/activate
   streamlit run app.py --server.port 8501
   ```
   
2. **Abrir la app en cloud** (para que vean en celulares):
   - Abrí `https://yolo-complexity-lab-unl.streamlit.app/`
   - Esperá 2-3 min si está "sleeping"
   - Compartí el link en el grupo de chat

## Estructura de la Presentación (10-15 minutos)

### 1. Introducción (2 min)
- **¿Qué es YOLO?** - Detector de objetos en tiempo real, una sola pasada
- **¿Qué hace esta app?** - Compara YOLO vs CNNs clásicos por tiempo y complejidad
- **Contexto:** Proyecto para la materia de Complejidad Computacional

### 2. Demo - Comparación CNN vs YOLO (3 min)
- **Ruta:** "Comparación CNN vs YOLO"
- **Fuente:** "Demo persona/perro/fruta"
- **Ejecutar:** Click en "Ejecutar comparación"
- **Mientras corre:** Explicar los 3 modelos que se comparan
  - Faster R-CNN (2 etapas, más lento)
  - SSDLite (1 etapa, más rápido)
  - YOLO11n (1 etapa, optimizado)

### 3. Mostrar Resultados (3 min)
- **Tabla comparativa:** Latencia, FPS, GFLOPs, parámetros
- **Gráficos:** Barras para visualizar diferencias
- **Detección visual:** Qué detectó cada modelo en la imagen
- **Interpretación:** Explicar que YOLO procesa más rápido porque es una sola pasada

### 4. Demo - Subir Imagen (2 min)
- Pedir a un compañero que suba una foto desde su celular
- Mostrar que la app procesa cualquier imagen
- Comparar resultados

### 5. Demo - YOLO en Vivo (3 min, solo en local)
- **Ruta:** "YOLO actual en vivo"
- **Fuente:** "Webcam OpenCV local"
- **Mostrar:** Cámara funcionando en tiempo real
- **Explicar:** Latencia por frame, FPS efectivo
- **Opcional:** Mostrar cómo detecta objetos en el aula

### 6. Cierre (1 min)
- Resumir: YOLO es más rápido porque optimiza la arquitectura
- Mostrar el link de la app para que prueben después
- Agradecer

## Tips para el Día de la Expo

### Si algo falla:
- **App no carga:** Recargar con `Ctrl+F5` o `Cmd+Shift+R`
- **Webcam no funciona:** Probá con la demo de imagen en vez de webcam
- **Error en cloud:** Usá la app local como backup
- **Internet lento:** La demo local no necesita internet

### Palabras clave para usar:
- "Complejidad computacional"
- "Latencia por frame"
- "FPS efectivos"
- "GFLOPs como proxy de complejidad"
- "Pasada única vs múltiples etapas"
- "Trade-off entre velocidad y precisión"

### Mostrar en celular:
- Compartí el link: `https://yolo-complexity-lab-unl.streamlit.app/`
- Deciles que prueben "Comparación CNN vs YOLO"
- Pueden subir sus propias fotos

## Recuerda
- **La app local es el plan A** (más rápida, tiene webcam)
- **La app cloud es el plan B** (accesible para todos, pero más lenta)
- **Si todo falla:** Mostrá screenshots o grabaciones de pantalla
- **Sonreí y hablá claro** - ¡Esta es tu app, mostrála con orgullo!

## Links Importantes
- **App Local:** `http://localhost:8501`
- **App Cloud:** `https://yolo-complexity-lab-unl.streamlit.app/`
- **Repo:** `https://github.com/AlejandroTatum/yolo-complexity-lab`

---

¡Éxitos en la expo! 🚀

# Deploy en Streamlit Cloud (CPU-only)

## Para compañeros: acceso a la app

**URL del deploy (cuando esté listo):** `https://alejandro-tatum-yolo-complexity-lab.streamlit.app`

## Requisitos para el deploy

1. **Repo público en GitHub** (ya está en `https://github.com/AlejandroTatum/yolo-complexity-lab`)
2. **Archivo principal**: `app.py` (ya está en la raíz)
3. **Dependencias**: `requirements.txt` (ya está en la raíz)
4. **Configuración**: `.streamlit/config.toml` (ya está configurado)

## Pasos para deployar (5 minutos)

1. Ir a **[share.streamlit.io](https://share.streamlit.io)**
2. Iniciar sesión con **GitHub**
3. Click en **"New app"**
4. Seleccionar el repo: `AlejandroTatum/yolo-complexity-lab`
5. Seleccionar rama: `main`
6. Archivo principal: `app.py` (debe detectarlo automáticamente)
7. Click en **Deploy**

## Notas importantes

- **GPU**: Streamlit Cloud usa **CPU-only**. El benchmark será más lento que en tu máquina local, pero funciona.
- **Webcam**: El modo webcam **NO funciona** en deploy remoto (no tiene acceso a tu cámara). Solo funciona: **Demo** y **Subir imagen**.
- **Dataset COCO**: El benchmark exhaustivo con COCO val2017 no está en el deploy. Solo la demo está disponible.
- **Modelo de gestos** (`best.pt`): El modelo personalizado de 131MB no está en el repo (es muy grande para GitHub). Solo los modelos descargables automáticamente (YOLO11n, Faster R-CNN, SSDlite) funcionan.

## Qué funciona en el deploy

- Tab **Inicio**: Ruta de comparación, uso rápido, glosario de métricas
- Tab **Benchmark**: Demo persona/perro/fruta, subir imagen, comparación de modelos
- Métricas: Latencia, FPS, detecciones, GFLOPs
- **NO funciona**: Webcam, COCO val2017 exhaustivo, modelo de gestos personalizado

## Solución de problemas

Si el deploy falla por dependencias pesadas:
1. Ir a **Settings** en Streamlit Cloud
2. Aumentar **Memory** a 2GB o 3GB
3. Reintentar deploy

## Compartir con compañeros

Una vez deployado, comparte el link directo:
```
https://alejandro-tatum-yolo-complexity-lab.streamlit.app
```

Los compañeros pueden abrirlo desde el celular o computadora sin instalar nada.

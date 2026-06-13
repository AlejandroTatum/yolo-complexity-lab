# Cómo colaborar

## Instalación local

```bash
git clone <URL_DEL_REPO>
cd yolo-complexity-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## Flujo recomendado

```bash
git checkout -b feat/nombre-corto
# hacer cambios
git status
git add archivo1 archivo2
git commit -m "feat: describe el cambio"
git push -u origin feat/nombre-corto
```

Luego abrir un Pull Request.

## Qué NO subir

- `.venv/`
- pesos `.pt`, `.onnx`, `.engine`
- resultados grandes de benchmark
- datos personales
- videos pesados

## Roles sugeridos

- Visual/front Streamlit: layout, tabs, cards, colores, gráficos.
- Benchmark: tiempos, FPS, MACs/GFLOPs, exportación CSV.
- Investigación: explicación Big-O, citas, interpretación académica.

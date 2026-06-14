import streamlit as st
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.title("Interacción en Tiempo Real con YOLO")
st.markdown("Demostración de complejidad $O(1)$: la interfaz reacciona instantáneamente al modelo.")

@st.cache_resource
def load_model_gestures():
    # Descarga y carga automatizada desde Hugging Face
    #model_path = hf_hub_download(repo_id="lewiswatson/yolov8x-tuned-hand-gestures", filename="model.pt")
    return YOLO("../../best.pt") 

try:
    model_gestures = load_model_gestures()
    st.success("Modelo cargado correctamente en GPU.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

reactions = {
    "thumbs up": "https://media.giphy.com/media/11ISwbgCxEzMyY/giphy.gif",
    "peace": "https://media.giphy.com/media/3o7TKoWXm3okO1kgHC/giphy.gif",
    "stop": "https://media.giphy.com/media/l2Jhtx8gP0EInQn9C/giphy.gif",
    "Ninguno": "https://via.placeholder.com/400x300?text=Esperando+Gesto..."
}

col_video, col_reaction = st.columns(2)

with col_video:
    st.subheader("Entrada de Cámara (YOLO)")
    video_placeholder = st.empty()

with col_reaction:
    st.subheader("Respuesta Dinámica")
    texto_gesto = st.empty()
    reaction_placeholder = st.empty()

st.markdown("---")
run_camera = st.checkbox("Encender Cámara", key="cam_gestos")

if run_camera:
    cap = cv2.VideoCapture(0)
    
    while run_camera:
        success, frame = cap.read()
        if not success:
            st.error("No se pudo acceder a la cámara.")
            break
            
        results = model_gestures.predict(frame, conf=0.5, verbose=False)
        
        detected_gesture = "Ninguno"
        if len(results[0].boxes) > 0:
            class_id = int(results[0].boxes.cls[0].item())
            detected_gesture = model_gestures.names[class_id]
            
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        video_placeholder.image(annotated_frame, channels="RGB", width="stretch")
        texto_gesto.markdown(f"**Gesto Actual:** `{detected_gesture}`")
        
        reaction_url = reactions.get(detected_gesture, "https://via.placeholder.com/400x300?text=" + detected_gesture)
        reaction_placeholder.image(reaction_url, width="stretch")
        
    cap.release()
else:
    st.info("Marca la casilla para encender la cámara.")

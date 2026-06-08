import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mtcnn import MTCNN
import os
import database as db

MODEL_DIR = "models"
final_model_path = os.path.join(MODEL_DIR, "face_recognition_final.keras")

def show_predict_image():
    st.title("Analyse d'Image")
    st.write("Televersez une photo pour identifier les personnes presentes.")
    
    # Parametrage du seuil
    threshold = st.slider("Seuil de confiance minimum", min_value=0.50, max_value=1.00, value=0.75, step=0.01)
    
    uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Chargement du modele et des classes
        if not os.path.exists(final_model_path):
            st.error("Le modele final n'existe pas. Veuillez lancer l'entrainement depuis le Tableau de bord.")
            return
            
        model = tf.keras.models.load_model(final_model_path)
        class_names = db.get_class_names()
        detector = MTCNN()
        
        # Convertir le fichier uploade pour OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        display_frame = img.copy()
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with st.spinner("Analyse et detection des visages en cours..."):
            faces = detector.detect_faces(rgb)
            
        if not faces:
            st.warning("Aucun visage n'a ete detecte sur cette image.")
            st.image(rgb, use_column_width=True)
            return
            
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            w, h = min(w, img.shape[1]-x), min(h, img.shape[0]-y)
            
            face_crop = img[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
                
            # Prediction
            face_resized = cv2.resize(face_crop, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(np.array(face_rgb, dtype=np.float32))
            face_input = np.expand_dims(face_input, axis=0)
            
            pred = model.predict(face_input, verbose=0)[0]
            idx = np.argmax(pred)
            confidence = pred[idx]
            
            if confidence >= threshold:
                label = f"{class_names[idx]} ({confidence*100:.1f}%)"
                color = (0, 255, 0) # Vert
            else:
                label = f"Inconnu (Plus proche: {class_names[idx]} {confidence*100:.1f}%)"
                color = (0, 0, 255) # Rouge
                
            # Dessin sur l'image de rendu
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(display_frame, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            
        # Affichage final dans Streamlit
        st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
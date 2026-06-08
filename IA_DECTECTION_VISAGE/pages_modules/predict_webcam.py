import streamlit as st
import cv2
import os
import subprocess
from pages_modules.predict_image import show_predict_image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mtcnn import MTCNN
import database as db

MODEL_DIR = "models"
final_model_path = os.path.join(MODEL_DIR, "face_recognition_final.keras")

def show_predict_webcam():
    st.title("Analyse Camera (Webcam)")
    
    mode = st.radio("Selectionnez le mode de capture", ["Prendre une photo instantanee", "Reconnaissance en Temps Reel (Fenetre active)"])
    
    if mode == "Prendre une photo instantanee":
        st.subheader("Prise de photo via l'interface")
        img_file_buffer = st.camera_input("Autorisez l'acces a votre camera")
        
        if img_file_buffer is not None:
            if not os.path.exists(final_model_path):
                st.error("Modele introuvable. Veuillez entrainer votre modele.")
                return
                
            model = tf.keras.models.load_model(final_model_path)
            class_names = db.get_class_names()
            detector = MTCNN()
            
            # Traitement de la photo prise
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            display_frame = cv2_img.copy()
            
            rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            
            if not faces:
                st.warning("Aucun visage detecte.")
                return
                
            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                w, h = min(w, cv2_img.shape[1]-x), min(h, cv2_img.shape[0]-y)
                
                face_crop = cv2_img[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
                    
                face_resized = cv2.resize(face_crop, (224, 224))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(np.array(face_rgb, dtype=np.float32))
                face_input = np.expand_dims(face_input, axis=0)
                
                pred = model.predict(face_input, verbose=0)[0]
                idx = np.argmax(pred)
                confidence = pred[idx]
                
                label = f"{class_names[idx]} ({confidence*100:.1f}%)" if confidence >= 0.75 else "Inconnu"
                color = (0, 255, 0) if confidence >= 0.75 else (0, 0, 255)
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
            st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            
    elif mode == "Reconnaissance en Temps Reel (Fenetre active)":
        st.subheader("Execution du flux continu")
        st.write("Le clic sur le bouton ci-dessous va initialiser votre flux video local base sur OpenCV.")
        st.write("Une fenetre externe s'affichera. Appuyez sur la touche 'Q' de votre clavier pour la fermer.")
        
        if st.button("Demarrer la camera temps reel"):
            with st.spinner("Lancement de la camera..."):
                # Execute de maniere propre votre script Realtimeprediction.py existant
                subprocess.run(["python", "Realtimeprediction.py"])
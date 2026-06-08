import os
import cv2
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mtcnn import MTCNN
from pathlib import Path

DATASET_DIR = "dataset"
CROPPED_DIR = "faces_cropped"
MODEL_DIR = "models"

final_model_path = os.path.join(MODEL_DIR, "face_recognition_final.keras")


# ====================== RECONNAISSANCE EN TEMPS RÉEL (VERSION ROBUSTE) ======================
def real_time_recognition(threshold=0.78, min_face_size=80):
    """
    threshold : confiance minimale pour afficher le nom
    min_face_size : taille minimale du visage en pixels (évite les faux positifs)
    """
    try:
        model = tf.keras.models.load_model(final_model_path)
        with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
            class_names = json.load(f)
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = MTCNN()
    print("🎥 Reconnaissance en temps réel démarrée (Appuie sur 'q' pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Copie pour affichage
        display_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            faces = detector.detect_faces(rgb)
        except Exception as e:
            print(f"Erreur MTCNN: {e}")
            faces = []

        for face in faces:
            x, y, w, h = face['box']
            confidence_face = face.get('confidence', 0)
            
            # Filtrage robuste
            if w < min_face_size or h < min_face_size or confidence_face < 0.7:
                continue
                
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame.shape[1]-x), min(h, frame.shape[0]-y)
            
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                continue

            # Préparation pour le modèle
            try:
                face_resized = cv2.resize(face_crop, (224, 224))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(np.array(face_rgb, dtype=np.float32))
                face_input = np.expand_dims(face_input, axis=0)

                predictions = model.predict(face_input, verbose=0)[0]
                idx = np.argmax(predictions)
                confidence = predictions[idx]

                if confidence >= threshold:
                    label = f"{class_names[idx]} {confidence*100:.1f}%"
                    color = (0, 255, 0)
                else:
                    label = "Inconnu"
                    color = (0, 0, 255)

                # Dessin
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
                
            except Exception as e:
                continue  # Skip ce visage en cas d'erreur

        cv2.imshow("Reconnaissance Faciale - Appuie sur Q", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Session terminée.")
# Lancer la reconnaissance en temps réel
real_time_recognition(threshold=0.75, min_face_size=70)
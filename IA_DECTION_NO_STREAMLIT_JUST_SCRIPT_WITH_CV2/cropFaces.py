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


# ==================== CONFIGURATION ====================
DATASET_DIR = "dataset"
CROPPED_DIR = "faces_cropped"
MODEL_DIR = "models"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Modifie ici avec tes vrais noms
PERSONNES = os.listdir("./dataset")   # ← Change selon tes dossiers


detector = MTCNN()

def crop_face_very_robust(input_path, output_path, required_size=(224, 224)):
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Impossible de lire : {input_path}")
            return False
        
        # Vérification taille minimale
        if img.shape[0] < 50 or img.shape[1] < 50:
            print(f"Image trop petite → {input_path}")
            # Fallback simple : redimensionner directement
            face = cv2.resize(img, required_size)
            cv2.imwrite(output_path, face)
            return True
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            results = detector.detect_faces(img_rgb)
        except Exception as e:
            print(f"MTCNN crash → fallback sur {input_path}")
            results = []
        
        if len(results) > 0:
            # Prendre le meilleur visage
            best_face = max(results, key=lambda x: x.get('confidence', 0))
            x, y, w, h = best_face['box']
            
            # Protection contre les coordonnées négatives ou trop grandes
            x, y = max(0, x), max(0, y)
            w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
            
            if w < 40 or h < 40:
                raise ValueError("Visage trop petit")
                
            face = img_rgb[y:y+h, x:x+w]
        else:
            print(f"Aucun visage détecté → fallback {input_path}")
            face = img_rgb  # On prend toute l'image
        
        # Redimensionnement final
        face = cv2.resize(face, required_size)
        cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        return True
        
    except Exception as e:
        print(f"❌ Erreur grave sur {input_path} → fallback simple")
        try:
            # Dernier recours : redimensionner l'image originale
            img = cv2.imread(input_path)
            if img is not None:
                face = cv2.resize(img, required_size)
                cv2.imwrite(output_path, face)
                return True
        except:
            pass
        return False


# ====================== LANCER LE CROPPING ======================
print("Début du cropping ROBUSTE...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue
    
    output_dir = os.path.join(CROPPED_DIR, person)
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for img_name in os.listdir(person_path):
        input_path = os.path.join(person_path, img_name)
        output_path = os.path.join(output_dir, img_name)
        
        if crop_face_very_robust(input_path, output_path):
            success_count += 1
    
    print(f"✅ {person} : {success_count} images traitées")

print("Cropping terminé avec fallback robuste !")
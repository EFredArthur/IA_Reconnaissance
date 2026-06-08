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


def predict_image(image_path, threshold=0.75):
    model = tf.keras.models.load_model(final_model_path)
    with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
        class_names = json.load(f)
    
    detector = MTCNN()
    img = cv2.imread(image_path)
    if img is None:
        print("Impossible de charger l'image")
        return
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    
    if not faces:
        print("Aucun visage détecté")
        return
    
    x, y, w, h = faces[0]['box']
    face_crop = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, (224, 224))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_input = preprocess_input(np.array(face_rgb, dtype=np.float32))
    face_input = np.expand_dims(face_input, axis=0)
    
    pred = model.predict(face_input, verbose=0)[0]
    idx = np.argmax(pred)
    confidence = pred[idx]
    
    if confidence >= threshold:
        print(f"✅ {class_names[idx]} ({confidence*100:.1f}%)")
    else:
        print(f"❓ Inconnu ({confidence*100:.1f}%) - le plus proche est {class_names[idx]}")
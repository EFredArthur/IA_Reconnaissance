import streamlit as st
import database as db
import os
import sys
import json
import cv2
import numpy as np

# On configure l'environnement au plus haut niveau pour couper les logs TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration des chemins d'accès locaux
DATASET_DIR = "dataset"
CROPPED_DIR = "faces_cropped"
MODEL_DIR = "models"

# Callback local d'apprentissage couplé à l'interface Streamlit
class StreamlitTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, total_epochs, offset_epoch=0):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs
        self.offset_epoch = offset_epoch

    def on_epoch_end(self, epoch, logs=None):
        real_epoch = self.offset_epoch + epoch + 1
        percent = min(real_epoch / self.total_epochs, 1.0)
        
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        self.progress_bar.progress(percent)
        self.status_text.text(
            f"Étape 2/2 : Entraînement de l'I.A... Époque {real_epoch}/{self.total_epochs}\n"
            f"Précision (Train): {acc*100:.1f}% | Précision (Val): {val_acc*100:.1f}%"
        )


def execute_native_cropping(progress_bar, status_text):
    """Exécute le recadrage robuste des visages directement dans Streamlit."""
    os.makedirs(CROPPED_DIR, exist_ok=True)
    
    if not os.path.exists(DATASET_DIR):
        st.error("Le dossier 'dataset' source est introuvable.")
        return False

    # Décompte initial des fichiers
    total_images = 0
    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if os.path.isdir(person_path):
            total_images += len([f for f in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, f))])

    if total_images == 0:
        st.error("Aucune image brute trouvée dans le dossier 'dataset'.")
        return False

    status_text.text("Initialisation du modèle de détection MTCNN...")
    try:
        detector = MTCNN()
    except Exception as e:
        st.error(f"Échec d'initialisation de MTCNN : {e}")
        return False

    current_index = 0

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue
        
        output_dir = os.path.join(CROPPED_DIR, person)
        os.makedirs(output_dir, exist_ok=True)
        
        for img_name in os.listdir(person_path):
            input_path = os.path.join(person_path, img_name)
            output_path = os.path.join(output_dir, img_name)
            
            try:
                img = cv2.imread(input_path)
                if img is None:
                    raise ValueError()
                
                if img.shape[0] < 50 or img.shape[1] < 50:
                    face = cv2.resize(img, (224, 224))
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = detector.detect_faces(img_rgb)
                    
                    if results:
                        best_face = max(results, key=lambda x: x.get('confidence', 0))
                        x, y, w, h = best_face['box']
                        x, y = max(0, x), max(0, y)
                        w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
                        face_crop = img[y:y+h, x:x+w]
                        face = cv2.resize(face_crop, (224, 224)) if face_crop.size > 0 else cv2.resize(img, (224, 224))
                    else:
                        face = cv2.resize(img, (224, 224))
                
                cv2.imwrite(output_path, face)
            except Exception:
                try:
                    img = cv2.imread(input_path)
                    if img is not None:
                        cv2.imwrite(output_path, cv2.resize(img, (224, 224)))
                except:
                    pass
            
            current_index += 1
            percent = min(current_index / total_images, 1.0)
            progress_bar.progress(percent)
            status_text.text(f"Étape 1/2 : Recadrage des visages... ({current_index} / {total_images} images)")
            
    return True


def execute_native_training(progress_bar, status_text):
    """Exécute l'entraînement du réseau de neurones MobileNetV2."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CROPPED_DIR):
        st.error("Dossier de visages recadrés introuvable.")
        return False
        
    classes = [d for d in os.listdir(CROPPED_DIR) if os.path.isdir(os.path.join(CROPPED_DIR, d))]
    if len(classes) < 2:
        st.error("L'entraînement requiert au moins 2 profils distincts contenant des images.")
        return False

    status_text.text("Étape 2/2 : Indexation des dossiers et data augmentation...")
    
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    try:
        train_gen = datagen.flow_from_directory(
            CROPPED_DIR, target_size=(224, 224), batch_size=16,
            class_mode='sparse', subset='training', shuffle=True
        )
        val_gen = datagen.flow_from_directory(
            CROPPED_DIR, target_size=(224, 224), batch_size=16,
            class_mode='sparse', subset='validation', shuffle=False
        )
    except Exception as e:
        st.error(f"Erreur d'accès aux images d'entraînement : {e}")
        return False

    if train_gen.samples == 0:
        st.error("Le jeu de données généré ne contient aucun échantillon d'apprentissage.")
        return False

    # Construction du modèle profond
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(len(train_gen.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    total_epochs = 15
    
    # Phase 1 : Entraînement de la tête de classification
    model.fit(
        train_gen, validation_data=val_gen, epochs=10, verbose=0,
        callbacks=[StreamlitTrainingCallback(progress_bar, status_text, total_epochs, offset_epoch=0)]
    )

    # Phase 2 : Fine-Tuning de l'architecture haute
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_gen, validation_data=val_gen, epochs=5, verbose=0,
        callbacks=[StreamlitTrainingCallback(progress_bar, status_text, total_epochs, offset_epoch=10)]
    )

    # Sauvegardes structurelles sur le stockage local
    model.save(os.path.join(MODEL_DIR, "face_recognition_final.keras"))
    
    class_names = list(train_gen.class_indices.keys())
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    return True


def show_dashboard():
    st.title("Tableau de bord")
    st.subheader("Statistiques du système")
    
    nb_personnes, nb_images = db.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Personnes enregistrées", value=nb_personnes)
    with col2:
        st.metric(label="Total des images brutes", value=nb_images)
        
    st.markdown("---")
    st.subheader("Gestion de l'Intelligence Artificielle")
    
    if nb_personnes < 2:
        st.warning("L'apprentissage requiert l'enregistrement d'au moins 2 profils contenant des images pour s'exécuter.")
        return

    if st.button("Lancer la mise à jour complète du Modèle"):
        st.markdown("### Suivi de l'apprentissage")
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        
        # Exécution native sans sous-processus
        crop_success = execute_native_cropping(progress_bar, status_text)
        
        if not crop_success:
            return
            
        progress_bar.progress(0.0)
        status_text.text("Initialisation de l'étape de classification (MobileNetV2)...")
        
        train_success = execute_native_training(progress_bar, status_text)
        
        if train_success:
            progress_bar.progress(1.0)
            status_text.text("Modèle optimisé mis à jour avec succès !")
            st.success("Succès ! Le fichier d'analyse a été généré à la racine. Le système de prédiction fixe et Live est opérationnel.")    
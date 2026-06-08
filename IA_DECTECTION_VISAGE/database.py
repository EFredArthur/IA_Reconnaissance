import os
import shutil
import json

DATASET_DIR = "dataset"
CROPPED_DIR = "faces_cropped"
MODEL_DIR = "models"

def get_stats():
    """Calcule les statistiques pour le tableau de bord."""
    if not os.path.exists(DATASET_DIR):
        return 0, 0
    
    personnes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    total_images = 0
    
    for p in personnes:
        p_path = os.path.join(DATASET_DIR, p)
        total_images += len([f for f in os.listdir(p_path) if os.path.isfile(os.path.join(p_path, f))])
        
    return len(personnes), total_images

def add_person(name, uploaded_files):
    """Ajoute une nouvelle personne et sauvegarde ses images."""
    if not name:
        return False, "Le nom ne peut pas etre vide."
    
    clean_name = name.strip().replace(" ", "_")
    person_dir = os.path.join(DATASET_DIR, clean_name)
    
    if os.path.exists(person_dir):
        return False, f"La personne '{name}' existe deja."
        
    os.makedirs(person_dir, exist_ok=True)
    
    for file in uploaded_files:
        file_path = os.path.join(person_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
            
    return True, f"Inscription reussie pour {name} ({len(uploaded_files)} images enregistrees)."

def delete_person(name):
    """Supprime definitivement une personne et ses visages croppes."""
    dataset_path = osface_management.py.path.join(DATASET_DIR, name)
    cropped_path = os.path.join(CROPPED_DIR, name)
    
    deleted = False
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        deleted = True
    if os.path.exists(cropped_path):
        shutil.rmtree(cropped_path)
        deleted = True
        
    if deleted:
        return True, f"La personne '{name}' a ete supprimee avec succes."
    return False, "Utilisateur introuvable."

def list_persons():
    """Liste toutes les personnes enregistrees."""
    if not os.path.exists(DATASET_DIR):
        return []
    return [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

def get_class_names():
    """Charge les noms de classes depuis le fichier JSON."""
    json_path = os.path.join(MODEL_DIR, "class_names.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return []
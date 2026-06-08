import streamlit as st
import os

# Configuration de la page (Design epure sans emoji)
st.set_page_config(
    page_title="Systeme de Reconnaissance Faciale",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import des modules de pages
from pages_modules.dashboard import show_dashboard
from pages_modules.face_management import show_management
from pages_modules.predict_image import show_predict_image
from pages_modules.predict_webcam import show_predict_webcam

# Initialisation des dossiers requis
for dossier in ["dataset", "faces_cropped", "models"]:
    os.makedirs(dossier, exist_ok=True)

# Menu de navigation lateral
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    ["Tableau de bord", "Gestion des profils", "Analyse d'image fixe", "Analyse via Webcam / Live"]
)

st.sidebar.markdown("---")
st.sidebar.info("Application de reconnaissance faciale basee sur MobileNetV2 et MTCNN.")

# Routage des pages
if page == "Tableau de bord":
    show_dashboard()
elif page == "Gestion des profils":
    show_management()
elif page == "Analyse d'image fixe":
    show_predict_image()
elif page == "Analyse via Webcam / Live":
    show_predict_webcam()
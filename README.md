# IA_Reconnaissance

#  Système Natif de Reconnaissance Faciale avec Streamlit & Deep Learning

Ce projet est une application web d'intelligence artificielle clé en main permettant de gérer des profils utilisateurs, de détecter et recadrer automatiquement des visages, d'entraîner un modèle de Deep Learning sur mesure (**MobileNetV2**) et de réaliser des prédictions en temps réel (via image ou webcam).

L'ensemble de l'application a été conçu de manière **native** sous Streamlit. Le traitement des données et l'apprentissage s'effectuent directement au sein de l'interface graphique, éliminant les crashs, les blocages de barres de progression et les conflits de scripts en arrière-plan.

---

##  Fonctionnalités du Système

* **Gestion des Profils :** Création de fiches profils et téléversement groupé de photos brutes.
* **Recadrage Automatique Étanche :** Intégration locale du framework **MTCNN** pour localiser, centrer et extraire les visages à la taille standard $224 \times 224$ pixels sans interférence système.
* **Pipeline d'Apprentissage Natif :** Entraînement par transfert d'apprentissage (*Transfer Learning*) puis ajustement fin (*Fine-Tuning*) avec un suivi dynamique des époques à l'écran.
* **Analyse d'Image Fixe :** Identification multi-visages sur photo téléversée avec encadrement de couleur et calcul du score de confiance.
* **Reconnaissance Webcam Live :** Flux vidéo fluide injecté directement dans le navigateur web sans ouverture de fenêtres système parasites.

---

## 📂 Structure du Projet

```text
📁 projet-reconnaissance-faciale/
│
├── 📄 faces_database.db       # Base de données SQLite (générée automatiquement)
├── 📄 database.py             # Fonctions de communication SQLite3
│
├── 📁 pages_modules/          # Modules de l'interface Streamlit
│   ├── 📄 dashboard.py       # Coeur unifié : Stats + Recadrage + Entraînement
│   ├── 📄 face_management.py  # Ajout des personnes et de leurs photos
│   ├── 📄 predict_image.py    # Analyse faciale sur image fixe
│   └── 📄 predict_webcam.py   # Flux webcam temps réel dans le navigateur
│
├── 📁 dataset/                # Stockage des photos brutes (organisé par nom)
├── 📁 faces_cropped/          # Visages normalisés 224x224 (généré par le traitement)
└── 📁 models/                 # Modèle d'I.A. .keras et étiquettes json (généré)

```

---

##  Installation

### 1. Prérequis

Assurez-vous d'avoir **Python 3.8** (ou une version supérieure) installé sur votre machine informatique.

### 2. Cloner le projet

```bash
git clone [https://github.com/votre-utilisateur/votre-repo.git](https://github.com/votre-utilisateur/votre-repo.git)
cd votre-repo

```

### 3. Installer les dépendances

Installez les bibliothèques requises à l'aide de votre gestionnaire de paquets :

```bash
pip install streamlit opencv-python numpy tensorflow mtcnn

```

---

##  Première Ouverture : Procédure de démarrage à blanc (Système Vierge)

Lorsque vous lancez l'application pour la première fois, la base de données, les répertoires d'images et le modèle de reconnaissance **n'existent pas**. Si vous vous rendez immédiatement sur l'onglet Analyse ou Webcam, le système affichera une alerte amicale vous indiquant que le modèle est introuvable.

Voici la procédure pas à pas pour configurer votre première intelligence artificielle de zéro :

### Étape 1 : Enregistrer les utilisateurs

1. Lancez l'application Streamlit avec la commande :
```bash
streamlit run main.py

```


2. Allez dans l'onglet **Gestion des Profils**.
3. Saisissez le *Nom / Prénom* d'une première personne (ex: `Jean_Dupont`).
4. Cliquez sur "Browse files" et sélectionnez plusieurs photos nettes de cette personne (idéalement entre 10 et 50 photos sous différents angles et éclairages).
5. Cliquez sur **Enregistrer le profil**.
6. **IMPORTANT :** Répétez l'opération pour une **deuxième personne au minimum** (ex: `Marie_Curie`). *L'algorithme de classification mathématique requiert au moins 2 classes distinctes pour pouvoir s'entraîner.*

### Étape 2 : Lancer le traitement et l'apprentissage

1. Rendez-vous sur l'onglet **Tableau de bord**. Vos statistiques affichent désormais le nombre de personnes et d'images brutes que vous venez d'ajouter.
2. Cliquez sur le bouton **Lancer la mise à jour complète du Modèle**.
3. **L'application s'occupe de tout :**
* **Phase 1 :** La barre de progression avance à mesure que le script extrait les visages de vos photos brutes pour les sauvegarder proprement dans `faces_cropped/`.
* **Phase 2 :** La barre se réinitialise et l'entraînement MobileNetV2 commence. Le système affiche la progression en temps réel époque par époque, ainsi que les scores de précision de l'I.A.


4. Patientez jusqu'au message de succès vert : *"Modèle optimisé mis à jour avec succès !"*.

### Étape 3 : Tester la reconnaissance

Vos fichiers de réseaux de neurones sont désormais compilés dans le dossier `models/`.

* Vous pouvez aller sur **Analyse d'Image Fixe** pour téléverser une photo inconnue.
* Vous pouvez activer la case à cocher dans **Analyse via Webcam** pour démarrer la reconnaissance faciale en direct.

---

##  Conseils pour obtenir une excellente précision

* **Variété :** Lors de l'ajout de photos dans la gestion des profils, mettez des photos avec des expressions différentes (sourire, sérieux), des éclairages variés et de légères inclinaisons de tête.
* **Seuil de confiance :** Utilisez le curseur de seuil présent sur les pages de prédiction. Un seuil à `0.80` signifie que l'I.A. doit être sûre à 80% avant d'afficher un nom, limitant grandement les faux positifs.


***


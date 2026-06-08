import streamlit as st
import database as db

def show_management():
    st.title("Gestion de la Base de Donnees")
    
    tab1, tab2 = st.tabs(["Ajouter une personne", "Supprimer une personne"])
    
    with tab1:
        st.subheader("Enregistrer un nouveau profil")
        with st.form("add_person_form", clear_on_submit=True):
            nom = st.text_input("Nom et Prenom")
            fichiers = st.file_uploader("Importer des photos (Plusieurs selectionnables)", 
                                        type=["jpg", "jpeg", "png"], 
                                        accept_multiple_files=True)
            submit = st.form_submit_button("Enregistrer le profil")
            
            if submit:
                if not fichiers:
                    st.error("Veuillez ajouter au moins une image.")
                else:
                    success, msg = db.add_person(nom, fichiers)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                        
    with tab2:
        st.subheader("Supprimer un profil existant")
        personnes = db.list_persons()
        
        if not personnes:
            st.info("Aucun profil enregistre pour le moment.")
        else:
            personne_a_supprimer = st.selectbox("Selectionner le profil a effacer", personnes)
            st.warning("Attention : Cette action supprimera les images sources ainsi que les visages recadres.")
            
            if st.button("Confirmer la suppression definitive"):
                success, msg = db.delete_person(personne_a_supprimer)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
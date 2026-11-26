"""Application Streamlit : PCA et clustering sur 4 datasets UCI.

Deux pages :
- l'analyse des datasets (page_analyse.py), à l'adresse principale ;
- la vidéo de présentation (page_video.py), à l'adresse /video.
Lancer avec :  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Exam ML : PCA et clustering", page_icon="📊", layout="wide")

pages = [
    st.Page("page_analyse.py", title="Exam ML : PCA et clustering", icon="📊", default=True),
    st.Page("page_video.py", title="Vidéo de présentation", icon="🎬", url_path="video"),
]
# Menu de navigation masqué : on passe d'une page à l'autre par les liens de la barre latérale
st.navigation(pages, position="hidden").run()

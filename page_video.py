"""Page « Vidéo de présentation », avec son propre lien : https://exam.jdlc-mikemy.tech/video"""
import os
from pathlib import Path

import streamlit as st

BASE = Path(__file__).parent
# La vidéo est servie par le conteneur nginx (exam-video) ; VIDEO_URL est fourni sur le serveur.
VIDEO_URL = os.environ.get("VIDEO_URL")
VIDEO_OK = bool(VIDEO_URL) and (BASE / "video" / Path(VIDEO_URL).name).is_file()

with st.sidebar:
    st.page_link("page_analyse.py", label="Retour à l'application", icon="📊")

st.title("Vidéo de présentation")
st.write("Exam ML : PCA et clustering. Explication du code des notebooks **Wine** et "
         "**Wholesale customers** (4 min chacun), puis démonstration de l'application (2 min).")

if VIDEO_OK:
    st.video(VIDEO_URL)
    st.markdown(f"[⬇️ Télécharger la vidéo]({VIDEO_URL})")
else:
    st.info("La vidéo n'est pas encore disponible.")

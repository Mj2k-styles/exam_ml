"""Application Streamlit : PCA et clustering sur 4 datasets UCI.

Datasets : Raisin, Wine, Wholesale customers (avec catégorie connue) et Clients télécom
(segmentation pure, sans catégorie : le notebook Customer_Segmentation).
Lancer avec :  streamlit run app.py
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

BASE = Path(__file__).parent

st.set_page_config(page_title="Exam ML : PCA et clustering", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- Chargement des données
# Même préparation que dans les notebooks : la variable cible 'Class' est en dernière colonne.
def load_raisin():
    return pd.read_excel(BASE / "Raisin_Dataset.xlsx")


def load_wine():
    df = pd.read_csv(BASE / "wine_dataset.csv")
    df["Class"] = df["Class"].map({1: "Cultivar 1", 2: "Cultivar 2", 3: "Cultivar 3"})
    return df[[c for c in df.columns if c != "Class"] + ["Class"]]


def load_wholesale():
    df = pd.read_csv(BASE / "wholesale_customers_dataset.csv")
    df["Class"] = df["Channel"].map({1: "Horeca", 2: "Retail"})
    return df.drop(columns=["Channel", "Region"])


def load_telecom():
    return pd.read_csv(BASE / "Customer_dataset.csv")


DATASETS = {
    "Raisin": dict(
        load=load_raisin, pca_model="raisin_pca_model.joblib", clust_model="raisin_clustering.joblib",
        unit="raisin", target="variété",
        desc="900 raisins secs de deux variétés (**Kecimen** et **Besni**) décrits par 7 variables "
             "morphologiques mesurées sur des images.",
        pca_txt="- **PC1 ≈ la taille** : Area, ConvexArea, Perimeter, MajorAxisLength.\n"
                "- **PC2 ≈ la forme** : Extent (compact) contre Eccentricity (allongé).\n\n"
                "Les deux variétés se séparent surtout le long de PC1 : les Besni sont en moyenne "
                "plus gros que les Kecimen.",
        clust_txt="Après `normalize`, Area et ConvexArea dominent la norme de chaque raisin : l'information "
                  "de taille disparaît et il ne reste que la régularité du contour (rapport "
                  "ConvexArea / Area). Les classes K-Means séparent donc les formes régulières des formes "
                  "irrégulières, et **ne retrouvent pas les variétés**. La CAH a une meilleure silhouette "
                  "(≈ 0,90) mais isole seulement 6 raisins atypiques : K-Means a été retenu.",
        source="https://archive.ics.uci.edu/dataset/850/raisin",
    ),
    "Wine": dict(
        load=load_wine, pca_model="wine_pca_model.joblib", clust_model="wine_clustering.joblib",
        unit="vin", target="cultivar",
        desc="178 vins italiens issus de **trois cultivars** (cépages), décrits par 13 variables "
             "chimiques.",
        pca_txt="- **PC1 ≈ les composés phénoliques** : Flavanoids, Total_phenols, OD280_OD315, "
                "Proanthocyanins.\n"
                "- **PC2 ≈ l'intensité du vin** : Color_intensity, Alcohol, Proline.\n\n"
                "Les cultivars 1 et 3 s'opposent le long de PC1, le cultivar 2 (vins moins alcoolisés "
                "et moins colorés) se distingue le long de PC2.",
        clust_txt="Après `normalize`, Proline (plusieurs centaines) domine la norme de chaque vin : les "
                  "classes K-Means se font surtout selon la teneur en proline. Le **Cultivar 1**, riche en "
                  "proline, est bien retrouvé (51 vins sur 59) ; les cultivars 2 et 3, de teneurs proches, "
                  "sont mélangés.",
        source="https://archive.ics.uci.edu/dataset/109/wine",
    ),
    "Wholesale customers": dict(
        load=load_wholesale, pca_model="wholesale_pca_model.joblib",
        clust_model="wholesale_clustering.joblib",
        unit="client", target="canal",
        desc="440 clients d'un grossiste portugais, décrits par leurs **dépenses annuelles** dans 6 "
             "catégories de produits, et répartis en deux canaux : **Horeca** (hôtels, restaurants, "
             "cafés) et **Retail** (commerce de détail).",
        pca_txt="- **PC1 ≈ l'épicerie et les produits ménagers** : Grocery, Detergents_Paper, Milk.\n"
                "- **PC2 ≈ les produits frais** : Frozen, Fresh, Delicassen.\n\n"
                "Les clients Retail se distinguent le long de PC1 par leurs gros achats d'épicerie.",
        clust_txt="Ici `normalize` transforme les dépenses en **répartition du budget** de chaque client. "
                  "K-Means sépare les clients qui dépensent surtout en produits frais (profil Horeca) de "
                  "ceux qui dépensent surtout en épicerie (profil Retail), avec environ 75 % de "
                  "concordance avec les vrais canaux.",
        source="https://archive.ics.uci.edu/dataset/292/wholesale+customers",
    ),
    # Notebook Customer_Segmentation : pas de catégorie connue, donc pas de modèle PCA + régression
    # logistique ; la PCA (données normalisées) sert seulement à visualiser les classes.
    "Clients télécom": dict(
        load=load_telecom, pca_model=None, clust_model="customer_clustering.joblib",
        unit="client", target=None,
        desc="3 150 clients d'un opérateur télécom, décrits par 12 variables d'utilisation (appels, "
             "SMS, durée d'abonnement, valeur client…). **Aucune catégorie connue** : on fait une "
             "segmentation pure (clustering).",
        pca_txt=None,
        clust_txt="Meilleur modèle : la **CAH** (silhouette ≈ 0,89 contre ≈ 0,78 pour K-Means). Elle sépare "
                  "158 clients **quasi inactifs** (≈ 2 secondes d'utilisation, valeur client ≈ 8), nommés "
                  "« non fidèles », des 2 992 autres clients actifs (≈ 4 700 secondes, valeur client "
                  "≈ 495), nommés « fidèles ».",
        source="https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
    ),
}


@st.cache_data
def get_data(name):
    return DATASETS[name]["load"]()


@st.cache_resource
def get_models(name):
    cfg = DATASETS[name]
    pca_pipe = joblib.load(BASE / cfg["pca_model"]) if cfg["pca_model"] else None
    return pca_pipe, joblib.load(BASE / cfg["clust_model"])


def predire_classe(clust, valeurs):
    """Même fonction que dans les notebooks : classe du centroïde le plus proche (après normalize)."""
    obs = normalize(np.asarray(valeurs, dtype=float).reshape(1, -1))
    distances = np.linalg.norm(clust["centroides"] - obs, axis=1)
    return clust["classes"][distances.argmin()]


# ---------------------------------------------------------------- Barre latérale
# Vidéo de présentation : l'entrée n'apparaît que sur le serveur (VIDEO_URL fourni)
# et une fois le fichier vidéo déposé dans le dossier video/.
VIDEO_URL = os.environ.get("VIDEO_URL")
VIDEO_OK = bool(VIDEO_URL) and (BASE / "video" / Path(VIDEO_URL).name).is_file()
VIDEO = "🎬 Vidéo de présentation"

with st.sidebar:
    st.header("Dataset")
    name = st.radio("Choisir un dataset", list(DATASETS) + ([VIDEO] if VIDEO_OK else []),
                    label_visibility="collapsed")
    if name == VIDEO:
        st.markdown("Présentation des notebooks **Wine** et **Wholesale customers**, "
                    "puis démonstration de l'application.")
    else:
        cfg = DATASETS[name]
        st.markdown(cfg["desc"])
        st.caption(f"Source : [UCI Machine Learning Repository]({cfg['source']})")

# ---------------------------------------------------------------- Page vidéo
if name == VIDEO:
    st.title("Vidéo de présentation")
    st.write("Explication du code des notebooks (4 min chacun), puis démonstration de l'application (2 min).")
    st.video(VIDEO_URL)
    st.markdown(f"[⬇️ Télécharger la vidéo]({VIDEO_URL})")
    st.stop()

df = get_data(name)
pca_pipe, clust = get_models(name)
SUPERVISE = pca_pipe is not None          # False : pas de catégorie connue (Clients télécom)
FEATURES = list(clust["variables"])
if SUPERVISE:
    assert FEATURES == list(pca_pipe.feature_names_in_), "Les modèles n'utilisent pas les mêmes variables"

# Partition du clustering (données normalisées, centroïde le plus proche) et sa PCA de visualisation
x_norm = normalize(df[FEATURES].values)
partition = np.array([predire_classe(clust, row) for row in df[FEATURES].values])
pca_norm = clust["pca"] if "pca" in clust else PCA(n_components=2).fit(x_norm)
proj_norm = pd.DataFrame(pca_norm.transform(x_norm), columns=["PC1", "PC2"])
proj_norm["Cluster"] = [clust["noms_classes"][c] for c in partition]


def projeter(X):
    """Coordonnées (PC1, PC2) dans le plan affiché pour ce dataset."""
    if SUPERVISE:
        return pca_pipe[:-1].transform(X)
    return pca_norm.transform(normalize(np.asarray(X, dtype=float)))


# Plan affiché : PCA des données standardisées (vraie catégorie en couleur) ou,
# sans catégorie connue, PCA des données normalisées (classe trouvée en couleur)
COULEUR = "Class" if SUPERVISE else "Cluster"
if SUPERVISE:
    proj = pd.DataFrame(projeter(df[FEATURES]), columns=["PC1", "PC2"])
    proj["Class"] = df["Class"].values
else:
    proj = proj_norm

st.title(f"PCA et clustering : {name}")

noms_onglets = ([f"Prédire un {cfg['unit']}", "Prédiction multiple"]
                + (["Analyse PCA"] if SUPERVISE else []) + ["Clustering", "Données"])
onglets = dict(zip(noms_onglets, st.tabs(noms_onglets)))
tab_pred, tab_multi = onglets[noms_onglets[0]], onglets["Prédiction multiple"]
tab_clust, tab_data = onglets["Clustering"], onglets["Données"]

# ---------------------------------------------------------------- Prédiction
with tab_pred:
    st.subheader(f"Caractéristiques du {cfg['unit']}")
    st.caption("Les valeurs par défaut sont les médianes du dataset.")
    cols = st.columns(4)
    values = {}
    for i, f in enumerate(FEATURES):
        s = df[f]
        is_int = pd.api.types.is_integer_dtype(s)
        with cols[i % 4]:
            values[f] = st.number_input(
                f,
                min_value=0.0,
                max_value=float(s.max()) * 2,
                value=float(s.median()),
                step=1.0 if is_int else float(round(s.std() / 20, 4)) or 0.01,
                format="%.0f" if is_int else "%.4f",
                key=f"{name}-{f}",
            )

    x_new = pd.DataFrame([values])[FEATURES]
    pc_new = projeter(x_new)[0]
    cluster_new = predire_classe(clust, x_new.values[0])

    c1, c2 = st.columns([1, 2])
    with c1:
        if SUPERVISE:
            proba = pca_pipe.predict_proba(x_new)[0]
            pred = pca_pipe.classes_[proba.argmax()]
            st.metric(f"{cfg['target'].capitalize()} prédit(e) (PCA + régression logistique)", pred,
                      f"{proba.max():.0%} de confiance", delta_color="off")
            st.dataframe(
                pd.DataFrame({cfg["target"].capitalize(): pca_pipe.classes_, "Probabilité": proba})
                .style.format({"Probabilité": "{:.1%}"}),
                hide_index=True,
            )
        st.metric(f"Classe {clust['modele']} (clustering)", f"Classe {cluster_new}")
        nom = clust["noms_classes"][cluster_new]
        st.info(f"C'est un {nom if nom.startswith(cfg['unit']) else cfg['unit'] + ' ' + nom}.")
    with c2:
        fig = px.scatter(proj, x="PC1", y="PC2", color=COULEUR, opacity=0.45,
                         title=f"Position du {cfg['unit']} saisi dans le plan (PC1, PC2)")
        fig.add_scatter(x=[pc_new[0]], y=[pc_new[1]], mode="markers", name=f"Votre {cfg['unit']}",
                        marker=dict(size=18, symbol="star", color="black",
                                    line=dict(width=1, color="white")))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Prédiction multiple
def lire_fichier(fichier):
    """Lit un CSV (séparateur , ou ; détecté automatiquement) ou un fichier Excel."""
    if fichier.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(fichier)
    return pd.read_csv(fichier, sep=None, engine="python")


def prediction_multiple(fichier):
    """Prédit chaque ligne du fichier chargé et affiche les résultats."""
    try:
        data = lire_fichier(fichier)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    data.columns = data.columns.astype(str).str.strip()
    manquantes = [f for f in FEATURES if f not in data.columns]
    if manquantes:
        st.error("Colonnes manquantes : " + ", ".join(f"`{f}`" for f in manquantes))
        return

    X_up = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    valides = X_up.notna().all(axis=1)
    if (~valides).any():
        st.warning(f"{(~valides).sum()} ligne(s) avec des valeurs manquantes ou non numériques "
                   "ont été ignorées.")
    if not valides.any():
        st.error("Aucune ligne exploitable dans le fichier.")
        return

    res = data[valides].copy()
    X_ok = X_up[valides]
    pcs = projeter(X_ok)
    clusters = np.array([predire_classe(clust, row) for row in X_ok.values])

    if SUPERVISE:
        probas = pca_pipe.predict_proba(X_ok)
        col_pred = f"{cfg['target'].capitalize()} prédit(e)"
        res[col_pred] = pca_pipe.classes_[probas.argmax(axis=1)]
        res["Confiance"] = probas.max(axis=1).round(3)
    res["PC1"], res["PC2"] = pcs[:, 0].round(3), pcs[:, 1].round(3)
    res["Classe clustering"] = clusters
    res["Profil clustering"] = [clust["noms_classes"][c] for c in clusters]

    st.success(f"{len(res)} {cfg['unit']}(s) prédit(s).")
    c1, c2 = st.columns(2)
    if SUPERVISE:
        with c1:
            st.markdown(f"**{col_pred}**")
            st.dataframe(res[col_pred].value_counts().rename("Nombre"))
    with (c2 if SUPERVISE else c1):
        st.markdown("**Profil clustering**")
        st.dataframe(res["Profil clustering"].value_counts().rename("Nombre"))

    fig = px.scatter(proj, x="PC1", y="PC2", color=COULEUR, opacity=0.25,
                     title=f"Les {cfg['unit']}s du fichier dans le plan (PC1, PC2)")
    fig.add_scatter(x=res["PC1"], y=res["PC2"], mode="markers", name="Fichier chargé",
                    marker=dict(size=11, symbol="star", color="black",
                                line=dict(width=1, color="white")),
                    text=res[col_pred if SUPERVISE else "Profil clustering"],
                    hovertemplate="%{text}<extra></extra>")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(res, width="stretch")
    st.download_button("Télécharger les résultats (CSV)", res.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{name.split()[0].lower()}.csv", mime="text/csv")


with tab_multi:
    st.subheader(f"Prédire plusieurs {cfg['unit']}s à partir d'un fichier")
    st.markdown(
        f"Le fichier (CSV ou Excel) doit contenir une ligne par {cfg['unit']} et ces "
        f"**{len(FEATURES)} colonnes** : " + ", ".join(f"`{f}`" for f in FEATURES) + ". "
        "Les autres colonnes sont ignorées et conservées dans le résultat."
    )
    exemple = df[FEATURES].sample(10, random_state=0)
    st.download_button("Télécharger un fichier exemple", exemple.to_csv(index=False).encode("utf-8"),
                       file_name=f"exemple_{name.split()[0].lower()}.csv", mime="text/csv")

    fichier = st.file_uploader("Charger un fichier", type=["csv", "xlsx", "xls"], key=f"upload-{name}")
    if fichier is not None:
        prediction_multiple(fichier)

# ---------------------------------------------------------------- Analyse PCA
if SUPERVISE:
    with onglets["Analyse PCA"]:
        scaler, pca = pca_pipe.named_steps["scaler"], pca_pipe.named_steps["pca"]
        ratio = pca.explained_variance_ratio_
        m1, m2, m3 = st.columns(3)
        m1.metric("Variance PC1", f"{ratio[0]:.1%}")
        m2.metric("Variance PC2", f"{ratio[1]:.1%}")
        m3.metric("Total (2 composantes)", f"{ratio.sum():.1%}")

        st.plotly_chart(
            px.scatter(proj, x="PC1", y="PC2", color="Class", opacity=0.7,
                       title=f"Les {len(df)} observations projetées sur les 2 premières composantes "
                             "(données standardisées)"),
            width="stretch",
        )

        z = pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES)
        corr = pd.DataFrame({pc: [z[f].corr(proj[pc]) for f in FEATURES] for pc in ["PC1", "PC2"]},
                            index=FEATURES)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Corrélations variables / composantes**")
            st.dataframe(corr.style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1))
        with c2:
            st.markdown(cfg["pca_txt"])

# ---------------------------------------------------------------- Clustering
with tab_clust:
    st.markdown(f"Modèle retenu : **{clust['modele']}** avec **{len(clust['classes'])} classes**, "
                "entraîné sur les données normalisées (`normalize`) "
                + ("sans utiliser la vraie catégorie." if SUPERVISE else
                   "(ce dataset n'a pas de catégorie connue)."))
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            px.scatter(proj_norm, x="PC1", y="PC2", color="Cluster", opacity=0.7,
                       title="Classes trouvées par le clustering (PCA des données normalisées)"),
            width="stretch",
        )
    with c2:
        counts = pd.Series(partition).value_counts().sort_index()
        st.plotly_chart(
            px.bar(x=counts.index.astype(str), y=counts.values, color=counts.index.astype(str),
                   text_auto=True, labels={"x": "Classe", "y": "Nombre d'observations"},
                   title="Nombre d'observations par classe"),
            width="stretch",
        )

    if SUPERVISE:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Classes trouvées / vraies catégories**")
            st.dataframe(pd.crosstab(df["Class"], partition, rownames=["Vraie catégorie"],
                                     colnames=["Classe"]))
        with c2:
            st.markdown(cfg["clust_txt"])
    else:
        st.markdown(cfg["clust_txt"])

    st.markdown("**Profil moyen de chaque classe**")
    profil = df[FEATURES].groupby(partition).mean().round(2)
    profil.index = [f"{c} : {clust['noms_classes'][c]}" for c in profil.index]
    st.dataframe(profil, width="stretch")

# ---------------------------------------------------------------- Données
with tab_data:
    st.dataframe(df, width="stretch")
    st.caption(f"Source : {cfg['source']}")

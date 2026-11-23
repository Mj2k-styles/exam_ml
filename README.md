# Exam ML : PCA et clustering sur 3 datasets UCI

Réduction de dimension (PCA) et clustering (K-Means, CAH) sur trois jeux de données du UCI Machine Learning Repository, avec une application Streamlit pour explorer les résultats et faire des prédictions.

## Datasets

| Dataset | Observations | Variables | Catégories | Source |
|---|---|---|---|---|
| Raisin | 900 raisins secs | 7 mesures morphologiques | Kecimen, Besni | https://archive.ics.uci.edu/dataset/850/raisin |
| Wine | 178 vins | 13 mesures chimiques | 3 cultivars | https://archive.ics.uci.edu/dataset/109/wine |
| Wholesale customers | 440 clients | 6 catégories de dépenses annuelles | Horeca, Retail | https://archive.ics.uci.edu/dataset/292/wholesale+customers |

## Contenu

- `PCA_On_Raisin_Dataset.ipynb`, `PCA_On_Wine_Dataset.ipynb`, `PCA_On_Wholesale_Customers_Dataset.ipynb` : un notebook par dataset, avec les mêmes étapes.
  1. Exploration des données.
  2. PCA « from scratch » (standardisation, matrice de covariance, valeurs et vecteurs propres, parts de variance) puis PCA avec scikit-learn, cercle des corrélations et interprétation.
  3. Clustering : K-Means (méthode du coude, silhouette) et CAH (dendrogramme, silhouette), comparaison des deux modèles et des classes trouvées avec les vraies catégories.
  4. Sauvegarde des modèles utilisés par l'application.
- `app.py` : l'application Streamlit.
- `*_pca_model.joblib` : standardisation, PCA à 2 composantes et régression logistique (prédiction de la catégorie).
- `*_clustering.joblib` : centroïdes et noms des classes du meilleur modèle de clustering (K-Means).

## L'application

On choisit un dataset dans la barre latérale. Cinq onglets :

- **Prédire** : saisir les mesures d'une observation pour obtenir sa catégorie prédite et sa classe de clustering.
- **Prédiction multiple** : charger un fichier CSV ou Excel et télécharger les prédictions pour toutes ses lignes.
- **Analyse PCA** : variance expliquée, projection sur (PC1, PC2), corrélations variables / composantes.
- **Clustering** : classes trouvées, effectifs, comparaison avec les vraies catégories, profil moyen de chaque classe.
- **Données** : le dataset brut.

## Lancer l'application en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Les modèles `.joblib` sont produits par les notebooks. Après une modification d'un notebook, il faut le réexécuter en entier pour les mettre à jour.

## Déploiement

L'application est en ligne sur **https://exam.jdlc-mikemy.tech**.

Elle tourne dans un conteneur Docker (`Dockerfile`, `docker-compose.yml`) sur le réseau `mediconnect_default` du serveur. Le reverse proxy Caddy du serveur la sert en HTTPS grâce au fichier `exam.caddy`, copié dans `/root/MediConnect/docker/sites/`.

Mettre à jour après un `git push` :

```bash
cd /root/exam_ml
git pull
docker compose up -d --build
```

### Vidéo de présentation

Le fichier vidéo n'est pas dans le dépôt (trop lourd). Il se dépose directement sur le serveur, dans `/root/exam_ml/video/presentation.mp4` : un conteneur nginx (`exam-video`) le sert sur `https://exam.jdlc-mikemy.tech/video/presentation.mp4`, et l'entrée « 🎬 Vidéo de présentation » apparaît automatiquement dans la barre latérale de l'application dès que le fichier est présent.

```bash
scp presentation.mp4 jeanprod:/root/exam_ml/video/
```

# Exam ML : PCA et clustering sur 4 datasets UCI

Réduction de dimension (PCA) et clustering (K-Means, CAH) sur quatre jeux de données du UCI Machine Learning Repository, avec une application Streamlit pour explorer les résultats et faire des prédictions.

## Datasets

| Dataset | Observations | Variables | Catégories | Source |
|---|---|---|---|---|
| Wine | 178 vins | 13 mesures chimiques | 3 cultivars | https://archive.ics.uci.edu/dataset/109/wine |
| Wholesale customers | 440 clients | 6 catégories de dépenses annuelles | Horeca, Retail | https://archive.ics.uci.edu/dataset/292/wholesale+customers |
| Wheat Seeds | 199 grains de blé | 7 mesures géométriques | aucune dans le fichier (segmentation pure) | https://archive.ics.uci.edu/dataset/236/seeds |
| Clients télécom | 3 150 clients | 12 variables d'utilisation (appels, SMS, abonnement, valeur client) | aucune (segmentation pure) | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset |

## Contenu

- `PCA_On_Wine_Dataset.ipynb`, `PCA_On_Wholesale_Customers_Dataset.ipynb` : un notebook par dataset, avec les mêmes étapes.
  1. Exploration des données.
  2. PCA « from scratch » (standardisation, matrice de covariance, valeurs et vecteurs propres, parts de variance) puis PCA avec scikit-learn, cercle des corrélations et interprétation.
  3. Clustering : K-Means (méthode du coude, silhouette) et CAH (dendrogramme, silhouette), comparaison des deux modèles et des classes trouvées avec les vraies catégories.
  4. Sauvegarde des modèles utilisés par l'application.
- `Kmeans_CAH_On_Wheat_Seeds_Dataset.ipynb` : PCA à 2 composantes pour visualiser, K-Means (méthode du coude, silhouette) puis CAH (dendrogramme, silhouette) et comparaison des deux modèles (silhouettes, tableau croisé des partitions). Le K-Means, meilleur modèle, sépare les grains grands, moyens et petits ; il est déployé dans l'application.
- `Customer_Segmentation.ipynb` : segmentation des clients télécom (normalisation, PCA à 2 composantes pour visualiser, K-Means et CAH, comparaison des silhouettes, profil des classes). Le meilleur modèle, la CAH, sépare les clients « fidèles » des clients « non fidèles » ; il est déployé dans l'application Streamlit (il l'était auparavant avec Gradio).
- `app.py` : point d'entrée de l'application Streamlit, avec deux pages : `page_analyse.py` (les datasets) et `page_video.py` (la vidéo de présentation, à l'adresse `/video`).
- `*_pca_model.joblib` : standardisation, PCA à 2 composantes et régression logistique (prédiction de la catégorie).
- `*_clustering.joblib` : centroïdes et noms des classes du meilleur modèle de clustering (K-Means pour Wine, Wholesale et Wheat Seeds, CAH pour les clients télécom).

## L'application

On choisit un dataset dans la barre latérale. Cinq onglets (quatre pour Wheat Seeds et les clients télécom : sans catégorie connue, il n'y a ni catégorie prédite ni onglet Analyse PCA) :

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

Le fichier vidéo n'est pas dans le dépôt (trop lourd). Il se dépose directement sur le serveur, dans `/root/exam_ml/video/presentation.mp4` : un conteneur nginx (`exam-video`) le sert sur `https://exam.jdlc-mikemy.tech/video/presentation.mp4`, et elle est présentée sur sa propre page, **https://exam.jdlc-mikemy.tech/video**, accessible aussi par le lien « Vidéo de présentation » de la barre latérale.

```bash
scp presentation.mp4 jeanprod:/root/exam_ml/video/
```

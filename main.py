import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write('''
# App simple pour la prévision des fleurs Iris
''')

# Charger le jeu de données iris
iris = datasets.load_iris()
X = iris.data  # Les données (features)
Y = iris.target  # Les variables cibles

# Configurer le classificateur de forêt aléatoire
clf = RandomForestClassifier()

# Ajuster le modèle aux données
clf.fit(X, Y)

# Créer l'application Streamlit
st.title("Classification d'iris avec Random Forest, Cette application prédit la catégorie des fleurs d'Iris")
st.header("Entrez les mesures de l'iris pour obtenir la prédiction")

# Ajouter les champs de saisie pour les caractéristiques de l'iris
sepal_length = st.slider("Longueur des sépales", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Largeur des sépales", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Longueur des pétales", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Largeur des pétales", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))


# Définir le bouton de prédiction
prediction_button = st.button("Prédire le type de fleur d'iris")

# Effectuer la prédiction lorsque le bouton est cliqué
if prediction_button:
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = clf.predict(input_data)
    target_names = iris.target_names
    predicted_class = target_names[prediction[0]]
    st.write(f"Type de fleur d'iris prédit : {predicted_class}")


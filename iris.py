import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Charger la base de données Iris
iris = datasets.load_iris()
X = iris.data  # Caractéristiques
y = iris.target  # Labels
class_names = iris.target_names

# Titre de l'application
st.title("Prédiction du type de fleur Iris")

# Instructions
st.write("""
Cette application prédit le type de fleur Iris en fonction des caractéristiques fournies.
""")

# Création des champs de saisie pour les caractéristiques
sepal_length = st.slider('Longueur du sépale', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider('Largeur du sépale', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider('Longueur du pétale', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider('Largeur du pétale', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Entrées utilisateur sous forme de tableau
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn.fit(X_train, y_train)

# Prédiction du type de fleur sur la base des caractéristiques entrées par l'utilisateur
if st.button('Prédire le type de fleur'):
    prediction = knn.predict(input_data)
    predicted_class = class_names[prediction][0]
    st.write(f"Le type de fleur prédit est : **{predicted_class}**")

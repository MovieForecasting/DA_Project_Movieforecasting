# 🎬 Movie Forecasting - Prédiction du Succès des Films 🚀  


## 📌 Introduction  
Bienvenue dans **Movie Forecasting**, un projet de Data Science visant à prédire le succès financier d’un film avant sa sortie en utilisant un modèle de Machine Learning entraîné sur les données de *The Movie Database (TMDB)*.  

Grâce à une analyse approfondie des tendances cinématographiques, nous avons développé un modèle permettant d’anticiper les recettes d’un film en fonction de plusieurs facteurs clés tels que :  
✅ Le budget 🎭  
✅ Le réalisateur 🎬  
✅ Les acteurs principaux 🎭  
✅ Le genre 🎞️  
✅ La période de sortie 📆  
✅ La popularité TMDB 📊  

---  

## ⚙️ Technologie et Stack  
Ce projet repose sur les technologies suivantes :  
- **Python** 🐍  
- **Pandas, NumPy, Matplotlib, Seaborn** pour l’analyse et la visualisation des données 📊  
- **Scikit-Learn** pour l’entraînement des modèles de Machine Learning 🤖  
- **Streamlit** pour la création d’une application interactive 🖥️  
- **Plotly** pour des graphiques dynamiques 🔥  

---

## 🎯 Objectifs et Méthodologie  

### 1️⃣ **Exploration des Données**  
- Analyse des tendances du cinéma  
- Étude de la popularité des films  
- Identification des variables influentes  

### 2️⃣ **Pré-Processing des Données**  
- Nettoyage et transformation des données  
- Gestion des valeurs manquantes  
- Encodage des variables catégorielles  
- Création de nouvelles features (ex: *actors_budget_interaction*)  

### 3️⃣ **Modélisation et Machine Learning**  
- Entraînement avec **Random Forest Regressor**  
- Optimisation via **GridSearchCV**  
- Évaluation du modèle avec **MSE et R²**  

### 4️⃣ **Déploiement de l’Application**  
- Interface interactive via **Streamlit**  
- Saisie manuelle des caractéristiques d’un film  
- Prédiction instantanée des recettes 📈  

---

## 🚀 Résultats  
🔹 **Score sur le jeu d'entraînement** : 0.9239  
🔹 **Score sur le jeu de test** : 0.6920  
🔹 **Amélioration de la robustesse grâce à GridSearchCV**  
🔹 **Détection des facteurs influents :** *Budget, Réalisateur, Popularité*  

---

## 🖥️ Comment Exécuter le Projet ?  
### 1️⃣ **Installation des dépendances**
```bash
pip install -r requirements.txt

streamlit run streamlit_project.py

📸 Aperçu de l’Application

🤝 Contributeurs
	•	Tristan Tansu - https://www.linkedin.com/in/tristan-tansu-42009365/
	•	Camille Laluque - https://www.linkedin.com/in/camille-cadet-51629b140/
	•	Samy Cao - https://www.linkedin.com/in/samy-cao
	•	Jean-Noël Duchevet - https://www.linkedin.com/in/jean-noel-duchevet/

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import r2_score

st.set_page_config(page_title="Prévision du succès d'un film", page_icon="🎥")

df_exploration = pd.read_csv("df_github.csv")

buffer = StringIO()
df_exploration.info(buf=buffer)
s = buffer.getvalue()

# Transformation du dataframe pour Exploration du Dataset
df_exploration['release_date'] = pd.to_datetime(df_exploration['release_date'], errors='coerce')
df_until_2023 = df_exploration[df_exploration['release_date'].dt.year <= 2023].copy()
df_until_2023_sorted = df_until_2023.sort_values(by='popularity', ascending=False)

st.title("Prévision du succès d'un film")

image_path = "logo_datascientest.png"
st.sidebar.image(image_path, width=180)

st.sidebar.title("Sommaire")

pages=["Présentation du projet 🚀", "Exploration du Dataset 🧐", "DataViz' 📊", "Pré-processing 👨‍💻", "Modélisation / Machine Learning ⚙️", "Conclusion 🎬"]

page=st.sidebar.radio("Aller vers", pages)

st.sidebar.write("__Auteurs__")

st.sidebar.write("[Camille Laluque](https://www.linkedin.com/in/camille-cadet-51629b140/)")
st.sidebar.markdown("[Samy Cao](https://www.linkedin.com/in/samy-cao)")
st.sidebar.write("[Jean-Noël Duchevet](https://www.linkedin.com/in/jean-noel-duchevet/)")
st.sidebar.write("[Tristan Tansu](https://www.linkedin.com/in/tristan-tansu-42009365/)")

st.sidebar.write("Promotion Data Analyst : Janvier 2025")

if page == pages[0]:
    image_path = "image_sommaire.png"
    st.image(image_path, width=700)
    st.write("### Présentation du projet")
    st.write("Ce projet a été réalisé dans le cadre de notre formation en Data Analyse via l'organisme Data Scientest.")
    st.write("L'objectif de ce projet est de prédire le succès d'un film au box-office en utilisant le jeu de donnée issu de TMDB : 'The Movie DataBase' où nous pouvons obtenir des informations comme le budget, les recettes, le genre, les acteurs et les réalisateurs.")
    st.write("Grâce à l’analyse de ces données, nous pouvons mieux comprendre lesquelles ont une influence dans le succès d'un film.")
  
    st.write("### Problématique")
  
    st.write("Quels sont les éléments clés qui influencent le succès d’un film ? Peut-on prédire ce succès à partir des données disponibles ?")

elif page == pages[1]:
    
    st.write("### Exploration du Dataset")
    image_path = "image_exploration.png"
    st.image(image_path, width=700)

    st.write("Ci-dessous un aperçu du dataset :")
    st.dataframe(df_exploration.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df_exploration.shape)

    st.write("Autres informations sur le dataframe :")

    with st.expander("Informations sur le dataset"):
        st.text(s)

    if st.checkbox("Montrer les valeurs manquantes"): 
        st.dataframe(df_exploration.isna().sum())
    
    if st.checkbox("Montrer les doublons") : 
        st.write(df_exploration.duplicated().sum())

elif page == pages[2]:

    st.write("Nous allons ensuite présenter divers graphiques exploitant nos jeux de données.")

    st.write("### Méthodologie")

    st.write("#### Variable 'Popularity'")

    st.write("Nous nous intéressons à l'indicateur 'popularity' de TMDB (The Movie Database) qui sera la variable d'analyse exploratoire car il reflète la popularité d'un film ou d'une série selon plusieurs critères. L'algorithme de calcul de cet indicateur n'est pas public mais nous savons qu'il est basé sur plusieurs facteurs :")
    st.write("""
    - Les vues des pages
    - Les votes des utilisateurs
    - Le nombre d'ajout en 'favoris' et/ou en 'watchlist'
    - Le nombre de recherches sur la plateforme
    - Les mentions sur les réseaux
    - Les années de lancement
    - Le 'popularity score' du jour précédent
             """)
    st.write("L'indicateur de popularité reflète un caractère dynamique. Il peut fluctuer en fonction des tendances actuelles, des sorties récentes, ou de l'impact viral sur les plateformes sociales grâce à l'interactions des utilisateurs. Plus un film ou une série est mentionné et/ou recherché, plus sa popularité sera élevée sur TMDB.")

    st.markdown("[Source TMDB](https://developer.themoviedb.org/docs/popularity-and-trending)")
    st.write("Notre première approche sur cet indicateur a été de montrer son évolution moyenne au fil des années c'est à dire de 1980 à 2025 soit sur l'ensemble des données de notre dataset.")

    st.write("Rapidement nous nous sommes rendu compte que l'année 2024 présente des valeurs extrêmes de popularité. En effet, comme expliqué précédement cet indicateur présente un caractère dynamique du fait qu'il est mis à jour régulièrement en fonction des paramètres qui le composent.")

    st.write("##### Avis 'métier'")

    st.write("A titre d'exemple, nous pouvons penser que les films sorti en 2024 dans notre dataset dont certains qui l'ont été récemment notamment sur le mois de décembre, ont un score élevé de popularité car ils suscitent beaucoup d'intérêt et de curiostité de la part des utilisateurs. Cela revêt donc une importance dans l'analyse de données temporelles.")
    st.write("Afin de limiter l'impact de ces valeurs sur nos analyses, nous décidons d'exclure les données relatives à 2024.")

    st.write("##### Conclusion & Exploitation")

    st.write("Ce travail d'exploration va s'articuler autour de la variable 'popularity' notamment sa saisonalité et son interaction avec différentes variables telles que le genre, le budget, les réalisateurs et les acteurs.")
    
    image_path = "boxplot_popularity2024.png"
    st.image(image_path, width=600)

    # Ajout d'onglets
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["DataViz' 1", "DataViz' 2", "DataViz' 3","DataViz' 4","DataViz' 5","DataViz' 6","DataViz' 7"])

    # Ajouter du contenu dans chaque onglet
    with tab1:
        st.title("Evolution de la popularité des films au fil des années")

        avg_popularity = df_until_2023_sorted.groupby('release_year')['popularity'].mean()
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(avg_popularity.index, avg_popularity.values)
        ax.set_title("Evolution de la popularité moyenne des films au fil des années");
        ax.set_xlabel("Année")
        ax.set_ylabel("Popularité")
        st.pyplot(fig2)
        
        st.write("##### Analyse du graphique")
        st.write("""
        Le graphique montre l’évolution de la popularité moyenne des films en fonction de leur année de sortie. Plusieurs tendances émergent :
        - **Tendance générale à la hausse :** On observe une augmentation progressive de la popularité des films au fil des décennies. Cette tendance peut être liée à une amélioration des techniques de production, à une meilleure accessibilité aux films et à une augmentation du nombre de spectateurs.
        - **Pics et baisses de popularité :** Certaines années affichent des pics, ce qui pourrait être dû à la sortie de films marquants qui ont dominé le box-office et influencé la tendance générale. La baisse notable en 2020 coïncide avec la crise sanitaire mondiale, qui a entraîné une diminution du nombre de films en salle et une réduction du nombre de spectateurs.
        """)

        st.write("##### Avis 'métier'")
        st.write("Cette analyse est pertinente pour comprendre l’évolution des attentes du public et l’impact des grandes tendances cinématographiques.")
        st.write("L'augmentation générale de la popularité peut être attribuée à l'évolution du marketing des films, à l'essor des grandes franchises, ainsi qu'à une meilleure structuration des sorties en salle.")
        st.write("Les baisses de certaines périodes peuvent être liées à des crises économiques, à des changements dans l’industrie cinématographique ou à une concurrence accrue entre les films.")
        st.write("L’étude de ces variations permet aux producteurs et distributeurs de mieux anticiper le marché et d’adapter leurs stratégies de sortie.")

        st.write("##### Conclusion & Exploitation")
        st.write("Si l’année de sortie influence la popularité, elle pourrait être intégrée comme une variable clé dans notre modèle de prédiction du succès au box-office.. Après lecture du graphique, on remarque une stabilisation à partir de 1995, il nous semble utile de réduite notre jeu de données de films à partir cette année.")

    with tab2:
        st.title("Popularité moyenne des films en fonction du mois de sortie")

        if 'release_month' not in df_until_2023_sorted.columns:
            df_until_2023_sorted['release_month'] = df_until_2023_sorted['release_date'].dt.month
        
        pop_by_month = df_until_2023_sorted.groupby('release_month')['popularity'].mean()

        fig3, ax = plt.subplots(figsize=(10,6))
        ax.plot(pop_by_month.index, pop_by_month.values, marker='o', linestyle='-', color='orange')
        ax.set_title("Popularité moyenne des films par mois de sortie");
        ax.set_xlabel("Mois")
        ax.set_ylabel("Popularité moyenne")

        # Mettre en avant les mois clés
        highlight_months = [7, 12]
        for month in highlight_months:
            if month in pop_by_month.index:
                ax.scatter(month, pop_by_month[month], color="red", s=100, zorder=3)
        
        st.pyplot(fig3)

        st.write("##### Analyse du graphique")
        st.write("""
        - Janvier : Faible popularité (1.5), ce qui pourrait s’expliquer par un creux post-fêtes.
        - Février - Mars : Augmentation notable, peut-être liée aux films sortis autour des Oscars et des vacances d’hiver.
        - Juin - Août : Hausse en été, ce qui correspond aux blockbusters estivaux.
        - Septembre - Novembre : Légère baisse, souvent une période de transition avec moins de grosses sorties.
        - Décembre : Pic de popularité qui remonte, sans doute grâce aux sorties de Noël et aux films familiaux/festifs.
        """)

        st.write("##### Avis 'métier'")
        st.write("Pour maximiser le succès au box-office, il est préférable de programmer la sortie d’un film durant l’été (juin-août) ou les fêtes de fin d’année (décembre), périodes où les films rencontrent le plus de popularité.")
        st.write("À l’inverse, une sortie en janvier ou au début du printemps (mars-avril) pourrait être plus risquée en termes d’audience.")

        st.write("##### Conclusion & Exploitation")
        st.write("Si la popularité d'un film est fortement influencé par son mois de sortie, il pourrait être intéressant d'intégrer cette variable dans notre modèle prédictif.")

    with tab3:
        st.title("Popularité moyenne par genre")

        st.write("##### Analyse du graphique")

        st.write("##### Avis 'métier'")

        st.write("##### Conclusion & Exploitation")
        
    with tab4:
        st.title("Distribution de la popularité par catégorie de budget")

        st.write("##### Analyse du graphique")

        st.write("##### Avis 'métier'")

        st.write("##### Conclusion & Exploitation")

    with tab5:
        st.title("Distribution des langues originales par popularité moyenne")

        st.write("##### Analyse du graphique")

        st.write("##### Avis 'métier'")

        st.write("##### Conclusion & Exploitation")

    with tab6:
        st.title("Distribution des acteurs par popularité et par weighted rating")

        st.write("##### Analyse du graphique")

        st.write("##### Avis 'métier'")

        st.write("##### Conclusion & Exploitation")

    with tab7:
        st.title("Distribution des réalisateurs par popularité et par weighted rating")

        st.write("##### Analyse du graphique")

        st.write("##### Avis 'métier'")

        st.write("##### Conclusion & Exploitation")

elif page == pages[3]:
    st.write("### Pré-processing")

# A COMPLETER

elif page == pages[4]:
    st.write("### Modélisation et Machine Learning")

# A COMPLETER

elif page == pages[5]:
    st.write("### Conclusion")

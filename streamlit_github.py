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
import requests
import io
# Base URL for GitHub raw files
github_base_url = "https://raw.githubusercontent.com/MovieForecasting/DA_Project_Movieforecasting/main/"
from sklearn.metrics import r2_score

st.set_page_config(page_title="Prévision du succès d'un film", page_icon="🎥")
st.markdown(
    """
    <style>
    /* Fond statique très clair */
    [data-testid="stAppViewContainer"] {
    background-color: #f7f7f7 !important;
    }
    
    /* Import de la police Orbitron pour un rendu futuriste */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .stApp {
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Barre latérale (bannière de côté) */
    [data-testid="stSidebar"] {
        background: #eef3f7 !important;
        color: #333333 !important;
        border-right: 1px solid #d0d7de;
    }
    
    /* Header : dégradé inspiré de la couleur de la sidebar */
    header {
        background: linear-gradient(180deg, #eef3f7, #dce4eb) !important;
        color: #333333 !important;
        border-bottom: 1px solid #d0d7de;
    }
    
    /* Zone principale */
    .main .block-container {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem auto;
        max-width: 1200px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Boutons : style moderne en utilisant la couleur de la sidebar comme référence */
    div.stButton > button, div.stForm > button {
        transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
        background: linear-gradient(180deg, #eef3f7, #dce4eb);
        color: #333333;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        font-weight: 600;
    }
    
    div.stButton > button:hover, div.stForm > button:hover {
        transform: scale(1.05);
        background: linear-gradient(180deg, #dce4eb, #bcc4cd);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        color: #333333;
    }
    
    /* Style pour les liens */
    a {
        color: #0077b6;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df_exploration = pd.read_csv(github_base_url + "df_github.csv")

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

import streamlit.components.v1 as components
music_url = github_base_url + "Time.mp3"
components.html(
    f"""
    <audio src="{music_url}" autoplay loop muted controls>
      Your browser does not support the audio element.
    </audio>
    """,
    height=0
)

pages=["Présentation du projet 🚀", "Exploration du Dataset 🧐", "DataViz' 📊", "Pré-processing 👨‍💻", "Modélisation / Machine Learning ⚙️", "Application 🎥", "Conclusion 🎬"]

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
        missing_df = df_exploration.isna().sum().reset_index()
        missing_df.columns = ["Colonne", "Nombre de NaN"]
        missing_df["Pourcentage (%)"] = (missing_df["Nombre de NaN"] / len(df_exploration)) * 100
        missing_df = missing_df.sort_values(by="Pourcentage (%)", ascending=False)
        st.table(missing_df)
    
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

    with st.expander("Création de la variable 'Weighted Rating"):
        st.write("#### Weighted Rating : Qu’est-ce que c’est ?")
        st.write("Le weighted rating (ou note pondérée) est une mesure qui vise à mieux refléter la qualité d’un film qu’une simple note moyenne. En effet, si un film n’a que 2 votes à 10/10, il ne doit pas dépasser un blockbuster ayant 10 000 votes à 8/10. Le weighted rating corrige ce problème en tenant compte à la fois de la note moyenne du film et de la note moyenne globale du catalogue, pondérées par le nombre de votes du film.")
        st.write("##### Analyse de la formule de calcul du Weighted Rating")
        st.code("""
        *C = df['vote_average'].mean()
        *C : la moyenne globale (vote_average) de tous les films.
        *m = df['vote_count'].quantile(0.90)
        *m : un seuil de votes, ici le 90ᵉ percentile de vote_count.
        
        Autrement dit, si le film dépasse ce seuil, il aura plus de poids dans son évaluation.
        
        Cette formule s’inspire de l’algorithme d’IMDb :

        Weighted Rating = (v / (v+m)) * R + (m / (v+m)) * C
        
        où :
        *v = nombre de votes pour ce film,
        *R = note moyenne du film (vote_average),
        *C = note moyenne globale,
        *m = seuil minimum de votes.
        """)

        st.write("##### Avis 'métier'")
        st.write("Une fois qu'un film possède un weighted rating, on peut l'associer :")
        st.write("""
        - aux acteurs (actrices) ayant participé à ce film
        - aux réalisteurs (réalisatrices)
        """)
        st.write("**Filtrage** : On peut imposer des seuils de popularité et de weighted rating pour retenir uniquement les acteurs/réalisateurs associés à des films “validés” par un consensus suffisant.")
        st.write("**Prédiction pré-sortie*** : Lorsque vous prévoyez le succès potentiel d’un futur film, savoir qu’un acteur ou réalisateur a souvent travaillé sur des films avec un fort weighted rating peut être un indicateur de qualité.")

        st.write("##### Conclusion & Exploitation")
        st.write("Le weighted rating permet de compenser la variable popularity en ajoutant une dimension de fiabilité dans l’appréciation. Si la popularité est la “température” instantanée de l’intérêt du public, le weighted rating est un “thermostat” plus stable, basé sur la quantité et la qualité des votes.")
        st.write("Cela nous permet d’obtenir une évaluation plus représentative et d’éviter que des films avec peu de votes ne faussent les résultats.")
    
    # Création de la variable "weighted_rating"
    C = df_until_2023_sorted['vote_average'].mean()
    m = df_until_2023_sorted['vote_count'].quantile(0.90)

    def weighted_rating(row, m=m, C=C):
        v = row['vote_count']
        R = row['vote_average']
        if (v+m) == 0:
            return R
        return (v/(v+m))*R + (m/(v+m))*C

    df_until_2023_sorted['weighted_rating'] = df_until_2023_sorted.apply(weighted_rating, axis=1)
    
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

        # Nettoyage de la variable 'Genres_clean'
        df_filtered = df_until_2023_sorted
        df_filtered['Genres_clean'] = df_filtered['Genres_clean'].fillna('')

        df_filtered['Genres_clean'] = df_filtered['Genres_clean'].apply(lambda x: [genre.strip() for genre in x] if isinstance(x, list) else [genre.strip() for genre in x.split(',') if genre.strip() != ''])

        df_filtered_exploded = df_filtered.explode('Genres_clean')
        df_filtered_exploded = df_filtered_exploded[df_filtered_exploded['Genres_clean'] != '']

        # Calculer la popularité moyenne par genre
        genre_popularity_filtered = df_filtered_exploded.groupby('Genres_clean')['popularity'].mean().sort_values(ascending=True)

        fig4, ax = plt.subplots(figsize=(12,8))
        ax.barh(genre_popularity_filtered.index, genre_popularity_filtered.values, color=plt.cm.YlOrRd_r(range(len(genre_popularity_filtered))))
        ax.set_xlabel("Popularité moyenne")
        ax.set_ylabel("Genre")
        ax.set_title("Popularité moyenne par genre (données filtrées)")
        st.pyplot(fig4)

        st.write("##### Analyse du graphique")
        st.write("Ce graphique met en évidence une forte popularité des films d'aventure, suivis de près par l'action et la science-fiction. À l'inverse, les documentaires et les films musicaux enregistrent une popularité bien plus faible. L'écart entre les films les plus et les moins populaires est significatif, soulignant une nette préférence du public pour les films à grand spectacle.")

        st.write("##### Avis 'métier'")
        st.write("Pour maximiser la popularité d’un film, il est recommandé d’opter pour un genre plébiscité comme l’aventure, l’action ou la science-fiction. Ces genres attirent un large public grâce à leurs effets spéciaux, leurs intrigues dynamiques et leur accessibilité internationale.")
        st.write("Cependant, un film appartenant à un genre moins populaire (romance, documentaire, musique, etc.) peut tout de même rencontrer le succès en misant sur un casting fort et une réalisation de qualité, qui peuvent compenser un intérêt initial plus faible.")

        st.write("##### Conclusion & Exploitation")
        st.write("Le genre d’un film a un impact significatif sur sa popularité et doit être pris en compte dans la prédiction du succès au box-office. Cette variable pourra être intégrée au modèle prédictif pour affiner les estimations en fonction des préférences du public.")
        
    with tab4:
        st.title("Distribution de la popularité par catégorie de budget")

        # Distribution de la popularité par catégorie de budget
        df_filtered = df_until_2023_sorted.copy()
        df_filtered = df_filtered[df_filtered['popularity'] <= 3000]

        bins = [0, 1e6, 10e6, 100e6, np.inf]
        labels = ["0-1M", "1M-10M", "10M-100M", "100M+"]
        df_filtered["budget_category"] = pd.cut(df_filtered["Budget"], bins=bins, labels=labels)

        fig5, ax = plt.subplots(figsize=(12,8))
        sns.stripplot(x="budget_category", y="popularity", data=df_filtered,
                      jitter=True, size=3, alpha=1, palette="YlOrRd")
        ax.set_xlabel("Catégorie de budget")
        ax.set_ylabel("Popularité")
        ax.set_title("Distribution de la popularité par catégorie de budget (jusqu'à 2023) - Strip Plot")
        st.pyplot(fig5)

        st.write("##### Analyse du graphique")
        st.write("Le graphique montre que les films avec un budget inférieur à 10M ont généralement une popularité plus faible. À l’inverse, les films bénéficiant d’un budget supérieur à 10M présentent une plus grande dispersion de popularité, avec plusieurs d’entre eux atteignant des niveaux très élevés.")

        st.write("##### Avis 'métier'")
        st.write("Le budget d’un film ne détermine pas à lui seul son succès. On observe que des films très populaires existent dans toutes les catégories de budget. Cependant, un budget plus élevé semble augmenter les chances qu’un film atteigne une popularité importante, probablement en raison d’un meilleur marketing, d’un casting attractif et de meilleures ressources techniques.")

        st.write("##### Conclusion & Exploitation")
        st.write("Bien que le budget ne soit pas le facteur principal de la popularité d’un film, il peut jouer un rôle utile dans son succès. Il serait donc pertinent de l’inclure comme variable dans le modèle prédictif, mais en le combinant avec d’autres facteurs comme le genre, le casting et la date de sortie.")

    with tab5:
        st.title("Distribution des langues originales par popularité moyenne")

        # Distribution des langues originales par popularité moyenne
        df_filtered = df_until_2023_sorted.copy()
        df_filtered = df_filtered[df_filtered['original_language'].notnull()]
        language_popularity = df_filtered.groupby('original_language')['popularity'].mean().sort_values(ascending=False).head(10)
        lang_mapping = {'en': 'English','fr': 'French','es': 'Spanish','it': 'Italian','de': 'German','ko': 'Korean','ja': 'Japanese','hi': 'Hindi','pt': 'Portuguese','ru': 'Russian','tn': 'Tswana','su': 'Sundanese','no': 'Norwegian','lv': 'Latvian','cn': 'Chinese','te': 'Telugu','dz': 'Dzongkha','yo': 'Yoruba','da': 'Danish'}
        language_popularity.index = language_popularity.index.map(lambda x: lang_mapping.get(x, x))

        # Tracer le graphique
        fig6, ax = plt.subplots(figsize=(12,8))
        ax.barh(y=language_popularity.index, width=language_popularity.values, color=plt.cm.YlOrRd_r(range(len(language_popularity))))
        ax.set_xlabel("Popularité moyenne")
        ax.set_ylabel("Langue originale")
        ax.set_title("Top 10 des langues originales par popularité moyenne")
        st.pyplot(fig6)

        st.write("##### Analyse du graphique")
        st.write("On observe un pic de populartité pour les films en tswana avec une note moyenne de 8.5. Cependant, les autres langues tournent autour de 3 à 5.5 de moyenne ce qui reste relativement homogène.")

        st.write("##### Avis 'métier'")
        st.write("La langue d’un film pourrait influencer sa popularité, mais en dehors du cas particulier du tswana, aucune tendance marquée ne se dégage. Cette anomalie peut être due à un faible nombre de films dans cette langue, ce qui fausse la moyenne.")

        st.write("##### Conclusion & Exploitation")
        st.write("La langue originale du film pourrait être ajoutée au modèle de prédiction, mais son impact semble limité. Elle pourrait être utile en complément d’autres variables comme le genre ou le pays de distribution.")

    with tab6:
        st.title("Distribution des acteurs par popularité et par weighted rating")

        # Nettoyage de la variable "Actors"
        df_actors = df_until_2023_sorted.copy()
        df_actors['Actors'] = df_actors['Actors'].fillna('')
        df_actors['Actors'] = df_actors['Actors'].apply(lambda x: x if isinstance(x, list) else [actor.strip() for actor in x.split(',') if actor.strip() != ''])

        df_actors_exploded = df_actors.explode('Actors')
        df_actors_exploded = df_actors_exploded[df_actors_exploded['Actors'] != "Unknown"]
        
        actor_popularity = df_actors_exploded.groupby('Actors')['popularity'].mean()

        # Calculer la popularité moyenne par acteurs
        actor_weighted_rating = df_actors_exploded.groupby('Actors')['weighted_rating'].mean()
        actor_counts = df_actors_exploded['Actors'].value_counts()
        actor_stats = pd.DataFrame({'avg_popularity': actor_popularity,'avg_weighted_rating': actor_weighted_rating,'film_count': actor_counts})

        # Tracer le graphique
        def plot_actors(min_popularity, min_weighted_rating):
        # Filtre les acteurs selon min_popularity et min_weighted_rating, calcule un score combiné (avg_popularity + avg_weighted_rating), trie et affiche un barplot du top 100."""
            global actor_stats
            actor_stats_filtered = actor_stats[(actor_stats['avg_popularity'] >= min_popularity) & (actor_stats['avg_weighted_rating'] >= min_weighted_rating)].copy()
            actor_stats_filtered['combined_score'] = (actor_stats_filtered['avg_popularity'] + actor_stats_filtered['avg_weighted_rating'])
            actor_stats_sorted = actor_stats_filtered.sort_values(by='combined_score', ascending=False)

            top100_actors = actor_stats_sorted.head(100)

            fig7, ax = plt.subplots(figsize=(12,20))
            ax.barh(y=top100_actors.index, width=top100_actors['combined_score'], color=plt.cm.YlOrRd_r(range(len(top100_actors))))
            ax.set_xlabel("Score combiné (Popularité + Weighted Rating)", fontsize=14)
            ax.set_ylabel("Acteur", fontsize=14)
            ax.set_title(f"Top 100 Acteurs (pop ≥ {min_popularity}, WR ≥ {min_weighted_rating})", fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            st.pyplot(fig7)

        st.write("##### Analyse du graphique")
        st.write("Ce graphique présente les 100 acteurs ayant une popularité ≥ 1.0 et un Weighted Rating (WR) ≥ 7.0.")
        st.write("Peter Crombie se démarque nettement en tête du classement.")
        st.write("Les premiers acteurs affichent une légère décroissance en popularité avant une stabilisation progressive. On observe une répartition assez homogène parmi le reste des acteurs, avec une baisse progressive des scores.")

        st.write("##### Avis 'métier'")
        st.write("L’analyse de la distribution des acteurs en fonction de leur popularité et de leur weighted rating permet d’identifier des tendances intéressantes.")
        st.write("Certains noms sont peu connus du grand public, ce qui pourrait indiquer un biais lié aux films récents ou à des productions spécifiques ayant bénéficié d’un bon accueil critique.")
        st.write("Il est possible que certains acteurs figurent en tête du classement en raison de films récents ayant eu une forte exposition médiatique.")

        st.write("##### Conclusion & Exploitation")
        st.write("L'intégration des acteurs dans le modèle de prédiction peut être pertinente.")
        st.write("La popularité d’un film peut être fortement influencée par la présence de certains acteurs en tête d'affiche.")
        st.write("Leur score combiné (Popularité + Weighted Rating) pourrait apporter une valeur ajoutée, notamment pour les films à fort budget misant sur des têtes d’affiche.")
        plot_actors(1.0, 7.0)

    with tab7:
        st.title("Distribution des réalisateurs par popularité et par weighted rating")

        # Nettoyage de la variable "Director"
        df_directors = df_until_2023_sorted.copy()
        df_directors['Director'] = df_directors['Director'].fillna('')
        df_directors['Director'] = df_directors['Director'].apply(lambda x: x if isinstance(x, list) else [dir.strip() for dir in x.split(',') if dir.strip() != ''])

        df_directors_exploded = df_directors.explode('Director')
        df_directors_exploded = df_directors_exploded[df_directors_exploded['Director'] != "Unknown"]
        director_popularity = df_directors_exploded.groupby('Director')['popularity'].mean()

        # Calculer la popularité moyenne par directors
        director_weighted_rating = df_directors_exploded.groupby('Director')['weighted_rating'].mean()
        director_counts = df_directors_exploded['Director'].value_counts()
        director_stats = pd.DataFrame({'avg_popularity': director_popularity,'avg_weighted_rating': director_weighted_rating,'film_count': director_counts})

        # Tracer le graphique
        def plot_directors(min_popularity, min_weighted_rating):
        # Filtre les réalisateurs selon min_popularity et min_weighted_rating, calcule le combined_score (pop + weighted_rating), trie et affiche un barplot du top 100.
            global director_stats
            director_stats_filtered = director_stats[(director_stats['avg_popularity'] >= min_popularity) & (director_stats['avg_weighted_rating'] >= min_weighted_rating)].copy()
            director_stats_filtered['combined_score'] = (director_stats_filtered['avg_popularity'] + director_stats_filtered['avg_weighted_rating'])

            director_stats_sorted = director_stats_filtered.sort_values(by='combined_score', ascending=False)
            top100_directors = director_stats_sorted.head(100)

            fig8, ax = plt.subplots(figsize=(20,20))
            ax.barh(y=top100_directors.index, width=top100_directors['combined_score'], color=plt.cm.YlOrRd_r(range(len(top100_directors))))
            ax.set_xlabel("Score combiné (Popularité + Weighted Rating)", fontsize=14)
            ax.set_ylabel("Réalisateur", fontsize=14)
            ax.set_title(f"Top 100 Réalisateurs (pop ≥ {min_popularity}, WR ≥ {min_weighted_rating})", fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            st.pyplot(fig8)

        st.write("##### Analyse du graphique")
        st.write("Le graphique met en évidence les réalisateurs les mieux classés en fonction d’un score combiné, intégrant popularité et note moyenne pondérée (WR).")
        st.write("Domination des réalisateurs d’animation : Jared Bush, Rodney Rothman, Bob Persichetti et Josh Cooley arrivent en tête. L’animation est un genre rentable, souvent associé à des films bien notés sur des plateformes comme IMDb.")
        st.write("Absence des grands noms du cinéma : Spielberg, Nolan ou Tarantino ne figurent pas dans le classement. Cela suggère que l’algorithme favorise les films récents ayant un fort impact immédiat plutôt que les réalisateurs au long palmarès.")
        
        st.write("##### Avis 'métier'")
        st.write("Ce classement met en avant une nouvelle génération de réalisateurs, particulièrement dans l’animation. Il souligne aussi le fait que des films bien notés mais moins populaires peuvent surpasser ceux de réalisateurs emblématiques, ce qui suggère que la popularité brute ne suffit pas à garantir une position élevée.")

        st.write("##### Conclusion & Exploitation")
        st.write("Le succès d’un film peut être fortement corrélé au réalisateur. Certains noms fonctionnent mieux en fonction du genre cinématographique et de la popularité associée. On peut donc l'intégrer dans le modèle de prédiction.")
        plot_directors(1.0, 7.0)

elif page == pages[3]:
    st.write("### Pré-processing")
    st.image("DeusExMachina.jpg", width=700)
    st.markdown(
        """
        <style>
        /* Animation pour les boutons sur la page 3 - mise à jour pour ressembler aux boutons de la page Modélisation / Machine Learning */
        div.stButton > button {
          transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
        }
        div.stButton > button:hover {
          transform: scale(1.05);
          background: linear-gradient(180deg, #dce4eb, #bcc4cd);
          box-shadow: 0 6px 12px rgba(0,0,0,0.2);
          color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Étape 1 : Chargement des données
    if st.button("Charger les données"):
        df = pd.read_csv("df_github.csv")
        st.write(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        st.write(df.head())
        st.session_state.df = df

    # Étape 2 : Conversion en datetime et filtrage par année
    if st.button("Conversion en datetime et filtrage par année"):
        df = st.session_state.df if st.session_state.df is not None else pd.read_csv("df_github.csv")
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df[(df['release_date'].dt.year >= 1995) & (df['release_date'].dt.year <= 2023)]
        st.write("Dimensions après filtrage :", df.shape)
        st.session_state.df = df

    # Étape 3 : Vérification et suppression des valeurs manquantes
    if st.button("Vérifier et supprimer les NaN"):
        df = st.session_state.df
        # Calcul du taux de NaN pour chaque colonne
        nan_percent = (df.isna().sum() / len(df)) * 100
        st.write("### Taux de NaN (%) par colonne :")
        st.table(nan_percent.reset_index().rename(columns={'index': 'Colonne', 0: 'Pourcentage'}))

        # Option d'afficher un aperçu des lignes contenant des NaN
        if st.checkbox("Afficher un aperçu des lignes avec des NaN"):
            df_nan = df[df.isna().any(axis=1)]
            st.dataframe(df_nan.head())

        # Suppression des lignes avec 3 NaN ou plus et des lignes sans 'Recettes'
        df_clean = df.loc[(df.isna().sum(axis=1)) < 3].dropna(subset=['Recettes'])
        st.write("Dimensions après suppression des lignes problématiques :", df_clean.shape)
        st.write("Nombre de lignes supprimées :", df.shape[0] - df_clean.shape[0])
        st.session_state.df = df_clean

    # Étape 4 : Extraction des informations temporelles
    if st.button("Extraction des informations temporelles"):
        df = st.session_state.df
        df['year'] = df['release_date'].dt.year
        df['month'] = df['release_date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df.drop(["release_date", "month"], axis=1, inplace=True)
        st.write("Extraction des informations temporelles effectuée.")
        st.session_state.df = df

    # Étape 5 : Traitement des colonnes textuelles
    if st.button("Traitement des colonnes textuelles"):
        df = st.session_state.df
        for col in ['Director', 'Actors', 'Genres_clean']:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].apply(lambda x: [s.strip() for s in x.split(',')])
        df['Director'] = df['Director'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
        df['Actors'] = df['Actors'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
        df['Genres_clean'] = df['Genres_clean'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
        st.write("Traitement des colonnes textuelles effectué.")
        st.session_state.df = df

    # Étape 6 : Calcul du logarithme des Recettes
    if st.button("Calcul logarithme des Recettes"):
        df = st.session_state.df
        df['log_Recettes'] = np.log1p(df['Recettes'])
        st.write("Calcul logarithme des Recettes effectué.")
        st.session_state.df = df

    # Étape 7 : Calcul du weighted_rating
    if st.button("Calcul du weighted_rating"):
        df = st.session_state.df
        C = df['vote_average'].mean()
        m_val = df['vote_count'].quantile(0.90)
        def weighted_rating(row):
            v = row['vote_count']
            R = row['vote_average']
            return (v / (v + m_val)) * R + (m_val / (v + m_val)) * C if (v + m_val) != 0 else R
        df['weighted_rating'] = df.apply(weighted_rating, axis=1)
        df.drop(["vote_count", "vote_average"], axis=1, inplace=True)
        st.write("Calcul du weighted_rating effectué.")
        st.session_state.df = df

    # Étape 8 : Calcul des moyennes pondérées pour Director et Actors
    if st.button("Calcul des moyennes pondérées pour Director et Actors"):
        df = st.session_state.df
        df['director_weighted_avg'] = df.groupby('Director')['weighted_rating'].transform('mean')
        df['actors_weighted_avg'] = df.groupby('Actors')['weighted_rating'].transform('mean')
        st.write("Calcul des moyennes pondérées effectué.")
        st.session_state.df = df

    # Étape 9 : Gestion du budget
    if st.button("Gestion du budget"):
        df = st.session_state.df
        df.loc[df['Budget'] == 1.0, 'Budget'] = 0
        df['is_blockbuster'] = (df['Budget'] >= 50000000).astype(int)
        df['actors_budget_interaction'] = df['actors_weighted_avg'] * df['Budget']
        df['log_Budget'] = np.log1p(df['Budget'])
        st.write("Gestion du budget effectuée.")
        st.session_state.df = df

    # Étape 10 : Imputation des valeurs manquantes
    if st.button("Imputation des valeurs manquantes"):
        df = st.session_state.df
        df['Budget'] = df.groupby("Genres_clean")['Budget'].transform(lambda x: x.fillna(x.median()))
        df['Recettes'] = df.groupby("Genres_clean")['Recettes'].transform(lambda x: x.fillna(x.median()))
        df['Budget'] = df.groupby("weighted_rating")['Budget'].transform(lambda x: x.fillna(x.median()))
        df['Recettes'] = df.groupby("weighted_rating")['Recettes'].transform(lambda x: x.fillna(x.median()))
        st.write("Imputation des valeurs manquantes effectuée.")
        st.write("NaN Budget :", df['Budget'].isna().sum(), " - NaN Recettes :", df['Recettes'].isna().sum())
        st.session_state.df = df

    # Étape 11 : Afficher le dataframe final
    if st.button("Afficher le dataframe final"):
        df = st.session_state.df
        st.write("Dimensions finales :", df.shape)
        st.dataframe(df.head())

elif page == pages[4]:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import streamlit as st

    st.title("Modélisation et Machine Learning ⚙️")
    # Afficher la photo Arrival.jpg (assure-toi qu'elle est bien dans le même dossier)
    st.image("Arrival.jpg", width=700)

    st.write("Nous allons relancer toutes les étapes du pipeline : chargement, feature engineering, imputation, entraînement du modèle...")

    # Bouton pour lancer le pipeline
    if st.button("Relancer le pipeline complet"):
        # -----------------------------
        # 1. Chargement & Nettoyage
        # -----------------------------
        df = pd.read_csv("df_github.csv")
        st.write(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

        # Filtrage
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df[(df['release_date'].dt.year >= 1995) & (df['release_date'].dt.year <= 2023)]
        st.write("Dimensions après filtrage :", df.shape)

        # Suppression des lignes problématiques
        df = df.loc[(df.isna().sum(axis=1)) < 3]
        df = df.dropna(subset=['Recettes'])
        st.write("Dimensions après suppression des NaN :", df.shape)

        # -----------------------------
        # 2. Feature Engineering
        # -----------------------------
        df['year'] = df['release_date'].dt.year
        df['month'] = df['release_date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df.drop(["release_date", "month"], axis=1, inplace=True)

        for col in ['Director', 'Actors', 'Genres_clean']:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].apply(lambda x: [s.strip() for s in x.split(',')])
        df['Director'] = df['Director'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")
        df['Actors'] = df['Actors'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")
        df['Genres_clean'] = df['Genres_clean'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")

        # log_Recettes
        df['log_Recettes'] = np.log1p(df['Recettes'])

        # Weighted rating
        C = df['vote_average'].mean()
        m_val = df['vote_count'].quantile(0.90)
        def weighted_rating(row):
            v = row['vote_count']
            R = row['vote_average']
            if (v + m_val) == 0:
                return R
            return (v/(v+m_val))*R + (m_val/(v+m_val))*C
        df['weighted_rating'] = df.apply(weighted_rating, axis=1)
        df.drop(["vote_count", "vote_average"], axis=1, inplace=True)

        director_weighted_avg = df.groupby('Director')['weighted_rating'].mean().to_dict()
        df['director_weighted_avg'] = df['Director'].map(director_weighted_avg)
        actors_weighted_avg = df.groupby('Actors')['weighted_rating'].mean().to_dict()
        df['actors_weighted_avg'] = df['Actors'].map(actors_weighted_avg)
        print("✅ director_weighted_avg keys:", list(director_weighted_avg.keys())[:5])
        print("✅ actors_weighted_avg keys:", list(actors_weighted_avg.keys())[:5])

        df.loc[df['Budget'] == 1.0, 'Budget'] = 0
        df['is_blockbuster'] = (df['Budget'] >= 50000000).astype(int)
        df['actors_budget_interaction'] = df['actors_weighted_avg'] * df['Budget']
        df['log_Budget'] = np.log1p(df['Budget'])

        # -----------------------------
        # 3. Imputation par groupe
        # -----------------------------
        df['Budget'] = df.groupby("Genres_clean")['Budget'].transform(lambda x: x.fillna(x.median()))
        df['Recettes'] = df.groupby("Genres_clean")['Recettes'].transform(lambda x: x.fillna(x.median()))
        df['Budget'] = df.groupby("weighted_rating")['Budget'].transform(lambda x: x.fillna(x.median()))
        df['Recettes'] = df.groupby("weighted_rating")['Recettes'].transform(lambda x: x.fillna(x.median()))

        # -----------------------------
        # 4. Préparation des données
        # -----------------------------
        X = df.drop(["Recettes", "title", "Budget", "log_Recettes", "weighted_rating"], axis=1)
        y = df['log_Recettes']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Variables num et cat
        num_train = X_train.select_dtypes(include=["float", "int"])
        cat_train = X_train.select_dtypes(include=["object", "category"])
        num_test = X_test.select_dtypes(include=["float", "int"])
        cat_test = X_test.select_dtypes(include=["object", "category"])

        imputer_num = SimpleImputer(strategy='median')
        num_train_imputed = pd.DataFrame(imputer_num.fit_transform(num_train),
                                        columns=num_train.columns, index=num_train.index)
        num_test_imputed = pd.DataFrame(imputer_num.transform(num_test),
                                        columns=num_test.columns, index=num_test.index)

        imputer_cat = SimpleImputer(strategy='most_frequent')
        cat_train_imputed = pd.DataFrame(imputer_cat.fit_transform(cat_train),
                                        columns=cat_train.columns, index=cat_train.index)
        cat_test_imputed = pd.DataFrame(imputer_cat.transform(cat_test),
                                        columns=cat_test.columns, index=cat_test.index)

        combined_cat = pd.concat([cat_train_imputed, cat_test_imputed])
        le = LabelEncoder()
        for col in cat_train_imputed.columns:
            le.fit(combined_cat[col])
            cat_train_imputed[col] = le.transform(cat_train_imputed[col])
            cat_test_imputed[col] = le.transform(cat_test_imputed[col])

        X_train_final = pd.concat([num_train_imputed, cat_train_imputed], axis=1)
        X_test_final = pd.concat([num_test_imputed, cat_test_imputed], axis=1)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_test_scaled = scaler.transform(X_test_final)

        # -----------------------------
        # 5. Entraînement du modèle
        # -----------------------------
        rf_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features='sqrt',
            max_depth=20,
            random_state=42,
            bootstrap=False
        )
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("🔹 Score sur Train :", rf_model.score(X_train_scaled, y_train))
        st.write("🔹 Score sur Test :", rf_model.score(X_test_scaled, y_test))
        st.write("🔹 MSE :", mse)
        st.write("🔹 R² :", r2)

        # -----------------------------
        # 6. Sauvegarde du pipeline
        # -----------------------------
        pipeline = {
            "model": rf_model,
            "scaler": scaler,
            "label_encoder": le,
            "director_mapping": director_weighted_avg,
            "actor_mapping": actors_weighted_avg,
            "expected_features": X_train_final.columns.tolist()
        }
        print("🔍 Clés du pipeline avant sauvegarde:", pipeline.keys())
        joblib.dump(pipeline, "pipeline.joblib")
        st.success("Pipeline complet relancé et sauvegardé !")

elif page == pages[5]:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import streamlit as st
 
    st.title("🎥 Application")
    st.image("StarWars.jpg", width=700)
    st.write("Ici, vous pouvez renseigner les informations d’un film pour obtenir la prédiction de ses recettes.")
 
    ##################################
    # FONCTION SAFE POUR L'ENCODAGE
    ##################################
    def safe_label_transform(value, le):
        if value in le.classes_:
            return le.transform([value])[0]
        else:
            if "Unknown" in le.classes_:
                return le.transform(["Unknown"])[0]
            else:
                return -1
 
    ##################################
    # FONCTION DE TRANSFORMATION POUR INFÉRENCE
    ##################################
    def transform_new_data_inference(df_new, scaler, le, director_map, actor_map, expected_features):
        df_trans = df_new.copy()
 
        # Comme la cible Recettes et les votes ne sont pas disponibles à l'inférence,
        # on crée des colonnes dummy (elles seront supprimées après transformation)
        df_trans["Recettes"] = 0
        df_trans["vote_count"] = 0
        df_trans["vote_average"] = 0.0
 
        # Transformation de la date
        df_trans['release_date'] = pd.to_datetime(df_trans['release_date'], errors='coerce')
        df_trans['year'] = df_trans['release_date'].dt.year
        df_trans['month'] = df_trans['release_date'].dt.month
        df_trans['month_sin'] = np.sin(2 * np.pi * df_trans['month'] / 12)
        df_trans['month_cos'] = np.cos(2 * np.pi * df_trans['month'] / 12)
        df_trans.drop(["release_date", "month"], axis=1, inplace=True)
 
        # Traitement des colonnes textuelles
        for col in ['Director', 'Actors', 'Genres_clean']:
            df_trans[col] = df_trans[col].fillna("Unknown")
            df_trans[col] = df_trans[col].apply(lambda x: [s.strip() for s in x.split(',')])
        df_trans['Director'] = df_trans['Director'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")
        df_trans['Actors'] = df_trans['Actors'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")
        df_trans['Genres_clean'] = df_trans['Genres_clean'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")
 
        # Comme nous n'avons pas de véritables Recettes ni votes, on définit ces colonnes sur 0
        df_trans['log_Recettes'] = 0
        df_trans['weighted_rating'] = 0
 
        # Gestion du budget et création d'interactions
        df_trans.loc[df_trans['Budget'] == 1.0, 'Budget'] = 0
        df_trans['is_blockbuster'] = (df_trans['Budget'] >= 50000000).astype(int)
        df_trans['actors_weighted_avg'] = df_trans['Actors'].apply(lambda x: actor_map.get(x, 0))
        df_trans['actors_budget_interaction'] = df_trans['actors_weighted_avg'] * df_trans['Budget']
        df_trans['log_Budget'] = np.log1p(df_trans['Budget'])
 
        # Ajout du mapping pour le réalisateur
        # on prend la moyenne si le réalisateur n'est pas connu
        df_trans["director_weighted_avg"] = df_trans["Director"].apply(lambda x: director_map.get(x, np.mean(list(director_map.values()))))
 
        # On intègre la note de popularité saisie et on crée la colonne release_year
        df_trans["release_year"] = df_trans["year"]
 
        # Préparation finale : suppression des colonnes non utilisées à l'inférence
        X_new = df_trans.drop(["Recettes", "title", "Budget", "log_Recettes", "weighted_rating"], axis=1)
 
        # Séparation en variables numériques et catégorielles
        X_num = X_new.select_dtypes(include=["float", "int"])
        X_cat = X_new.select_dtypes(include=["object", "category"])
 
        # Encodage safe des variables catégorielles
        for col in X_cat.columns:
            X_cat[col] = X_cat[col].apply(lambda x: safe_label_transform(x, le))
 
        X_new_final = pd.concat([X_num, X_cat], axis=1)
        # Réindexer pour que l'ordre des colonnes corresponde aux features utilisées lors de l'entraînement
        X_new_final = X_new_final.reindex(columns=expected_features)
 
        # Optionnel : affichage intermédiaire pour vérification
        st.write("**Aperçu avant scaling**", X_new_final.head())
 
        X_new_scaled = scaler.transform(X_new_final)
        return X_new_scaled
 
    # Chargement du pipeline sauvegardé
    joblib_url = "https://github.com/MovieForecasting/DA_Project_Movieforecasting/releases/download/JR/pipeline.joblib"
    response = requests.get(joblib_url)
    pipeline = joblib.load(io.BytesIO(response.content))
    print("🔍 Clés du pipeline chargé:", pipeline.keys())
 
    st.write("#### Veuillez saisir les informations du film :")
 
    with st.form("my_form"):
        import datetime
        release_date = st.date_input("Date de sortie :", value=datetime.date(2017, 12, 15))
        Budget = st.number_input("Budget (en dollars):", min_value=0, value=317000000)
        Director = st.text_input("Nom du réalisateur:", value="Rian Johnson")
        Actors = st.text_input("Liste des acteurs (séparés par une virgule):", value="Daisy Ridley, John Boyega, Adam Driver")
        genres = ["Action", "Aventure", "Comédie", "Drame", "Science Fiction", "Horreur", "Romance", "Thriller", "Animation", "Documentaire"]
        Genres_clean = st.selectbox("Genre principal:", options=genres, index=genres.index("Science Fiction"))
        popularity_options = [
            "0-100 (Pas populaire)",
            "100-500 (Indépendant)",
            "500-1000 (Plutôt populaire)",
            "1000-5000 (Populaire)",
            "5000+ (Très populaire)"
        ]
        
        # Selectbox for choosing the popularity range
        popularity_range = st.selectbox("Sélectionnez le niveau de popularité :", options=popularity_options, index=popularity_options.index("5000+ (Très populaire)"))
        
        # Mapping from the selected range to a representative numeric value
        popularity_mapping = {
            "0-100 (Pas populaire)": 50,
            "100-500 (Indépendant)": 300,
            "500-1000 (Plutôt populaire)": 750,
            "1000-5000 (Populaire)": 3000,
            "5000+ (Très populaire)": 8000
        }
        
        popularity = popularity_mapping[popularity_range]
 
        submitted = st.form_submit_button("Prédire les recettes")
        if submitted:
            data_future = {
                "title": ["Rentrer le titre du film"],
                "release_date": [release_date],
                "Budget": [Budget],
                "Director": [Director],
                "Actors": [Actors],
                "Genres_clean": [Genres_clean],
                "popularity": [popularity]  # On inclut la popularité ici
            }
            df_future = pd.DataFrame(data_future)
            
            # Vérification de l'existence du réalisateur
            director_input = Director.strip()
            if director_input in pipeline["director_mapping"]:
                st.info(f"Le réalisateur '{director_input}' est reconnu dans notre dataset.")
            else:
                st.warning(f"Le réalisateur '{director_input}' n'est pas reconnu dans notre dataset.")
 
            # Vérification de l'existence des acteurs
            actor_inputs = [actor.strip() for actor in Actors.split(",")]
            recognized_actors = [actor for actor in actor_inputs if actor in pipeline["actor_mapping"]]
            unrecognized_actors = [actor for actor in actor_inputs if actor not in pipeline["actor_mapping"]]
 
            if recognized_actors:
                st.info(f"Acteurs reconnus: {', '.join(recognized_actors)}")
            if unrecognized_actors:
                st.warning(f"Acteurs non reconnus: {', '.join(unrecognized_actors)}")
 
            # Transformation des données
            X_future = transform_new_data_inference(df_future,
                                                    pipeline["scaler"],
                                                    pipeline["label_encoder"],
                                                    pipeline["director_mapping"],
                                                    pipeline["actor_mapping"],
                                                    pipeline["expected_features"])
 
            prediction = pipeline["model"].predict(X_future)
            st.write("**Prédiction (log_Recettes)**:", prediction[0])
            recettes_pred = np.expm1(prediction)
            st.write("**Prédiction (Recettes)**:", recettes_pred[0])
 
            # Affichage en millions
            recettes_millions = recettes_pred[0] / 1e6
            st.success(f"Prédiction (Recettes) : {recettes_millions:.2f} millions de dollars")
elif page == pages[6]:
    st.image("Matrix.jpg", width=700)
    st.write("### Conclusion")
    st.write("""
    La démarche adoptée nous a permis de développer un modèle robuste et fiable pour prédire la performance financière des films. En combinant un traitement rigoureux des données (gestion des valeurs manquantes, extraction de nouvelles variables pertinentes telles que “is_blockbuster”, l’interaction “actors_budget_interaction” et l’intégration de la saisonnalité) avec une optimisation précise des hyper paramètres du modèle Random Forest Regressor via GridSearchCV, nous avons significativement amélioré la performance prédictive.
    
    L’optimisation fine des hyper paramètres a permis une réduction notable de l’overfitting, avec un score sur le jeu d'entraînement passant de 0.9489 à 0.9239 et une amélioration du score sur le jeu de données de test de 0.6179 à 0.6920. Cette progression démontre l’importance d’une sélection pertinente des variables, notamment l’inclusion d’interactions (comme “actors_budget_interaction”) et des variables temporelles telles que la saisonnalité.
    
    En conclusion, ce projet met en évidence que notre approche permet de fournir des prédictions fiables et précises des recettes futures, ouvrant ainsi des perspectives prometteuses pour une exploitation concrète dans l’industrie du cinéma, notamment pour la prise de décisions stratégiques et financières.
    
    En outre, la résolution de l’overfitting est un défi fréquent en data science. Pour y faire face, des techniques comme la validation croisée K-Fold, qui n’ont pas été abordées dans notre formation, pourraient être envisagées pour améliorer encore la robustesse du modèle.
    """)

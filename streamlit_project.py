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

    st.title("🎬 Modélisation et Machine Learning - Page Indépendante")
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
 
    st.title("🎥 Application – Inférence")
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
    pipeline = joblib.load("pipeline.joblib")
    print("🔍 Clés du pipeline chargé:", pipeline.keys())
 
    st.write("#### Veuillez saisir les informations du film :")
 
    with st.form("my_form"):
        release_date = st.text_input("Date de sortie (YYYY-MM-DD): ", value="2025-05-10")
        Budget = st.number_input("Budget (en dollars):", min_value=0, value=120000000)
        Director = st.text_input("Nom du réalisateur:", value="Christopher Nolan")
        Actors = st.text_input("Liste des acteurs (séparés par une virgule):", value="Leonardo DiCaprio, Emma Stone")
        Genres_clean = st.text_input("Genre principal:", value="Science Fiction")
        popularity = st.number_input("Note de popularité (ex. 5000):", min_value=0.0, value=5000.0)
 
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

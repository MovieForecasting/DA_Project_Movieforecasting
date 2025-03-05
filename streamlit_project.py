import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import r2_score

df_exploration = pd.read_csv("df_github.csv")

st.title("Prévision du succès d'un film")

st.sidebar.title("Sommaire")

pages=["Présentation du projet", "Exploration du Dataset et DataViz'", "Pré-processing", "Modélisation et Machine Learning", "Conclusion"]

page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
  
  st.write("### Présentation du projet")
  
  st.write("Ce projet a été réalisé dans le cadre de notre formation en Data Analyse via l'organisme Data Scientest.")
  st.write("L'objectif de ce projet est de prédire le succès d'un film au box-office en utilisant le jeu de donnée issu de TMDB : 'The Movie DataBase' où nous pouvons obtenir des informations comme le budget, les recettes, le genre, les acteurs et les réalisateurs.")
  st.write("Grâce à l’analyse de ces données, nous pouvons mieux comprendre lequelles ont une influence dans le succès d'un film.")
  
  st.write("### Problématique")
  
  st.write("Quels sont les éléments clés qui influencent le succès d’un film ? Peut-on prédire ce succès à partir des données disponibles ?")

elif page == pages[1]:
    st.write("### Exploration du Dataset et DataViz'")

    st.dataframe(df_exploration.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df_exploration.shape)

# A COMPLETER

elif page == pages[2]:
    st.write("### Pré-processing")

# A COMPLETER

elif page == pages[3]:
    st.write("### Modélisation et Machine Learning")

# A COMPLETER

elif page == pages[4]:
    st.write("### Conclusion")

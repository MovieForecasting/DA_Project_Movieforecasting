import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Prévision du succès d'un film")

st.sidebar.title("Sommaire")
pages=["Présentation du projet", "Exploration du Dataset et DataViz'", "Pré-processing","Modélisation","Machine Learning","Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

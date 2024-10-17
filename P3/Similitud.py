import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import math
import numpy as np
import spacy
import os
import pickle

@st.cache_resource
def load_spacy_model():
    return spacy.load("es_core_news_sm")

nlp = load_spacy_model()

# Función para normalizar y tokenizar usando spaCy (del notebook)
def normalizador(corpus: pd.DataFrame, col_name: str, obj_nlp) -> pd.DataFrame:
    words_category = ["DET", "ADP", "CCONJ", "SCONJ","PRON"]
    corpus.fillna('', inplace=True)
    list_col = corpus[col_name].tolist()
    list_final = []
    
    for i in range(len(list_col)):
        list_to_normal = list_col[i].lower()
        doc = obj_nlp(list_to_normal)
        list_normal = []
        for token in doc:
            if token.pos_ not in words_category:
                list_normal.append(token.lemma_)
        text_norm = ' '.join(list_normal)
        list_final.append(text_norm)
    
    corpus[col_name] = list_final
    return corpus

if not os.path.exists('models'):
    os.makedirs('models')

# Cargar el corpus automáticamente desde archivo CSV
@st.cache_data
def load_corpus():
    df = pd.read_csv('raw_data_corpus.csv')
    df = normalizador(df, 'Title', nlp)
    df = normalizador(df, 'Content', nlp)
    return df

st.sidebar.title("Similitud de Documentos - Práctica III")

# Cargar arcrhivo de prueba
archivo_prueba = st.sidebar.file_uploader("Carga el documento de prueba (archivo .txt obligatorio)")

# Opciones de selección
st.sidebar.subheader("Selecciona las características del corpus para analizar")
use_title = st.sidebar.checkbox("Título")
use_content = st.sidebar.checkbox("Contenido")
use_title_content = st.sidebar.checkbox("Título + Contenido")

# Selección de n-gramas
st.sidebar.subheader("Selecciona el tipo de características")
use_unigramas = st.sidebar.checkbox("Unigrama")
use_bigramas = st.sidebar.checkbox("Bigrama")

# Selección de método de vectorización
st.sidebar.subheader("Selecciona el tipo de vectorización")
use_frecuencia = st.sidebar.checkbox("Frecuencia")
use_binarizada = st.sidebar.checkbox("Binarizada")
use_tfidf = st.sidebar.checkbox("TF-IDF")

st.title("Resultados de la similitud de documentos")

if archivo_prueba is not None:
    a_prueba = archivo_prueba.read().decode('utf-8')
    a_prueba_norm = normalizador(pd.DataFrame({'a_prueba': [a_prueba]}, index=[0]), 'a_prueba', nlp)['a_prueba'].iloc[0]

    # Crear tabla de prueba
    


tabla_prueba = {
    'Corpus document': 'si',
    'Vector reprensentation': 'si',
    'Extracted features': 'si',
    'Comparison element': 'si',
    'Similarity value': 1.0    
}

df_prueba = pd.DataFrame(tabla_prueba, index=[0])
st.table(df_prueba)
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import math
import numpy as np
import spacy
import os
import pickle

# Cargar el modelo en español de spaCy
nlp = spacy.load("es_core_news_sm")

# Función para normalizar y tokenizar usando spaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token) > 1
    ]
    return " ".join(tokens)

# Función para contar los archivos .pkl en la carpeta 'models'
def count_pkl_files(directory='models'):
    if os.path.exists(directory):
        return len([f for f in os.listdir(directory) if f.endswith('.pkl')])
    return 0

# Crear carpeta para almacenar los .pkl si no existe
if not os.path.exists('models'):
    os.makedirs('models')

# Título de la aplicación
st.title("Similitud de Documentos - Práctica III")

# Mostrar el número de archivos .pkl creados
st.subheader(f"Archivos .pkl creados en 'models': {count_pkl_files()}")

# Subir archivo del corpus
uploaded_file = st.file_uploader("Sube el corpus de noticias (CSV)", type=["csv"])
if uploaded_file is not None:
    # Cargar el archivo como dataframe de Pandas
    df = pd.read_csv(uploaded_file)
    st.write("Corpus cargado con éxito.")

    # Asegurarse de que el archivo tenga las columnas necesarias
    if 'Title' in df.columns and 'Content' in df.columns:
        # Seleccionar columnas a utilizar
        feature_selection = st.multiselect("Selecciona los elementos del corpus para analizar", ['Title', 'Content', 'Title + Content'])
        
        if feature_selection:
            # Crear corpus concatenando las columnas seleccionadas y reemplazar NaN con cadenas vacías
            corpus = []
            if 'Title' in feature_selection:
                corpus.extend(df['Title'].fillna('').tolist())  # Reemplazar NaN con cadenas vacías
            if 'Content' in feature_selection:
                corpus.extend(df['Content'].fillna('').tolist())  # Reemplazar NaN con cadenas vacías
            if 'Title + Content' in feature_selection:
                corpus.extend((df['Title'].fillna('') + " " + df['Content'].fillna('')).tolist())  # Reemplazar NaN con cadenas vacías

            # Normalizar y tokenizar el corpus utilizando spaCy
            corpus = [preprocess_text(doc) for doc in corpus]

            # Selección de n-gramas
            ngram_range = st.radio("Selecciona el tipo de características", ["Unigramas", "Bigramas"])

            # Selección de método de vectorización
            method = st.selectbox("Selecciona el tipo de vectorización", ["Frecuencia", "Binarizada", "TF-IDF"])

            # Vectorizar corpus con opción de guardar/cargar desde .pkl
            if 'vectorized_corpus' not in st.session_state:
                st.session_state['vectorized_corpus'] = None
                st.session_state['vectorizer'] = None

            if st.button("Vectorizar Corpus"):
                # Definir el rango de n-gramas según la selección
                ngram_values = (1, 1) if ngram_range == "Unigramas" else (2, 2)

                # Crear y guardar vocabulario y representación matricial
                def save_vector_and_vocab(vectorizer, corpus, feature_name, ngram_type):
                    # Crear el vectorizador y vectorizar el corpus
                    X = vectorizer.fit_transform(corpus)
                    
                    # Guardar el vocabulario
                    vocab_file_name = os.path.join('models', f'vocab_{feature_name}_{ngram_type}_{method}.pkl')
                    with open(vocab_file_name, 'wb') as vocab_file:
                        pickle.dump(vectorizer.vocabulary_, vocab_file)
                    
                    # Guardar la representación matricial
                    matrix_file_name = os.path.join('models', f'matrix_{feature_name}_{ngram_type}_{method}.pkl')
                    with open(matrix_file_name, 'wb') as matrix_file:
                        pickle.dump(X, matrix_file)
                    st.write(f"Archivos guardados para {feature_name} con {ngram_type} y {method}:")
                    st.write(f"Vocabulario: {vocab_file_name}, Matriz: {matrix_file_name}")

                # Definir el vectorizador según el método seleccionado
                if method == 'Frecuencia':
                    vectorizer = CountVectorizer(ngram_range=ngram_values)
                elif method == 'Binarizada':
                    vectorizer = CountVectorizer(binary=True, ngram_range=ngram_values)
                elif method == 'TF-IDF':
                    vectorizer = TfidfVectorizer(ngram_range=ngram_values)

                # Guardar para cada característica seleccionada
                for feature in feature_selection:
                    if feature == 'Title':
                        save_vector_and_vocab(vectorizer, df['Title'].fillna('').tolist(), 'Title', ngram_range)
                    elif feature == 'Content':
                        save_vector_and_vocab(vectorizer, df['Content'].fillna('').tolist(), 'Content', ngram_range)
                    elif feature == 'Title + Content':
                        combined_data = (df['Title'].fillna('') + " " + df['Content'].fillna('')).tolist()
                        save_vector_and_vocab(vectorizer, combined_data, 'Title_Content', ngram_range)

                # Actualizar el número de archivos .pkl creados
                st.subheader(f"Archivos .pkl creados en 'models': {count_pkl_files()}")
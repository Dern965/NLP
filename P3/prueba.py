import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import math
import numpy as np
import spacy

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

# Título de la aplicación
st.title("Similitud de Documentos - Práctica III")

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

            # Vectorizar corpus
            if 'vectorized_corpus' not in st.session_state:
                st.session_state['vectorized_corpus'] = None
                st.session_state['vectorizer'] = None

            if st.button("Vectorizar Corpus"):
                # Definir el rango de n-gramas según la selección
                ngram_values = (1, 1) if ngram_range == "Unigramas" else (2, 2)

                # Seleccionar el vectorizador adecuado
                if method == 'Frecuencia':
                    vectorizer = CountVectorizer(ngram_range=ngram_values)
                elif method == 'Binarizada':
                    vectorizer = CountVectorizer(binary=True, ngram_range=ngram_values)
                elif method == 'TF-IDF':
                    vectorizer = TfidfVectorizer(ngram_range=ngram_values)

                # Generar la representación vectorial y guardar en session_state
                X = vectorizer.fit_transform(corpus)
                st.session_state['vectorized_corpus'] = X
                st.session_state['vectorizer'] = vectorizer
                st.session_state['feature_names'] = vectorizer.get_feature_names_out()
                st.write("Características extraídas:", st.session_state['feature_names'])
                st.write("Representación vectorial generada.")

            # Mostrar características extraídas después de la vectorización
            if st.session_state['vectorized_corpus'] is not None:
                st.write("Características extraídas:", st.session_state['feature_names'])
                st.write("Representación vectorial generada.")

            # Ingresar documento de prueba
            if 'test_doc' not in st.session_state:
                st.session_state['test_doc'] = ""

            test_option = st.radio("Selecciona cómo ingresar el documento de prueba", ["Escribir texto", "Cargar archivo"])
            if test_option == "Escribir texto":
                st.session_state['test_doc'] = st.text_area("Ingresa el documento de prueba", value=st.session_state['test_doc'])
            elif test_option == "Cargar archivo":
                test_file = st.file_uploader("Carga el archivo del documento de prueba", type=["txt"])
                if test_file is not None:
                    st.session_state['test_doc'] = test_file.read().decode('utf-8')

            # Normalizar y tokenizar el documento de prueba
            if st.session_state['test_doc']:
                test_doc_processed = preprocess_text(st.session_state['test_doc'])

            # Calcular Similitud
            if st.button("Calcular Similitud"):
                if st.session_state['vectorized_corpus'] is not None and test_doc_processed:
                    # Vectorizar el documento de prueba
                    test_vector = st.session_state['vectorizer'].transform([test_doc_processed]).toarray()

                    # Verificar si el vector de prueba no es completamente cero
                    if np.all(test_vector == 0):
                        st.warning("El documento de prueba no tiene características válidas después de la tokenización y vectorización.")
                    else:
                        # Calcular similitud con cada documento del corpus
                        def cosine_similarity(x, y):
                            # Calcular el producto punto y las magnitudes
                            val = sum(x[index] * y[index] for index in range(len(x)))
                            sr_x = math.sqrt(sum(x_val**2 for x_val in x))
                            sr_y = math.sqrt(sum(y_val**2 for y_val in y))
                            # Verificar que las magnitudes no sean cero
                            if sr_x == 0 or sr_y == 0:
                                return np.nan
                            return val / (sr_x * sr_y)

                        similarities = []
                        for vector in st.session_state['vectorized_corpus'].toarray():
                            sim = cosine_similarity(test_vector[0], vector)
                            similarities.append(sim)

                        # Mostrar los 10 documentos más similares
                        sorted_indices = sorted(range(len(similarities)), key=lambda k: similarities[k] if not np.isnan(similarities[k]) else -1, reverse=True)[:10]
                        if sorted_indices:
                            st.subheader("Documentos Más Similares")
                            for idx in sorted_indices:
                                st.write(f"Documento {idx + 1}: Similitud = {similarities[idx]}")
                        else:
                            st.warning("No se encontraron documentos similares.")
                else:
                    st.error("Primero debes ingresar el documento de prueba y generar la representación vectorial del corpus.")

    else:
        st.error("El archivo subido no tiene las columnas 'Title' y 'Content'.")

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import math
import numpy as np
import spacy
import os
import pickle

# Cargar el modelo en español de spaCy con caché para mejorar el rendimiento
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
        feature_selection = st.multiselect(
            "Selecciona los elementos del corpus para analizar",
            ['Title', 'Content', 'Title + Content']
        )

        if feature_selection:
            # Crear corpus concatenando las columnas seleccionadas y reemplazar NaN con cadenas vacías
            corpus = []
            if 'Title' in feature_selection:
                corpus.extend(df['Title'].fillna('').tolist())
            if 'Content' in feature_selection:
                corpus.extend(df['Content'].fillna('').tolist())
            if 'Title + Content' in feature_selection:
                corpus.extend((df['Title'].fillna('') + " " + df['Content'].fillna('')).tolist())

            # Normalizar el corpus utilizando la función normalizador
            with st.spinner("Procesando el corpus..."):
                df = normalizador(df, 'Title', nlp)
                df = normalizador(df, 'Content', nlp)

            # Selección de n-gramas
            ngram_range = st.radio("Selecciona el tipo de características", ["Unigramas", "Bigramas"])

            # Selección de método de vectorización
            method = st.selectbox("Selecciona el tipo de vectorización", ["Frecuencia", "Binarizada", "TF-IDF"])

            # Inicializar diccionarios en session_state para almacenar vectores y vectorizadores
            if 'vectorized_corpus_dict' not in st.session_state:
                st.session_state['vectorized_corpus_dict'] = {}
                st.session_state['vectorizers'] = {}

            if st.button("Vectorizar Corpus"):
                # Definir el rango de n-gramas según la selección
                ngram_values = (1, 1) if ngram_range == "Unigramas" else (2, 2)

                # Función para guardar el vectorizador y la matriz
                def save_vector_and_vocab(vectorizer, corpus_data, feature_name, ngram_type, method_type):
                    # Crear el vectorizador y vectorizar el corpus
                    X = vectorizer.fit_transform(corpus_data)

                    # Guardar el vocabulario
                    vocab_file_name = os.path.join('models', f'vocab_{feature_name}_{ngram_type}_{method_type}.pkl')
                    with open(vocab_file_name, 'wb') as vocab_file:
                        pickle.dump(vectorizer.vocabulary_, vocab_file)

                    # Guardar la representación matricial
                    matrix_file_name = os.path.join('models', f'matrix_{feature_name}_{ngram_type}_{method_type}.pkl')
                    with open(matrix_file_name, 'wb') as matrix_file:
                        pickle.dump(X, matrix_file)

                    # Actualizar session_state
                    key = f"{feature_name}_{ngram_type}_{method_type}"
                    st.session_state['vectorized_corpus_dict'][key] = X
                    st.session_state['vectorizers'][key] = vectorizer

                    st.success(f"Archivos guardados para {feature_name} con {ngram_type} y {method_type}:")
                    st.write(f"Vocabulario: {vocab_file_name}, Matriz: {matrix_file_name}")

                # Definir el vectorizador según el método seleccionado
                if method == 'Frecuencia':
                    VectorizerClass = CountVectorizer
                    vectorizer_params = {'ngram_range': ngram_values}
                elif method == 'Binarizada':
                    VectorizerClass = CountVectorizer
                    vectorizer_params = {'binary': True, 'ngram_range': ngram_values}
                elif method == 'TF-IDF':
                    VectorizerClass = TfidfVectorizer
                    vectorizer_params = {'ngram_range': ngram_values}

                # Guardar para cada característica seleccionada
                for feature in feature_selection:
                    if feature == 'Title':
                        data = df['Title'].fillna('').tolist()
                        feature_name = 'Title'
                    elif feature == 'Content':
                        data = df['Content'].fillna('').tolist()
                        feature_name = 'Content'
                    elif feature == 'Title + Content':
                        data = (df['Title'].fillna('') + " " + df['Content'].fillna('')).tolist()
                        feature_name = 'Title_Content'

                    # Crear y guardar el vectorizador y la matriz
                    vectorizer = VectorizerClass(**vectorizer_params)
                    save_vector_and_vocab(vectorizer, data, feature_name, ngram_range, method)

                # Actualizar el número de archivos .pkl creados
                st.subheader(f"Archivos .pkl creados en 'models': {count_pkl_files()}")

        else:
            st.warning("Por favor, selecciona al menos una característica para analizar.")

        # Sección de Similitud de Documentos
        st.header("Cálculo de Similitud entre Documentos")

        # Verificar si existen configuraciones vectorizadas
        if 'vectorized_corpus_dict' in st.session_state and st.session_state['vectorized_corpus_dict']:
            # Generar las opciones de configuración disponibles
            available_configs = list(st.session_state['vectorized_corpus_dict'].keys())
            similarity_config = st.selectbox(
                "Selecciona la configuración de vectorización para calcular la similitud",
                available_configs
            )

            if similarity_config:
                # Obtener la matriz vectorizada y el vectorizador correspondiente
                X_corpus = st.session_state['vectorized_corpus_dict'][similarity_config]
                vectorizer = st.session_state['vectorizers'][similarity_config]

                # Ingresar documento de prueba
                test_option = st.radio("Selecciona cómo ingresar el documento de prueba", ["Escribir texto", "Cargar archivo"])

                if test_option == "Escribir texto":
                    test_doc = st.text_area("Ingresa el documento de prueba")
                elif test_option == "Cargar archivo":
                    test_file = st.file_uploader("Carga el archivo del documento de prueba", type=["txt"])
                    if test_file is not None:
                        test_doc = test_file.read().decode('utf-8')
                    else:
                        test_doc = ""

                # Almacenar el documento de prueba en session_state
                if 'test_doc' not in st.session_state:
                    st.session_state['test_doc'] = ""

                if test_option == "Escribir texto":
                    st.session_state['test_doc'] = test_doc
                elif test_option == "Cargar archivo" and test_file is not None:
                    st.session_state['test_doc'] = test_doc

                # Procesar el documento de prueba
                if st.session_state['test_doc']:
                    test_doc_processed = normalizador(pd.DataFrame({'test_doc': [st.session_state['test_doc']]}, index=[0]), 'test_doc', nlp)['test_doc'].iloc[0]

                    if st.button("Calcular Similitud"):
                        if test_doc_processed:
                            # Vectorizar el documento de prueba
                            test_vector = vectorizer.transform([test_doc_processed]).toarray()

                            # Verificar si el vector de prueba no es completamente cero
                            if np.all(test_vector == 0):
                                st.warning("El documento de prueba no tiene características válidas después de la tokenización y vectorización.")
                            else:
                                # Calcular similitud con cada documento del corpus
                                with st.spinner("Calculando similitudes..."):
                                    # Utilizar operaciones vectorizadas para eficiencia
                                    X_corpus_dense = X_corpus.toarray()
                                    similarities = np.dot(X_corpus_dense, test_vector.T).flatten()
                                    norms_corpus = np.linalg.norm(X_corpus_dense, axis=1)
                                    norm_test = np.linalg.norm(test_vector)
                                    similarities = similarities / (norms_corpus * norm_test)
                                    similarities = np.nan_to_num(similarities)  # Reemplazar NaN con 0

                                    # Obtener los índices de los 10 documentos más similares
                                    top_indices = similarities.argsort()[-10:][::-1]
                                    top_similarities = similarities[top_indices]

                                # Mostrar los 10 documentos más similares
                                st.subheader("Documentos Más Similares")
                                for idx, sim in zip(top_indices, top_similarities):
                                    st.write(f"Documento {idx + 1}: Similitud = {sim:.4f}")
                        else:
                            st.error("El documento de prueba no contiene texto válido después del preprocesamiento.")
        else:
            st.info("Por favor, vectoriza el corpus primero para habilitar la sección de similitud.")
    else:
        st.warning("Por favor, sube un archivo CSV que contenga las columnas 'Title' y 'Content'.")
else:
    st.info("Esperando a que subas un archivo CSV para comenzar.")

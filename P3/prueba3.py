import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import spacy
import os
import pickle

# Cargar el modelo en español de spaCy con caché para mejorar el rendimiento
@st.cache_resource
def load_spacy_model():
    return spacy.load("es_core_news_sm")

nlp = load_spacy_model()

# Función para normalizar y tokenizar usando spaCy
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

# Crear carpeta para almacenar los .pkl si no existe
if not os.path.exists('models'):
    os.makedirs('models')

# Función para guardar el vectorizador y la matriz como archivos .pkl
def save_vector_and_matrix(vectorizer, X, feature_name, ngram_type, method_type):
    # Guardar el vocabulario
    vocab_file = f'models/vocab_{feature_name}_{ngram_type}_{method_type}.pkl'
    with open(vocab_file, 'wb') as f:
        pickle.dump(vectorizer.vocabulary_, f)

    # Guardar la representación matricial
    matrix_file = f'models/matrix_{feature_name}_{ngram_type}_{method_type}.pkl'
    with open(matrix_file, 'wb') as f:
        pickle.dump(X, f)

    st.write(f"Archivos guardados: {vocab_file} y {matrix_file}")  # Depuración
    
# Cargar el corpus automáticamente desde archivo CSV
@st.cache_data
def load_corpus():
    df = pd.read_csv('raw_data_corpus.csv')  # Ajusta la ruta si es necesario
    df = normalizador(df, 'Title', nlp)
    df = normalizador(df, 'Content', nlp)
    return df

def load_vector_and_matrix(feature_name, ngram_type, method_type):
    vocab_file = f'models/vocab_{feature_name}_{ngram_type}_{method_type}.pkl'
    matrix_file = f'models/matrix_{feature_name}_{ngram_type}_{method_type}.pkl'

    if os.path.exists(vocab_file) and os.path.exists(matrix_file):
        st.write(f"Modelo cargado: {vocab_file} y {matrix_file}")  # Depuración
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
        with open(matrix_file, 'rb') as f:
            X = pickle.load(f)

        # Crear el vectorizador con el vocabulario cargado
        if method_type == 'Frecuencia' or method_type == 'Binarizada':
            vectorizer = CountVectorizer(vocabulary=vocab, binary=(method_type == 'Binarizada'))
        else:
            vectorizer = TfidfVectorizer(vocabulary=vocab)

        return vectorizer, X  # Retorna el vectorizador ajustado y la matriz
    else:
        st.warning(f"No se encontraron los archivos: {vocab_file} o {matrix_file}")  # Depuración
        return None, None


# Calcular similitud y generar resultados
def calcular_similitud(test_doc_norm, corpus, vectorizer, X_corpus):
    test_vector = vectorizer.transform([test_doc_norm]).toarray()

    if np.all(test_vector == 0):
        st.warning("El documento de prueba no tiene características válidas después de la tokenización y vectorización.")
    else:
        X_corpus_dense = X_corpus.toarray()
        similarities = np.dot(X_corpus_dense, test_vector.T).flatten()
        norms_corpus = np.linalg.norm(X_corpus_dense, axis=1)
        norm_test = np.linalg.norm(test_vector)
        similarities = similarities / (norms_corpus * norm_test)
        similarities = np.nan_to_num(similarities)
        return similarities
    return None

# Vectorizar corpus y generar pruebas exhaustivas
def realizar_pruebas_exhaustivas(test_doc_norm):
    corpus_to_analyze = []
    if use_title:
        corpus_to_analyze.extend(df['Title'].fillna('').tolist())
    if use_content:
        corpus_to_analyze.extend(df['Content'].fillna('').tolist())
    if use_title_content:
        combined_col = (df['Title'].fillna('') + " " + df['Content'].fillna('')).tolist()
        corpus_to_analyze.extend(combined_col)

    resultados_similitud = []

    for ngram_range, ngram_name in [((1, 1), 'Unigrama'), ((2, 2), 'Bigrama')]:
        if (ngram_name == 'Unigrama' and use_unigramas) or (ngram_name == 'Bigrama' and use_bigramas):
            for method, method_name in [(CountVectorizer, 'Frecuencia'), 
                                        (CountVectorizer, 'Binarizada'), 
                                        (TfidfVectorizer, 'TF-IDF')]:
                if (method_name == 'Frecuencia' and use_frecuencia) or \
                   (method_name == 'Binarizada' and use_binarizada) or \
                   (method_name == 'TF-IDF' and use_tfidf):

                    for feature in ['Title', 'Content', 'Title_Content']:
                        if (feature == 'Title' and use_title) or \
                           (feature == 'Content' and use_content) or \
                           (feature == 'Title_Content' and use_title_content):

                            vectorizer, X_corpus = load_vector_and_matrix(feature, ngram_name, method_name)
                            if vectorizer is not None and X_corpus is not None:
                                # Generar las similitudes
                                similarities = calcular_similitud(test_doc_norm, corpus_to_analyze, vectorizer, X_corpus)
                                if similarities is not None:
                                    top_indices = similarities.argsort()[-10:][::-1]
                                    top_similarities = similarities[top_indices]
                                    for idx, sim in zip(top_indices, top_similarities):
                                        resultados_similitud.append({
                                            'Documento': idx + 1,
                                            'Similitud': f"{sim:.4f}",
                                            'Vector Representation': method_name,
                                            'Features': ngram_name,
                                            'Comparison': feature
                                        })

    return resultados_similitud

# Diseño de la interfaz acorde al boceto
st.title("Similitud de Documentos - Práctica III")

# Cargar corpus automáticamente
df = load_corpus()

# Selección de características
st.subheader("Selecciona las características del corpus para analizar")
use_title = st.checkbox("Título")
use_content = st.checkbox("Contenido")
use_title_content = st.checkbox("Título + Contenido")

# Selección de n-gramas
st.subheader("Selecciona el tipo de características")
use_unigramas = st.checkbox("Unigrama")
use_bigramas = st.checkbox("Bigrama")

# Selección de método de vectorización
st.subheader("Selecciona el tipo de vectorización")
use_frecuencia = st.checkbox("Frecuencia")
use_binarizada = st.checkbox("Binarizada")
use_tfidf = st.checkbox("TF-IDF")

# Ingreso de documento de prueba obligatorio desde archivo .txt
st.subheader("Carga el documento de prueba (archivo .txt obligatorio)")
test_file = st.file_uploader("Carga el archivo del documento de prueba", type=["txt"])

if test_file is not None:
    test_doc = test_file.read().decode('utf-8')
    test_doc_norm = normalizador(pd.DataFrame({'test_doc': [test_doc]}, index=[0]), 'test_doc', nlp)['test_doc'].iloc[0]

    # Realizar pruebas exhaustivas automáticamente
    resultados_similitud = realizar_pruebas_exhaustivas(test_doc_norm)
    
    if resultados_similitud:
        # Mostrar los resultados en la tabla solicitada
        st.subheader("Top 10 Documentos Más Similares")
        results_df = pd.DataFrame(resultados_similitud)
        st.table(results_df)
    else:
        st.warning("No se encontraron resultados de similitud.")
else:
    st.warning("Por favor, carga un archivo de prueba en formato .txt para continuar.")

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Función para cargar y mostrar el corpus
def cargar_corpus(file):
    try:
        corpus_data = pd.read_csv(file)
        st.success("Corpus cargado correctamente")
        return corpus_data
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Función para aplicar vectorización
def vectorizar(corpus, vector_type, ngram_range):
    if vector_type == "Frecuencia":
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    elif vector_type == "TF-IDF":
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    else:
        st.error("Tipo de vectorización no válido")
        return None
    
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# Función para calcular la similitud de coseno
def calcular_similitud(doc_prueba, corpus_vectorizado, vectorizer):
    doc_vectorizado = vectorizer.transform([doc_prueba])
    similitudes = cosine_similarity(doc_vectorizado, corpus_vectorizado).flatten()
    return similitudes

# Configuración de la interfaz
st.title("Similitud de Documentos")
st.write("Cargar el archivo del corpus y seleccionar el documento de prueba para calcular la similitud.")

# Paso 1: Cargar el corpus
corpus_file = st.file_uploader("Sube el archivo CSV del corpus", type="csv")

if corpus_file is not None:
    corpus_data = cargar_corpus(corpus_file)
    
    if corpus_data is not None:
        # Mostrar una vista previa del corpus
        st.write("Vista previa del corpus:")
        st.write(corpus_data.head())

        # Seleccionar las columnas de "Title" y "Content"
        corpus_title = corpus_data['Title'].fillna('')
        corpus_summary = corpus_data['Content'].fillna('')
        corpus_combined = corpus_title + " " + corpus_summary

        # Paso 2: Seleccionar cómo ingresar el documento de prueba (escribir o archivo)
        option = st.selectbox("Selecciona cómo ingresar el documento de prueba", ("Subir archivo", "Escribir texto"))

        if option == "Subir archivo":
            test_file = st.file_uploader("Sube el documento de prueba (archivo de texto)", type="txt")
            if test_file is not None:
                test_document = test_file.read().decode('utf-8')
                st.write("Contenido del documento de prueba:")
                st.write(test_document)

        elif option == "Escribir texto":
            test_document = st.text_area("Escribe el texto del documento de prueba")
            if test_document:
                st.write("Contenido del documento de prueba:")
                st.write(test_document)

        # Paso 3: Seleccionar opciones de vectorización
        vector_type = st.selectbox("Selecciona el tipo de vectorización", ["Frecuencia", "TF-IDF"])
        ngram_type = st.selectbox("Selecciona las características", ["Unigramas", "Bigrams"])

        ngram_range = (1, 1) if ngram_type == "Unigramas" else (2, 2)

        # Paso 4: Calcular la similitud
        if st.button("Calcular Similitud"):
            if test_document:
                corpus_vectorizado, vectorizer = vectorizar(corpus_combined, vector_type, ngram_range)

                if corpus_vectorizado is not None:
                    similitudes = calcular_similitud(test_document, corpus_vectorizado, vectorizer)

                    # Ordenar y mostrar los 10 documentos más similares
                    top_10_similar_indices = similitudes.argsort()[-10:][::-1]

                    st.write("Documentos más similares:")
                    for idx in top_10_similar_indices:
                        st.write(f"Documento {idx}: Similitud {similitudes[idx]:.4f}")
            else:
                st.error("Por favor, ingresa o sube un documento de prueba.")

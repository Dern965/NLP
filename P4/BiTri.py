import streamlit as st
import pandas as pd
import spacy
import os
from collections import Counter
from itertools import islice

# Carga el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Crear la carpeta 'Ngrams' si no existe
if not os.path.exists("Ngrams"):
    os.makedirs("Ngrams")

# Función para cargar y procesar el corpus
def load_corpus(file):
    corpus = pd.read_csv(file)
    if 'Mensaje' in corpus.columns:
        corpus.columns = ['Text']  # Cambiar nombre de "Mensaje" a "Text"
    return corpus

# Función para tokenizar el corpus y conservar signos de puntuación y caracteres especiales
def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

# Función para generar n-gramas (bigrams o trigrams)
def generate_ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))

# Función para calcular la frecuencia y probabilidad condicional de bigramas
def bigram_stats(tokens):
    bigrams = generate_ngrams(tokens, 2)
    bigram_freq = Counter(bigrams)
    context_freq = Counter([bigram[0] for bigram in bigrams])
    bigram_data = []
    
    for (term1, term2), freq in bigram_freq.items():
        context_count = context_freq[term1]
        prob = freq / context_count
        bigram_data.append([term1, term2, freq, context_count, prob])
        
    return pd.DataFrame(bigram_data, columns=["Term 1", "Term 2", "Bigram Frequency", "Context Frequency", "Conditional Probability"])

# Función para calcular la frecuencia y probabilidad condicional de trigrams
def trigram_stats(tokens):
    trigrams = generate_ngrams(tokens, 3)
    trigram_freq = Counter(trigrams)
    context_freq = Counter([(trigram[0], trigram[1]) for trigram in trigrams])
    trigram_data = []
    
    for (term1, term2, term3), freq in trigram_freq.items():
        context_count = context_freq[(term1, term2)]
        prob = freq / context_count
        trigram_data.append([term1, term2, term3, freq, context_count, prob])
        
    return pd.DataFrame(trigram_data, columns=["Term 1", "Term 2", "Term 3", "Trigram Frequency", "Context Frequency", "Conditional Probability"])

# Configuración de la interfaz
st.title("Create Language Models")
st.header("Load Corpus and Generate N-grams")

# Botón para cargar el archivo
uploaded_file = st.file_uploader("Load corpus", type="csv")
if uploaded_file:
    corpus = load_corpus(uploaded_file)
    
    # Verificar si el corpus tiene datos
    if corpus.empty:
        st.error("The corpus is empty or not formatted as expected.")
    else:
        st.success("Corpus loaded successfully.")
        
        # Visualización del corpus original
        st.write("Original Corpus Sample:")
        st.dataframe(corpus.head())
        
        # Tokenización del corpus completo
        tokens = []
        for text in corpus['Text']:
            tokens.extend(tokenize_text(text))
        
        # Extraer el nombre base del archivo de corpus sin extensión
        corpus_name = os.path.splitext(uploaded_file.name)[0].replace("Corpus_", "")
        
        # Botones para generar bigramas y trigramas
        if st.button("Generate bigrams"):
            bigram_df = bigram_stats(tokens)
            bigram_df = bigram_df.reset_index(drop=True)  # Solo reiniciar el índice sin ordenar
            
            # Asegurar que los valores de frecuencia se muestren sin formato de miles
            bigram_df["Bigram Frequency"] = bigram_df["Bigram Frequency"].astype(int)
            bigram_df["Context Frequency"] = bigram_df["Context Frequency"].astype(int)
            
            # Mostrar bigramas
            st.write("Bigrams Table:")
            st.dataframe(bigram_df)
            
            # Guardar en CSV
            bigram_filename = f"Ngrams/Bigramas_{corpus_name}.csv"
            bigram_df.to_csv(bigram_filename, index=False)
            
            # Botón para descargar el archivo de bigramas
            st.download_button(label="Download bigrams CSV", data=open(bigram_filename, 'rb').read(), file_name=f"Bigramas_{corpus_name}.csv", mime='text/csv')

        if st.button("Generate trigrams"):
            trigram_df = trigram_stats(tokens)
            trigram_df = trigram_df.reset_index(drop=True)  # Solo reiniciar el índice sin ordenar
            
            # Asegurar que los valores de frecuencia se muestren sin formato de miles
            trigram_df["Trigram Frequency"] = trigram_df["Trigram Frequency"].astype(int)
            trigram_df["Context Frequency"] = trigram_df["Context Frequency"].astype(int)
            
            # Mostrar trigramas
            st.write("Trigrams Table:")
            st.dataframe(trigram_df)
            
            # Guardar en CSV
            trigram_filename = f"Ngrams/Trigramas_{corpus_name}.csv"
            trigram_df.to_csv(trigram_filename, index=False)
            
            # Botón para descargar el archivo de trigramas
            st.download_button(label="Download trigrams CSV", data=open(trigram_filename, 'rb').read(), file_name=f"Trigramas_{corpus_name}.csv", mime='text/csv')


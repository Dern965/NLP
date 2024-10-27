import streamlit as st 
import pandas as pd
import spacy
from collections import Counter
import csv

# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Función para leer el corpus desde un archivo CSV y omitir el encabezado
def read_corpus(file):
    df = pd.read_csv(file, header=0)  # Omitir el encabezado
    corpus = " ".join(df.iloc[:, 0].astype(str).tolist())  # Combina todo en un solo string
    return corpus

# Función para extraer bigramas y trigramas
def generate_ngrams(corpus, n=2):
    doc = nlp(corpus)
    ngrams = []
    for i in range(len(doc) - n + 1):
        ngram = tuple(doc[j].text.lower() for j in range(i, i + n))  # Convertir a minúsculas para evitar duplicados
        ngrams.append(ngram)
    return Counter(ngrams)

# Función para calcular la probabilidad condicional
def calculate_conditional_probability(ngrams, context_frequency):
    probabilities = {}
    for ngram, freq in ngrams.items():
        context = ngram[:-1]
        context_freq = context_frequency.get(context, 0)
        
        # Verificar los valores de frecuencia
        print(f"N-grama: {ngram}, Frecuencia del N-grama: {freq}, Frecuencia del Contexto: {context_freq}")
        
        if context_freq > 0:
            probabilities[ngram] = freq / context_freq
        else:
            probabilities[ngram] = 0
    return probabilities

# Generar archivo CSV de bigramas
def export_bigrams(corpus):
    bigrams = generate_ngrams(corpus, n=2)
    # Calcula las frecuencias de los contextos (primer término de cada bigrama)
    context_freq = Counter([bigram[:1] for bigram in bigrams])  # Corregir para obtener el contexto como tupla
    conditional_prob = calculate_conditional_probability(bigrams, context_freq)

    with open('bigrams.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Term 1", "Term 2", "Frequency of Bigram", "Frequency of Context (Term 1)", "Conditional Probability"])
        for bigram, freq in bigrams.items():
            context = bigram[:1]
            writer.writerow([bigram[0], bigram[1], freq, context_freq[context], round(conditional_prob[bigram], 6)])

    st.success("Archivo de bigramas exportado como 'bigrams.csv'")

# Generar archivo CSV de trigramas
def export_trigrams(corpus):
    trigrams = generate_ngrams(corpus, n=3)
    # Calcula las frecuencias de los contextos (primeros dos términos de cada trigrama)
    context_freq = Counter([trigram[:2] for trigram in trigrams])  # Corregir para obtener el contexto como tupla
    conditional_prob = calculate_conditional_probability(trigrams, context_freq)

    with open('trigrams.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Term 1", "Term 2", "Term 3", "Frequency of Trigram", "Frequency of Context (Bigram Term1 + Term2)", "Conditional Probability"])
        for trigram, freq in trigrams.items():
            context = trigram[:2]
            writer.writerow([trigram[0], trigram[1], trigram[2], freq, context_freq[context], round(conditional_prob[trigram], 6)])

    st.success("Archivo de trigramas exportado como 'trigrams.csv'")

# Interfaz de usuario en Streamlit
st.title("Generador de Bigrama y Trigrama")
uploaded_file = st.file_uploader("Cargar archivo de texto (CSV)", type=["csv"])

if uploaded_file:
    corpus = read_corpus(uploaded_file)
    
    if st.button("Generar Bigrama"):
        export_bigrams(corpus)
    
    if st.button("Generar Trigrama"):
        export_trigrams(corpus)

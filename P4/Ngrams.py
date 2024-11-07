import streamlit as st
import pandas as pd
import spacy
import os
from collections import Counter

# Carga el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Crear la carpeta 'Ngrams' si no existe
if not os.path.exists("Ngrams"):
    os.makedirs("Ngrams")

# Función para cargar y procesar el corpus
def load_corpus(df):
    sentences = []
    for mensaje in df['Mensaje']:
        # Procesar cada mensaje con spaCy y agregar los tokens con etiquetas de inicio y fin
        doc = nlp(mensaje)
        for sent in doc.sents:
            # Insertar las etiquetas de inicio "<" y fin ">"
            tokenized_sentence = ['<'] + [token.text.lower() for token in sent] + ['>']
            sentences.append(tokenized_sentence)
    return sentences

# Función para generar bigramas
def generate_bigrams(sentences):
    bigrams = []
    unigrams = []
    for sentence in sentences:
        unigrams.extend(sentence)
        bigrams.extend([(sentence[i], sentence[i + 1]) for i in range(len(sentence) - 1)])
    bigram_freq = Counter(bigrams)
    unigram_freq = Counter(unigrams)
    return bigram_freq, unigram_freq

def bigram_dataframe(bigram_freq, unigram_freq):
    bigram_data = []
    for bigram, freq in bigram_freq.items():
        term1, term2 = bigram
        context_freq = unigram_freq[term1]
        cond_prob = freq / context_freq
        bigram_data.append([term1, term2, freq, context_freq, cond_prob])

    bigram_df = pd.DataFrame(bigram_data, columns=["Term 1", "Term 2", "Frequency of Bigram",
                                                   "Frequency of Context (Term 1)", "Conditional Probability"])
    bigram_df = bigram_df.sort_values(by="Frequency of Bigram", ascending=False)
    return bigram_df

# Función para generar trigramas
def generate_trigrams(sentences):
    trigrams = []
    bigrams = []
    for sentence in sentences:
        bigrams.extend([(sentence[i], sentence[i + 1]) for i in range(len(sentence) - 1)])
        trigrams.extend([(sentence[i], sentence[i + 1], sentence[i + 2]) for i in range(len(sentence) - 2)])
    trigram_freq = Counter(trigrams)
    bigram_freq = Counter(bigrams)
    return trigram_freq, bigram_freq

def trigram_dataframe(trigram_freq, bigram_freq):
    trigram_data = []
    for trigram, freq in trigram_freq.items():
        term1, term2, term3 = trigram
        context_freq = bigram_freq[(term1, term2)]
        cond_prob = freq / context_freq
        trigram_data.append([term1, term2, term3, freq, context_freq, cond_prob])
    trigram_df = pd.DataFrame(trigram_data, columns=["Term 1", "Term 2", "Term 3", "Frequency of Trigram",
                                                     "Frequency of Context (Term 1, Term 2)", "Conditional Probability"])
    trigram_df = trigram_df.sort_values(by="Frequency of Trigram", ascending=False)
    return trigram_df

st.title("Create Language Models")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.write("The file is empty. Please upload a valid CSV file.")
    else:
        st.success("File uploaded successfully.")
        st.write(" Original Corpus Sample:")
        st.write(df.head())
        sentences = load_corpus(df)

        file_name = os.path.splitext(uploaded_file.name)[0].replace("Corpus_", "")

        if st.button("Generate Bigramas"):
            bigram_freq, unigram_freq = generate_bigrams(sentences)
            bigram_df = bigram_dataframe(bigram_freq, unigram_freq)

            st.write("Bigram DataFrame:")
            st.write(bigram_df)
            bigram_df.to_csv(f"Ngrams/Bigramas_{file_name}.csv", index=False)
            st.success("Bigram DataFrame saved successfully.")
            st.download_button(label="Download Bigram DataFrame", data=bigram_df.to_csv(), file_name=f"Bigramas_{file_name}.csv", mime="text/csv")
        
        if st.button("Generate Trigramas"):
            trigram_freq, bigram_freq = generate_trigrams(sentences)
            trigram_df = trigram_dataframe(trigram_freq, bigram_freq)

            st.write("Trigram DataFrame:")
            st.write(trigram_df)
            trigram_df.to_csv(f"Ngrams/Trigramas_{file_name}.csv", index=False)
            st.success("Trigram DataFrame saved successfully.")
            st.download_button(label="Download Trigram DataFrame", data=trigram_df.to_csv(), file_name=f"Trigramas_{file_name}.csv", mime="text/csv")
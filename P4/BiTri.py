import streamlit as st
import spacy
import pandas as pd
import csv
import os
from collections import Counter

# Función para generar bigramas #
def Generar_Bigramas(corpus, nombre_base):
    nlp = spacy.load('es_core_news_sm')

    # Paso 1: Tokenizar
    archivo_tokenizado = f'{nombre_base}_tokenizado.csv'
    with open(archivo_tokenizado, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mensaje Tokenizado'])
        for linea in corpus:
            doc = nlp(linea)
            tokens = [token.text for token in doc]
            writer.writerow([' '.join(tokens)])

    # Paso 2: Crear bigramas
    archivo_bigramas = f'{nombre_base}_bigramas.csv'
    with open(archivo_tokenizado, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        with open(archivo_bigramas, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['Bigramas'])
            for linea in reader:
                tokens = linea[0].split()
                bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                for bigram in bigrams:
                    writer.writerow([' '.join(bigram)])

    # Paso 3: Calcular frecuencia y probabilidad conjunta
    archivo_frecuencias_bigramas = f'{nombre_base}_frecuencia_bigrama.csv'
    with open(archivo_bigramas, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        bigramas = []
        tokens_bi = []
        for row in reader:
            bigrama = row[0]
            bigramas.append(bigrama)
            tokens_bi.append(row[0].split())
        freq_bigramas = Counter(bigramas)
        freq_total_bigramas = len(tokens_bi)

    with open(archivo_frecuencias_bigramas, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Bigrama", "Frecuencia", "Frecuencia Relativa"])
        for bigrama, freq in freq_bigramas.items():
            writer.writerow([bigrama, freq, freq / freq_total_bigramas])

    # Paso 4: Calcular la probabilidad condicional
    archivo_frecuencias_proba_bigramas = f'{nombre_base}_frecuencia_proba_bigrama.csv'
    with open(archivo_frecuencias_proba_bigramas, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Bigrama", "Frecuencia", "Probabilidad Conjunta"])
        for bigrama, freq in freq_bigramas.items():
            proba = freq / freq_total_bigramas
            writer.writerow([bigrama, freq, round(proba, 6)])

    # Paso 5: Juntar los datos
    archivo_bigramas_final = f'{nombre_base}_bigramas_final.csv'
    with open(archivo_frecuencias_proba_bigramas, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        datos_bigramas = [row for row in reader]

    new_data = [["Term 1", "Term 2", "Frequency of Bigram", "Frequency of Context", "Conditional Probability of Bigram"]]
    terms = [term for row in datos_bigramas for term in row[0].split()]
    frecuencia_terms = Counter(terms)

    for row in datos_bigramas:
        bigrama = row[0]
        frecuencia_bigrama = int(row[1])
        probabilidad_conjunta = float(row[2])
        term1, term2 = bigrama.split()
        frecuencia_term = frecuencia_terms[term1]
        new_data.append([term1, term2, frecuencia_bigrama, frecuencia_term, probabilidad_conjunta])

    new_data_sorted = sorted(new_data[1:], key=lambda x: x[2], reverse=True)
    new_data_sorted.insert(0, new_data[0])

    with open(archivo_bigramas_final, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_data_sorted)

    st.write(f"Bigramas generados y guardados en: {archivo_bigramas_final}")


# Función para generar trigramas #
def Generar_Trigramas(corpus, nombre_base):
    nlp = spacy.load('es_core_news_sm')

    # Paso 1: Tokenizar
    archivo_tokenizado = f'{nombre_base}_tokenizado.csv'
    with open(archivo_tokenizado, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mensaje Tokenizado'])
        for linea in corpus:
            doc = nlp(linea)
            tokens = [token.text for token in doc]
            writer.writerow([' '.join(tokens)])

    # Paso 2: Crear trigramas
    archivo_trigramas = f'{nombre_base}_trigramas.csv'
    with open(archivo_tokenizado, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        with open(archivo_trigramas, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['Trigramas'])
            for linea in reader:
                tokens = linea[0].split()
                trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
                for trigram in trigrams:
                    writer.writerow([' '.join(trigram)])

    # Paso 3: Calcular frecuencia y probabilidad conjunta
    archivo_frecuencias_trigramas = f'{nombre_base}_frecuencia_trigrama.csv'
    with open(archivo_trigramas, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        trigramas = []
        tokens_tri = []
        for row in reader:
            trigrama = row[0]
            trigramas.append(trigrama)
            tokens_tri.append(trigrama.split())
    freq_trigramas = Counter(trigramas)
    freq_total_trigramas = len(tokens_tri)

    with open(archivo_frecuencias_trigramas, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Trigrama", "Frecuencia", "Frecuencia Relativa"])
        for trigrama, freq in freq_trigramas.items():
            writer.writerow([trigrama, freq, freq / freq_total_trigramas])

    # Paso 4: Calcular la probabilidad condicional
    archivo_frecuencias_proba_trigramas = f'{nombre_base}_frecuencia_proba_trigrama.csv'
    with open(archivo_frecuencias_proba_trigramas, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Trigrama", "Frecuencia", "Probabilidad Conjunta"])
        for trigrama, freq in freq_trigramas.items():
            proba = freq / freq_total_trigramas
            writer.writerow([trigrama, freq, round(proba, 6)])

    # Paso 5: Juntar los datos
    archivo_trigramas_final = f'{nombre_base}_trigramas_final.csv'
    with open(archivo_frecuencias_proba_trigramas, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        datos_trigramas = []
        for row in reader:
            datos_trigramas.append(row)
    
    new_data=[]

    new_data.append(["Term 1", "Term 2", "Term 3", "Frequency of Trigram", "Conditional Probability of Trigram"])
    terms = [term for row in datos_trigramas for term in row[0].split()]
    frecuencia_terms = Counter(terms)

    for row in datos_trigramas:
        trigrama = row[0]
        frecuencia_trigrama = int(row[1])
        probabilidad_conjunta = float(row[2])

        term1, term2, term3 = trigrama.split()
        frecuencia_term1 = frecuencia_terms[term1]
        frecuencia_term2 = frecuencia_terms[term2]

        frecuencia_contexto = frecuencia_term1 + frecuencia_term2

        new_data.append([term1, term2, term3, frecuencia_trigrama, frecuencia_contexto, probabilidad_conjunta])

    new_data_sorted = sorted(new_data[1:], key=lambda x: x[3], reverse=True)
    new_data_sorted.insert(0, new_data[0])

    with open(archivo_trigramas_final, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_data_sorted)

    st.write(f"Trigramas generados y guardados en: {archivo_trigramas_final}")


# Interfaces #
def interface_language_models():
    st.title("Create Language Models")
    
    # Cargar Corpus
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)

        # Convertir el corpus a una lista de textos
        corpus = data.iloc[:, 0].tolist()
        nombre_base = os.path.splitext(uploaded_file.name)[0]  # Obtener nombre base sin extensión

        # Botón para generar bigramas
        if st.button("Generate bigrams"):
            Generar_Bigramas(corpus, nombre_base)
            st.success("Bigrams generated successfully!")
        
        # Botón para generar trigramas
        if st.button("Generate trigrams"):
            Generar_Trigramas(corpus, nombre_base)
            st.success("Trigrams generated successfully!")

# Página principal #
st.sidebar.title("Practica 4")

interface = st.sidebar.radio("Choose an Interface", ("Create Language Models", "Predictive Text", "Text Generation", "Conditional Probability"))

if interface == "Create Language Models":
    interface_language_models()

import streamlit as st
import csv
import spacy
from collections import Counter
import os

# Función para obtener el nombre base del archivo
def obtener_nombre_base(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]

# Función para procesar bigramas
def procesar_bigramas(uploaded_file, file_name):
    base_name = obtener_nombre_base(file_name)
    
    # Cargar el modelo de Spacy
    nlp = spacy.load('es_core_news_sm')
    
    # Paso 1: Tokenización y exportar corpus tokenizado
    tokenized_corpus_path = f'Corpus_telegram_tokenizado_{base_name}.csv'
    with open(tokenized_corpus_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mensaje Tokenizado"])
        for row in uploaded_file:
            mensaje = row.decode('utf-8').strip()  # Leer y decodificar la línea del archivo
            doc = nlp(mensaje)
            tokens = [token.text for token in doc]
            writer.writerow([" ".join(tokens)])

    # Paso 2: Generar bigramas
    bigramas_path = f'Corpus_telegram_bigramas_{base_name}.csv'
    bigrams_counter = Counter()  # Para almacenar la frecuencia de cada bigrama
    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar encabezado
        for row in reader:
            tokens = row[0].split()
            bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            bigrams_counter.update(bigrams)  # Actualizar el contador con los bigramas generados

    # Escribir los bigramas y sus frecuencias en un nuevo archivo CSV
    with open(bigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Bigrama", "Frecuencia"])
        for bigrama, freq in bigrams_counter.items():
            writer.writerow([' '.join(bigrama), freq])

    # Paso 3: Calcular frecuencias relativas y probabilidad condicional para los bigramas
    freq_bigramas_path = f'Frecuencias_bigramas_{base_name}.csv'
    freq_total_bigramas = sum(bigrams_counter.values())  # Total de bigramas

    with open(freq_bigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Bigrama", "Frecuencia", "Frecuencia Relativa"])
        for bigrama, freq in bigrams_counter.items():
            writer.writerow([' '.join(bigrama), freq, freq / freq_total_bigramas])

    # Paso 4: Calcular la probabilidad condicional
    prob_bigramas_path = f'Corpus_telegram_bigramas_final_{base_name}.csv'
    terms = [term for bigrama in bigrams_counter for term in bigrama]
    frecuencia_terms = Counter(terms)
    
    new_data = [["Term 1", "Term 2", "Frecuencia del Bigrama", "Frecuencia del Contexto", "Probabilidad Condicional"]]
    
    for bigrama, freq_bigrama in bigrams_counter.items():
        term1, term2 = bigrama
        frecuencia_term1 = frecuencia_terms[term1]
        probabilidad_conjunta = freq_bigrama / freq_total_bigramas
        new_data.append([term1, term2, freq_bigrama, frecuencia_term1, probabilidad_conjunta])

    new_data_sorted = sorted(new_data[1:], key=lambda x: x[2], reverse=True)
    new_data_sorted.insert(0, new_data[0])
    
    with open(prob_bigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_data_sorted)
    
    return {
        "tokenized_corpus": tokenized_corpus_path,
        "bigram_frequencies": freq_bigramas_path,
        "bigram_probabilities": prob_bigramas_path
    }

# Función para procesar trigramas
def procesar_trigramas(uploaded_file, file_name):
    base_name = obtener_nombre_base(file_name)
    
    # Cargar el modelo de Spacy
    nlp = spacy.load('es_core_news_sm')
    
    # Paso 1: Tokenización y exportar corpus tokenizado
    tokenized_corpus_path = f'Corpus_telegram_tokenizado_{base_name}.csv'
    with open(tokenized_corpus_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mensaje Tokenizado"])
        for row in uploaded_file:
            mensaje = row.decode('utf-8').strip()  # Leer y decodificar la línea del archivo
            doc = nlp(mensaje)
            tokens = [token.text for token in doc]
            writer.writerow([" ".join(tokens)])

    # Paso 2: Generar trigramas
    trigramas_path = f'Corpus_telegram_trigramas_{base_name}.csv'
    trigrams_counter = Counter()  # Para almacenar la frecuencia de cada trigrama
    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar encabezado
        for row in reader:
            tokens = row[0].split()
            trigramas = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
            trigrams_counter.update(trigramas)  # Actualizar el contador con los trigramas generados

    # Escribir los trigramas y sus frecuencias en un nuevo archivo CSV
    with open(trigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Trigrama", "Frecuencia"])
        for trigrama, freq in trigrams_counter.items():
            writer.writerow([' '.join(trigrama), freq])

    # Paso 3: Calcular frecuencias relativas y probabilidad condicional para los trigramas
    freq_trigramas_path = f'Frecuencias_trigramas_{base_name}.csv'
    freq_total_trigramas = sum(trigrams_counter.values())  # Total de trigramas

    with open(freq_trigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Trigrama", "Frecuencia", "Frecuencia Relativa"])
        for trigrama, freq in trigrams_counter.items():
            writer.writerow([' '.join(trigrama), freq, freq / freq_total_trigramas])

    # Paso 4: Calcular la probabilidad condicional
    prob_trigramas_path = f'Corpus_telegram_trigramas_final_{base_name}.csv'
    terms = [term for trigrama in trigrams_counter for term in trigrama]
    frecuencia_terms = Counter(terms)
    
    new_data = [["Term 1", "Term 2", "Term 3", "Frecuencia del Trigrama", "Frecuencia del Contexto", "Probabilidad Condicional"]]
    
    for trigrama, freq_trigrama in trigrams_counter.items():
        term1, term2, term3 = trigrama
        frecuencia_term1 = frecuencia_terms[term1]
        frecuencia_term2 = frecuencia_terms[term2]
        frecuencia_contexto = frecuencia_term1 + frecuencia_term2  # Calcular frecuencia del contexto (2 términos anteriores)
        probabilidad_conjunta = freq_trigrama / freq_total_trigramas
        new_data.append([term1, term2, term3, freq_trigrama, frecuencia_contexto, probabilidad_conjunta])

    new_data_sorted = sorted(new_data[1:], key=lambda x: x[3], reverse=True)
    new_data_sorted.insert(0, new_data[0])
    
    with open(prob_trigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_data_sorted)
    
    return {
        "tokenized_corpus": tokenized_corpus_path,
        "trigram_frequencies": freq_trigramas_path,
        "trigram_probabilities": prob_trigramas_path
    }

# Interfaz de Streamlit
st.title("Procesamiento de Bigramas y Trigramas")

# Cargar el archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    file_name = uploaded_file.name  # Nombre del archivo subido

    # Mostrar los botones
    if st.button('Generar Bigramas'):
        result = procesar_bigramas(uploaded_file, file_name)
        st.write("Corpus Tokenizado:", result["tokenized_corpus"])
        st.write("Frecuencias de Bigramas:", result["bigram_frequencies"])

    if st.button('Generar Trigramas'):
        result = procesar_trigramas(uploaded_file, file_name)
        st.write("Corpus Tokenizado:", result["tokenized_corpus"])
        st.write("Frecuencias de Trigramas:", result["trigram_frequencies"])



import streamlit as st
import csv
import spacy
from collections import Counter
import os

# Función para obtener el nombre base del archivo
def obtener_nombre_base(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]

# Función para procesar bigramas y generar un solo archivo de probabilidades condicionales
def procesar_bigramas(uploaded_file, file_name):
    base_name = obtener_nombre_base(file_name)
    
    # Cargar el modelo de Spacy
    nlp = spacy.load('es_core_news_sm')
    
    # Tokenización y procesamiento del corpus
    tokens_list = []
    for row in uploaded_file:
        mensaje = row.decode('utf-8').strip()  # Leer y decodificar la línea del archivo
        doc = nlp(mensaje)
        tokens = [token.text for token in doc]
        tokens_list.append(tokens)

    # Generar bigramas
    bigrams_counter = Counter()  # Para almacenar la frecuencia de cada bigrama
    for tokens in tokens_list:
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        bigrams_counter.update(bigrams)  # Actualizar el contador con los bigramas generados

    # Calcular frecuencias relativas y probabilidad condicional
    freq_total_bigramas = sum(bigrams_counter.values())  # Total de bigramas
    terms = [term for bigrama in bigrams_counter for term in bigrama]
    frecuencia_terms = Counter(terms)
    
    # Crear el archivo de salida con probabilidades condicionales
    prob_bigramas_path = f'Corpus_telegram_bigramas_final_{base_name}.csv'
    new_data = [["Term 1", "Term 2", "Frecuencia del Bigrama", "Frecuencia del Contexto", "Probabilidad Condicional"]]
    
    for bigrama, freq_bigrama in bigrams_counter.items():
        term1, term2 = bigrama
        frecuencia_term1 = frecuencia_terms[term1]
        probabilidad_conjunta = freq_bigrama / freq_total_bigramas
        new_data.append([term1, term2, freq_bigrama, frecuencia_term1, probabilidad_conjunta])

    # Guardar el archivo CSV con las probabilidades condicionales
    with open(prob_bigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_data)
    
    return prob_bigramas_path

# Función para procesar trigramas y generar un solo archivo de probabilidades condicionales
def procesar_trigramas(uploaded_file, file_name):
    base_name = obtener_nombre_base(file_name)
    
    # Cargar el modelo de Spacy
    nlp = spacy.load('es_core_news_sm')
    
    # Tokenización y procesamiento del corpus
    tokens_list = []
    for row in uploaded_file:
        mensaje = row.decode('utf-8').strip()  # Leer y decodificar la línea del archivo
        doc = nlp(mensaje)
        tokens = [token.text for token in doc]
        tokens_list.append(tokens)

    # Generar trigramas
    trigrams_counter = Counter()  # Para almacenar la frecuencia de cada trigrama
    for tokens in tokens_list:
        trigramas = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
        trigrams_counter.update(trigramas)  # Actualizar el contador con los trigramas generados

    # Calcular frecuencias relativas y probabilidad condicional
    freq_total_trigramas = sum(trigrams_counter.values())  # Total de trigramas
    terms = [term for trigrama in trigrams_counter for term in trigrama]
    frecuencia_terms = Counter(terms)
    
    # Crear el archivo de salida con probabilidades condicionales
    prob_trigramas_path = f'Corpus_telegram_trigramas_final_{base_name}.csv'
    new_data = [["Term 1", "Term 2", "Term 3", "Frecuencia del Trigrama", "Frecuencia del Contexto", "Probabilidad Condicional"]]
    
    for trigrama, freq_trigrama in trigrams_counter.items():
        term1, term2, term3 = trigrama
        frecuencia_term1 = frecuencia_terms[term1]
        frecuencia_term2 = frecuencia_terms[term2]
        frecuencia_contexto = frecuencia_term1 + frecuencia_term2  # Calcular frecuencia del contexto (2 términos anteriores)
        probabilidad_conjunta = freq_trigrama / freq_total_trigramas
        new_data.append([term1, term2, term3, freq_trigrama, frecuencia_contexto, probabilidad_conjunta])

    # Guardar el archivo CSV con las probabilidades condicionales
    with open(prob_trigramas_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_data)
    
    return prob_trigramas_path

# Interfaz de Streamlit
st.title("Procesamiento de Bigramas y Trigramas")

# Cargar el archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    file_name = uploaded_file.name  # Nombre del archivo subido

    # Mostrar los botones
    if st.button('Generar Bigramas'):
        prob_file_path = procesar_bigramas(uploaded_file, file_name)
        st.write(f"Archivo generado: {prob_file_path}")

    if st.button('Generar Trigramas'):
        prob_file_path = procesar_trigramas(uploaded_file, file_name)
        st.write(f"Archivo generado: {prob_file_path}")

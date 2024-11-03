import streamlit as st
import csv
import random
import math
from io import StringIO

# Cargar el archivo de bigramas o trigramas en un diccionario
def cargar_corpus(file, ngram_type):
    corpus = {}
    file_content = StringIO(file.getvalue().decode("utf-8"))
    reader = csv.reader(file_content)
    next(reader)  # Saltar la cabecera
    for row in reader:
        if ngram_type == "bigram":
            context = (row[0],)
            word = row[1]
            probability = float(row[4])
        elif ngram_type == "trigram":
            context = (row[0], row[1])
            word = row[2]
            probability = float(row[5])
        else:
            raise ValueError("El tipo de ngrama debe ser 'bigram' o 'trigram'")

        if context not in corpus:
            corpus[context] = []
        corpus[context].append((word, probability))
    return corpus

# Selección de palabra según la ruleta, con un pequeño ajuste para mayor aleatoriedad
def seleccion_ruleta(word_list):
    total = sum(prob for _, prob in word_list)
    pick = random.uniform(0, total)
    current = 0
    # Añadir un pequeño ajuste aleatorio para palabras con probabilidades similares
    jitter = 0.02 * total  # Añadir un 2% de variabilidad para evitar selecciones repetitivas
    pick = min(total, max(0, pick + random.uniform(-jitter, jitter)))
    
    for word, prob in word_list:
        current += prob
        if current >= pick:
            return word
    return word_list[-1][0]

# Generación automática de oración basada en el contexto
def generar_oracion(corpus, ngram_type):
    sentence = ["|"]  # Comenzar siempre con el marcador de inicio
    context = ("|",) if ngram_type == 2 else ("|", "|")  # Contexto inicial para bigramas o trigramas

    while True:
        next_words = corpus.get(context)
        if not next_words:
            break
        next_word = seleccion_ruleta(next_words)
        if next_word == "|":
            break  # Terminar si se encuentra el marcador de fin de oración
        sentence.append(next_word)
        context = (sentence[-1],) if ngram_type == 2 else (sentence[-2], sentence[-1])

    return ' '.join(sentence[1:])  # Excluir el marcador de inicio en la salida

# Aplicar Laplace smoothing y calcular probabilidad conjunta
def calcular_probabilidad_conjunta(corpus, sentence, ngram_type):
    words = sentence.split()
    log_prob = 0
    vocab_size = len(set(word for context in corpus for word, _ in corpus[context]))  # Tamaño del vocabulario

    for i in range(len(words) - (ngram_type - 1)):
        context = tuple(words[i:i + (ngram_type - 1)])
        word = words[i + (ngram_type - 1)]
        word_list = corpus.get(context, [])
        count_word = sum(1 for w, _ in word_list if w == word)
        count_context = len(word_list)
        
        # Aplicar Laplace smoothing
        prob = (count_word + 1) / (count_context + vocab_size)
        log_prob += math.log(prob)

    return math.exp(log_prob)  # Convertir la suma de logaritmos a probabilidad conjunta

# Interfaz de Streamlit
st.title("Automatic Text Generation with Conditional Probability")
st.header("Load Corpus and Generate Sentence Automatically")

# Cargar el archivo de bigramas o trigramas
uploaded_file = st.file_uploader("Load bigram or trigram corpus", type="csv")

# Detectar el tipo de n-grama basado en el nombre del archivo
if uploaded_file:
    filename = uploaded_file.name.lower()
    if "bigram" in filename:
        ngram_type = 2  # Asignar 2 para bigramas
    elif "trigram" in filename:
        ngram_type = 3  # Asignar 3 para trigramas
    else:
        st.error("The file name must contain 'bigram' or 'trigram' to detect the n-gram type.")
        st.stop()  # Detener la ejecución si no se puede detectar el tipo de n-grama
    
    # Cargar el corpus basado en el tipo detectado
    corpus = cargar_corpus(uploaded_file, ngram_type="bigram" if ngram_type == 2 else "trigram")
    st.success(f"Corpus loaded successfully as {'bigram' if ngram_type == 2 else 'trigram'}.")

    if st.button("Generate sentence"):
        # Generar la oración automáticamente desde el inicio
        generated_sentence = generar_oracion(corpus, ngram_type=ngram_type)
        st.write("Generated sentence:", generated_sentence)

        # Calcular la probabilidad conjunta
        joint_prob = calcular_probabilidad_conjunta(corpus, generated_sentence, ngram_type=ngram_type)
        st.write("Joint probability of the generated sentence:", joint_prob)

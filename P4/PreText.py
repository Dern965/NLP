import streamlit as st
import pandas as pd

# Función para cargar el archivo y determinar si es bigramas o trigramas
def load_ngrams(file):
    df = pd.read_csv(file)
    if "Term 3" in df.columns:
        n = 3  # Es un archivo de trigramas
        df = df[["Term 1", "Term 2", "Term 3", "Frequency"]]
    else:
        n = 2  # Es un archivo de bigramas
        df = df[["Term 1", "Term 2", "Frequency"]]
    return df, n

# Función para predecir la siguiente palabra
def predict_next_word(ngrams, context, n, top_k=3):
    if n == 2:
        # Filtrar bigramas por el contexto
        candidates = ngrams[ngrams["Term 1"] == context]
        candidates = candidates[["Term 2", "Frequency"]].sort_values(by="Frequency", ascending=False)
        return candidates.head(top_k)["Term 2"].tolist()
    elif n == 3:
        # Filtrar trigramas por el contexto
        term1, term2 = context.split()
        candidates = ngrams[(ngrams["Term 1"] == term1) & (ngrams["Term 2"] == term2)]
        candidates = candidates[["Term 3", "Frequency"]].sort_values(by="Frequency", ascending=False)
        return candidates.head(top_k)["Term 3"].tolist()

# Interfaz de usuario
st.title("Texto Predictivo")

# Cargar archivo CSV de bigramas o trigramas
uploaded_file = st.file_uploader("Cargar archivo de bigramas o trigramas", type=["csv"])
if uploaded_file:
    ngrams, n = load_ngrams(uploaded_file)
    st.write(f"Archivo cargado correctamente. Detectado como {'bigrama' if n == 2 else 'trigrama'}.")

    # Input inicial para comenzar la predicción
    initial_word = st.text_input("Escribe una palabra inicial para comenzar la predicción de texto:")

    # Espacio para el texto generado
    generated_text = st.empty()
    generated_sequence = []

    # Estado de la predicción actual
    if "context" not in st.session_state:
        st.session_state["context"] = initial_word
        st.session_state["generated_text"] = initial_word

    # Botón para predecir la siguiente palabra
    if st.button("Siguiente palabra"):
        context = st.session_state["context"]
        # Obtener las 3 palabras más probables
        next_words = predict_next_word(ngrams, context, n)

        # Agregar opción para terminar la secuencia con un punto
        next_words.append(".")
        
        # Desplegar opciones de palabras siguientes
        selected_word = st.selectbox("Selecciona la siguiente palabra:", next_words)
        
        # Actualizar el contexto y el texto generado
        if selected_word == ".":
            st.write("Predicción finalizada.")
        else:
            st.session_state["generated_text"] += " " + selected_word
            generated_sequence.append(selected_word)
            if n == 2:
                st.session_state["context"] = selected_word  # Para bigramas, solo usamos una palabra
            elif n == 3:
                # Para trigramas, mantenemos el contexto de dos palabras
                st.session_state["context"] = f"{context.split()[-1]} {selected_word}"

    # Mostrar texto generado hasta el momento
    generated_text.write("Texto generado: " + st.session_state["generated_text"])

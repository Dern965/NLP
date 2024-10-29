import streamlit as st
import pandas as pd

# Función para cargar el archivo de bigramas o trigramas
def load_language_model(file):
    return pd.read_csv(file)

# Función para predecir las siguientes palabras
def predict_next_words(df, current_text, n_grams=2):
    words = current_text.strip().split()
    # Generar el contexto considerando bigramas o trigramas
    context = tuple(words[-(n_grams-1):]) if len(words) >= n_grams-1 else tuple(words)
    
    # Filtrar el DataFrame para obtener sugerencias según el contexto
    suggestions = df[df.iloc[:, :n_grams-1].apply(tuple, axis=1) == context]
    
    # Ordenar por probabilidad condicional y seleccionar las 3 palabras más probables
    suggestions = suggestions.sort_values(by="Conditional Probability", ascending=False).head(3)
    
    # Devolver las palabras sugeridas y el punto (".") para terminar la secuencia
    return list(suggestions.iloc[:, n_grams-1]) + ['.'] if not suggestions.empty else ['.']

# Configuración de la interfaz
st.title("Predictive Text")
st.header("Load Language Model")

# Cargar el modelo de bigramas o trigramas
uploaded_file = st.file_uploader("Load language model", type="csv", key="language_model")
if uploaded_file:
    df = load_language_model(uploaded_file)
    st.success("Language model loaded successfully.")
    
    # Detectar el tipo de modelo (bigramas o trigramas)
    n_grams = 2 if "Bigram" in uploaded_file.name else 3
    st.write(f"Model type detected: {'Bigram' if n_grams == 2 else 'Trigram'}")

    # Inicializar variables en session_state si no existen
    if "generated_text" not in st.session_state:
        st.session_state["generated_text"] = ""
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []
    if "first_time" not in st.session_state:
        st.session_state["first_time"] = True

    # Entrada inicial para comenzar la frase
    st.header("Write a word (or two words to start a sentence)")
    input_text = st.text_input("Start with a word:", "")

    # Al hacer clic en "Next word", establece la palabra inicial y genera predicciones
    if st.button("Next word"):
        if st.session_state["first_time"]:
            st.session_state["generated_text"] = input_text
            st.session_state["first_time"] = False
        st.session_state["predictions"] = predict_next_words(df, st.session_state["generated_text"], n_grams)
    
    # Mostrar las opciones de palabras en un selectbox
    if st.session_state["predictions"]:
        selected_word = st.selectbox("3 most probable words", st.session_state["predictions"], key="select_word")
        
        # Botón para añadir la palabra seleccionada
        if st.button("Add word"):
            if selected_word == '.':
                st.write("Sentence generation finished.")
            else:
                st.session_state["generated_text"] += f" {selected_word}"
                # Actualizar las predicciones para la nueva secuencia de texto
                st.session_state["predictions"] = predict_next_words(df, st.session_state["generated_text"], n_grams)

    # Mostrar el texto generado hasta ahora
    st.text_area("Generated text", st.session_state["generated_text"], height=200)

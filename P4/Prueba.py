import streamlit as st
import pandas as pd
from collections import defaultdict, Counter

# Función para cargar el archivo de bigramas o trigramas
def load_model(file):
    df = pd.read_csv(file)
    return df

# Función para predecir las siguientes palabras basado en bigramas/trigramas
def predict_next_word(model, input_words, n=3):
    # Usar defaultdict para almacenar las frecuencias de los bigramas/trigramas
    predictions = defaultdict(Counter)
    
    # Separar las palabras del input
    input_tokens = input_words.split()

    if len(input_tokens) == 1:  # Bigramas
        term1 = input_tokens[0]
        for _, row in model.iterrows():
            bigram = row[0].split()
            if len(bigram) < 2:
                continue  # Ignorar líneas con bigramas incompletos
            if bigram[0] == term1:
                predictions[term1][bigram[1]] += row[2]  # Asumimos que row[2] es la probabilidad o frecuencia
    elif len(input_tokens) == 2:  # Trigramas
        term1, term2 = input_tokens
        for _, row in model.iterrows():
            trigram = row[0].split()
            if len(trigram) < 3:
                continue  # Ignorar líneas con trigramas incompletos
            if trigram[0] == term1 and trigram[1] == term2:
                predictions[f"{term1} {term2}"][trigram[2]] += row[2]

    # Obtener las top N predicciones
    if len(predictions) > 0:
        return predictions[input_words].most_common(n)
    else:
        return None

# Función principal para ejecutar la aplicación
def main():
    st.title("Predictive Text")
    
    # Cargar el archivo de bigramas o trigramas
    uploaded_file = st.file_uploader("Load your bigram or trigram model (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        st.success("Model loaded successfully.")
        
        # Cargar el modelo como DataFrame
        model = load_model(uploaded_file)
        
        # Seleccionar el tipo de modelo (bigram o trigram)
        model_type = st.selectbox("Select the kind of feature to extract", ["Bigram", "Trigram"])
        
        # Escribir palabra(s) para comenzar la predicción
        input_words = st.text_input("Write a word (or two words to start a sentence)")
        
        if st.button("Next word"):
            if input_words:
                # Mostrar las 3 palabras más probables
                predictions = predict_next_word(model, input_words)
                
                if predictions:
                    st.write("The 3 most probable words are:")
                    for word, prob in predictions:
                        st.write(f"{word} (Probability: {prob})")
                    st.write("Select one of the words or a dot (.) to finish the sentence.")
                else:
                    st.warning("No predictions available. Try with a different word or input more text.")
            
            else:
                st.warning("Please enter a word to start the prediction.")
        
        # Añadir la palabra seleccionada
        selected_word = st.selectbox("3 most probable words", ["", ".", "Word1", "Word2", "Word3"])  # Reemplazar con la lista de palabras
        if st.button("Add word"):
            if selected_word == ".":
                st.write("Sentence finished.")
            else:
                st.write(f"You have added the word: {selected_word}")
                input_words += " " + selected_word
                st.write(f"Current sentence: {input_words}")

        # Mostrar el texto generado hasta el momento
        st.text_area("Generated text", value=input_words, height=200)

if __name__ == "__main__":
    main()

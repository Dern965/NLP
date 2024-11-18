import streamlit as st
import pandas as pd

# Variables globales
ngram_type = "bigrams"
bigram_df = None
trigram_df = None
generated_sentence = []  
context = ''
frase = []  

def load_language_model(file):
    global bigram_df, trigram_df, ngram_type, generated_sentence, frase
    bigram_df, trigram_df = load_ngrams(file)
    if bigram_df is not None:
        ngram_type = 'bigrams'
    elif trigram_df is not None:
        ngram_type = 'trigrams'
    generated_sentence.clear() 
    frase.clear()  

def load_ngrams(file):
    try:
        df = pd.read_csv(file)
        if 'Term 1' in df.columns and 'Term 2' in df.columns and 'Frequency of Bigram' in df.columns:
            st.write("Bigram file loaded successfully.")
            return df, None
        elif 'Term 1' in df.columns and 'Term 2' in df.columns and 'Term 3' in df.columns:
            st.write("Trigram file loaded successfully.")
            return None, df
        else:
            st.error("Invalid file structure. Please select a valid bigram or trigram file.")
            return None, None
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None, None

def predict_next_words(start_text):
    if not start_text:
        return

    words = start_text.strip().split()
    if (ngram_type == 'bigrams' and len(words) != 1) or (ngram_type == 'trigrams' and len(words) > 2):
        st.warning(f"Please enter {'1 word' if ngram_type == 'bigrams' else '1 or 2 words'} to continue.")
        return

    if ngram_type == 'bigrams' and bigram_df is not None:
        ngrams = bigram_df[bigram_df['Term 1'] == words[0]]
    elif ngram_type == 'trigrams' and trigram_df is not None:
        if len(words) == 1:
            ngrams = trigram_df[trigram_df['Term 1'] == words[0]]
        else:
            ngrams = trigram_df[(trigram_df['Term 1'] == words[0]) & (trigram_df['Term 2'] == words[1])]
    else:
        st.warning("No n-grams found for the given context.")
        return

    if ngrams.empty:
        st.warning("No suggestions available for the current context.")
        return

    sorted_ngrams = ngrams.sort_values(by='Frequency of Bigram' if ngram_type == 'bigrams' else 'Frequency of Trigram', ascending=False).head(3)
    top_3_words = sorted_ngrams['Term 2' if ngram_type == 'bigrams' else 'Term 3'].tolist()
    top_3_words.append('.')
    st.session_state['predicted_words'] = top_3_words

def add_word(next_word):
    if next_word and next_word != '3 most probable words':
        if next_word == '.':
            frase.append('.')  
        else:
            frase.append(next_word)  
            predict_next_words(next_word)
        st.session_state['temp_input'] = next_word

st.title("Predictive Text")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    load_language_model(uploaded_file)
    st.success("File uploaded successfully.")

    if 'temp_input' not in st.session_state:
        st.session_state['temp_input'] = ""

    start_text = st.text_input("Write a word (or two words to start a sentence)", value=st.session_state['temp_input'], key="start_text_input")
    if st.button("Next word"):
        predict_next_words(start_text)
        if start_text not in frase:
            frase.append(start_text)
        st.session_state['temp_input'] = start_text

    predicted_words = st.session_state.get('predicted_words', ["3 most probable words"])
    next_word = st.selectbox("Choose the next word", options=predicted_words, key="selected_word")
    if st.button("Add word"):
        add_word(next_word)

    st.write("Generated sentence:", ' '.join(frase))

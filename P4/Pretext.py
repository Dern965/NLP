import streamlit as st
import pandas as pd

# Global variables
ngram_type = "bigrams"
bigram_df = None
trigram_df = None
generated_sentence = []  # This will store the entire sentence history

def load_language_model(file):
    global bigram_df, trigram_df, ngram_type, generated_sentence
    bigram_df, trigram_df = load_ngrams(file)
    if bigram_df is not None:
        ngram_type = 'bigrams'
    elif trigram_df is not None:
        ngram_type = 'trigrams'
    generated_sentence = []  # Reset sentence history when a new model is loaded

def load_ngrams(file):
    try:
        df = pd.read_csv(file)
        if 'Term 1' in df.columns and 'Term 2' in df.columns and 'Frequency of Bigram' in df.columns:
            st.write("Bigram file loaded successfully.")
            return df, None
        elif 'Term 1' in df.columns and 'Term 2' in df.columns and 'Term 3' in df.columns and 'Frequency of Trigram' in df.columns:
            st.write("Trigram file loaded successfully.")
            return None, df
        else:
            st.error("Invalid file structure. Please select a valid bigram or trigram file.")
            return None, None
    except pd.errors.EmptyDataError:
        st.error("The file is empty. Please select a valid CSV file.")
        return None, None
    except pd.errors.ParserError:
        st.error("Error parsing the file. Please ensure it is a valid CSV.")
        return None, None
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None, None

def predict_next_words(start_text):
    if not start_text:
        return

    # Add the initial word to the generated sentence if it's the first word
    if not generated_sentence:
        generated_sentence.append(start_text)
        st.session_state['generated_text'] = ' '.join(generated_sentence)

    # Determine n-grams based on input length
    words = start_text.strip().split()
    if (ngram_type == 'bigrams' and len(words) != 1) or (ngram_type == 'trigrams' and len(words) != 2):
        st.warning(f"Please enter {'1 word' if ngram_type == 'bigrams' else '2 words'} to continue.")
        return

    # Fetch n-grams based on the selected model type
    if ngram_type == 'bigrams' and bigram_df is not None:
        ngrams = bigram_df[bigram_df['Term 1'] == words[0]]
    elif ngram_type == 'trigrams' and trigram_df is not None:
        ngrams = trigram_df[(trigram_df['Term 1'] == words[0]) & (trigram_df['Term 2'] == words[1])]
    else:
        st.warning("No n-grams found for the given context.")
        return

    if ngrams.empty:
        st.warning("No suggestions available for the current context.")
        return

    # Sort and select the top 3 suggestions
    sorted_ngrams = ngrams.sort_values(by='Frequency of Bigram' if ngram_type == 'bigrams' else 'Frequency of Trigram', ascending=False).head(3)
    top_3_words = sorted_ngrams['Term 2' if ngram_type == 'bigrams' else 'Term 3'].tolist()
    top_3_words.append('.')  # Add period option to end the sentence

    # Update session state with predictions
    st.session_state['predicted_words'] = top_3_words

def add_word(next_word):
    global generated_sentence
    if next_word and next_word != '3 most probable words':
        # If period is selected, complete the sentence
        if next_word == '.':
            generated_sentence.append('.')  # Add period to sentence history
            st.session_state['generated_text'] = ' '.join(generated_sentence)
        else:
            generated_sentence.append(next_word)  # Add selected word to sentence history
            st.session_state['generated_text'] = ' '.join(generated_sentence)
            # Set the new word as the next input in the "Write a word" field
            st.session_state['start_text_input'] = next_word
            # Refresh suggestions based on the new input word
            predict_next_words(next_word)

st.title("Predictive Text")

# Load language model
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    load_language_model(uploaded_file)

# Initialize session state for text input and generated text
if 'start_text_input' not in st.session_state:
    st.session_state['start_text_input'] = ""
if 'generated_text' not in st.session_state:
    st.session_state['generated_text'] = ""

# Input field for the start word
start_text = st.text_input("Write a word (or two words to start a sentence)", value=st.session_state['start_text_input'])
if st.button("Next word"):
    predict_next_words(start_text)

# Dropdown for predicted words
predicted_words = st.session_state.get('predicted_words', ["3 most probable words"])
next_word = st.selectbox("Choose the next word", options=predicted_words, key="selected_word")
if st.button("Add word"):
    add_word(next_word)

# Display the generated sentence with the full history
st.text_area("Generated text", value=st.session_state['generated_text'], height=150)

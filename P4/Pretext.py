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
    generated_sentence.clear()  # Reset sentence history when a new model is loaded

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

    # Split the input text to handle both bigram and trigram cases
    words = start_text.strip().split()

    # For bigrams, we require exactly one word; for trigrams, one or two words
    if (ngram_type == 'bigrams' and len(words) != 1) or (ngram_type == 'trigrams' and len(words) > 2):
        st.warning(f"Please enter {'1 word' if ngram_type == 'bigrams' else '1 or 2 words'} to continue.")
        return

    # Fetch n-grams based on the selected model type and current input
    if ngram_type == 'bigrams' and bigram_df is not None:
        ngrams = bigram_df[bigram_df['Term 1'] == words[0]]
    elif ngram_type == 'trigrams' and trigram_df is not None:
        if len(words) == 1:
            # If only one word is given for trigram, fetch suggestions based on the first word
            ngrams = trigram_df[trigram_df['Term 1'] == words[0]]
        else:
            # If two words are given, fetch based on both words for trigrams
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
        else:
            generated_sentence.append(next_word)  # Add selected word to sentence history
            # Call predict_next_words with the new word without modifying start_text_input directly
            predict_next_words(next_word)
            # Set the temporary variable to display the next word in the input field
            st.session_state['temp_input'] = next_word

    # Update the generated text area to show the full sentence
    st.session_state['generated_text'] += ' '.join(generated_sentence)

st.title("Predictive Text")

# Load language model
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    load_language_model(uploaded_file)
    st.success("File uploaded successfully.")

    # Initialize session state for text input and generated text
    if 'temp_input' not in st.session_state:
        st.session_state['temp_input'] = ""
    if 'generated_text' not in st.session_state:
        st.session_state['generated_text'] = ""

    # Input field for the start word, with a default value based on temp_input
    start_text = st.text_input("Write a word (or two words to start a sentence)", value=st.session_state['temp_input'], key="start_text_input")
    if st.button("Next word"):
        predict_next_words(start_text)
        # Add the manually entered word to the generated_sentence if it's not already added
        if not generated_sentence or generated_sentence[-1] != start_text:
            generated_sentence.append(start_text)
        # Update generated text with the new word added
        st.session_state['generated_text'] = ' '.join(generated_sentence)
        # Reset temp_input to prevent overwriting in the next run
        st.session_state['temp_input'] = start_text

    # Dropdown for predicted words
    predicted_words = st.session_state.get('predicted_words', ["3 most probable words"])
    next_word = st.selectbox("Choose the next word", options=predicted_words, key="selected_word")
    if st.button("Add word"):
        add_word(next_word)

    # Display the generated sentence with the full history
    st.text_area("Generated text", value=st.session_state['generated_text'], height=150)

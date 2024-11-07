import streamlit as st
import pandas as pd
import random

# Variables to store the DataFrames for bigrams and trigrams
bigram_df = None
trigram_df = None

# Function to load bigrams or trigrams from a file
def load_ngrams(file):
    try:
        # Verify file type
        if not file.name.endswith(".csv"):
            st.error("Invalid file type. Please select a CSV file.")
            return None, None

        # Read the CSV file
        df = pd.read_csv(file)

        # Check for bigrams
        if {'Term 1', 'Term 2', 'Frequency of Bigram', 'Frequency of Context (Term 1)'}.issubset(df.columns):
            st.success("Bigram file loaded successfully.")
            return df, None

        # Check for trigrams
        elif {'Term 1', 'Term 2', 'Term 3', 'Frequency of Trigram', 'Frequency of Context (Term 1, Term 2)'}.issubset(df.columns):
            st.success("Trigram file loaded successfully.")
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

# Function to get start n-grams
def get_start_ngrams(ngram_type, bigram_df, trigram_df):
    if ngram_type == 'bigrams':
        return bigram_df[bigram_df['Term 1'] == '<']
    elif ngram_type == 'trigrams':
        return trigram_df[(trigram_df['Term 1'] == '<') & (trigram_df['Term 2'] != '>')]

# Roulette wheel selection for choosing next n-gram
def roulette_wheel_selection(ngrams_df):
    if ngrams_df.empty:
        return None
    total_frequency = ngrams_df['Frequency of Bigram' if 'Frequency of Bigram' in ngrams_df.columns else 'Frequency of Trigram'].sum()
    pick = random.uniform(0, total_frequency)
    current = 0
    for _, row in ngrams_df.iterrows():
        current += row['Frequency of Bigram' if 'Frequency of Bigram' in row else 'Frequency of Trigram']
        if current > pick:
            return row

# Function to generate text using n-grams
def generate_text(ngram_type, bigram_df, trigram_df):
    sentence = []
    start_ngrams = get_start_ngrams(ngram_type, bigram_df, trigram_df)
    selected_ngram = roulette_wheel_selection(start_ngrams)

    if ngram_type == 'bigrams':
        word = selected_ngram['Term 2']
        sentence.append(word)
        context = word

        while word != '>':
            next_ngrams = bigram_df[bigram_df['Term 1'] == context]
            selected_ngram = roulette_wheel_selection(next_ngrams)
            if selected_ngram is None:
                break
            word = selected_ngram['Term 2']
            sentence.append(word)
            context = word

    elif ngram_type == 'trigrams':
        word1, word2 = selected_ngram['Term 2'], selected_ngram['Term 3']
        sentence.extend([word1, word2])
        context = (word1, word2)

        while word2 != '>':
            next_ngrams = trigram_df[(trigram_df['Term 1'] == context[0]) & (trigram_df['Term 2'] == context[1])]
            selected_ngram = roulette_wheel_selection(next_ngrams)
            if selected_ngram is None:
                break
            word1, word2 = selected_ngram['Term 2'], selected_ngram['Term 3']
            sentence.append(word2)
            context = (word1, word2)

    return ' '.join([w for w in sentence if w not in ['<', '>']])

# Streamlit interface
st.title("Text Generation using N-grams")
st.header("Load Corpus")

uploaded_file = st.file_uploader("Upload a CSV file for bigrams or trigrams", type="csv")
if uploaded_file is not None:
    bigram_df, trigram_df = load_ngrams(uploaded_file)
    if bigram_df is not None:
        st.info("Bigram model ready.")
    elif trigram_df is not None:
        st.info("Trigram model ready.")

if st.button("Generate Sentence"):
    if bigram_df is not None:
        sentence = generate_text('bigrams', bigram_df, None)
        st.subheader("Generated Text")
        st.write(sentence)
    elif trigram_df is not None:
        sentence = generate_text('trigrams', None, trigram_df)
        st.subheader("Generated Text")
        st.write(sentence)
    else:
        st.error("Please load a corpus file first.")

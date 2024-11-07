import streamlit as st
import pandas as pd
import math

# Global dictionaries to store multiple bigram and trigram models
bigram_models = {}
trigram_models = {}

# Function to load n-grams from a file
def load_ngrams(file):
    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Check for bigram structure
        if 'Term 1' in df.columns and 'Term 2' in df.columns and 'Frequency of Bigram' in df.columns and 'Frequency of Context (Term 1)' in df.columns:
            st.success(f"Bigram file '{file.name}' loaded successfully.")
            return df, None

        # Check for trigram structure
        elif 'Term 1' in df.columns and 'Term 2' in df.columns and 'Term 3' in df.columns and 'Frequency of Trigram' in df.columns and 'Frequency of Context (Term 1, Term 2)' in df.columns:
            st.success(f"Trigram file '{file.name}' loaded successfully.")
            return None, df

        # Invalid file structure
        else:
            st.error(f"Invalid file structure for '{file.name}'. Please select a valid bigram or trigram file.")
            return None, None

    except Exception as e:
        st.error(f"Error loading the file '{file.name}': {e}")
        return None, None

# Function to calculate conditional probability with Laplace smoothing
def calculate_conditional_probability(n_gram, ngram_type, bigram_df=None, trigram_df=None, vocab_size=0):
    if ngram_type == 'bigrams':
        term1, term2 = n_gram
        ngram_count = bigram_df[(bigram_df['Term 1'] == term1) & (bigram_df['Term 2'] == term2)]['Frequency of Bigram'].sum()
        context_count = bigram_df[bigram_df['Term 1'] == term1]['Frequency of Context (Term 1)'].sum()
    elif ngram_type == 'trigrams':
        term1, term2, term3 = n_gram
        ngram_count = trigram_df[(trigram_df['Term 1'] == term1) & (trigram_df['Term 2'] == term2) & (trigram_df['Term 3'] == term3)]['Frequency of Trigram'].sum()
        context_count = trigram_df[(trigram_df['Term 1'] == term1) & (trigram_df['Term 2'] == term2)]['Frequency of Context (Term 1, Term 2)'].sum()

    probability = (ngram_count + 1) / (context_count + vocab_size)
    return probability

# Function to calculate joint probability
def calculate_joint_probability(test_sentence, ngram_type, bigram_df=None, trigram_df=None):
    test_sentence = test_sentence.split()
    vocab_size = len(set(bigram_df['Term 2']) if ngram_type == 'bigrams' else trigram_df['Term 3'])

    if ngram_type == 'bigrams':
        n_grams = [(test_sentence[i], test_sentence[i + 1]) for i in range(len(test_sentence) - 1)]
    elif ngram_type == 'trigrams':
        n_grams = [(test_sentence[i], test_sentence[i + 1], test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]

    joint_probability_log = 0

    for n_gram in n_grams:
        conditional_prob = calculate_conditional_probability(n_gram, ngram_type, bigram_df, trigram_df, vocab_size)
        joint_probability_log += math.log(conditional_prob)

    return math.exp(joint_probability_log)

# Streamlit interface
st.title("Conditional Probability Calculator")

# File upload for multiple model loading
uploaded_files = st.file_uploader("Upload up to 3 CSV files for n-gram models", type="csv", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        bigram_df, trigram_df = load_ngrams(file)

        if bigram_df is not None:
            bigram_models[file_name] = bigram_df
            st.write(f"Bigram Model '{file_name}' added.")
        if trigram_df is not None:
            trigram_models[file_name] = trigram_df
            st.write(f"Trigram Model '{file_name}' added.")

# Input for test sentence
test_sentence = st.text_input("Enter a test sentence")

# Button to calculate joint probability
if st.button("Calculate Joint Probability"):
    if test_sentence:
        probabilities = []

        if bigram_models:
            for model_name, bigram_df in bigram_models.items():
                prob = calculate_joint_probability(test_sentence, 'bigrams', bigram_df=bigram_df)
                probabilities.append((model_name, "Bigram Model", prob))

        if trigram_models:
            for model_name, trigram_df in trigram_models.items():
                prob = calculate_joint_probability(test_sentence, 'trigrams', trigram_df=trigram_df)
                probabilities.append((model_name, "Trigram Model", prob))

        if probabilities:
            # Sort probabilities in descending order
            probabilities.sort(key=lambda x: x[2], reverse=True)
            for model_name, model_type, prob in probabilities:
                st.write(f"{model_name} - {model_type} Joint Probability: {prob}")
        else:
            st.write("No models have been loaded.")
    else:
        st.warning("Please enter a test sentence.")

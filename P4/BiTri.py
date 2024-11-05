import streamlit as st
import pandas as pd
import spacy
from collections import Counter

# Load spaCy Spanish model
nlp = spacy.blank("es")

# Function to preprocess and normalize text (without removing punctuation)
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]

# Function to calculate bigrams and trigrams with frequency and conditional probability
def calculate_ngrams(text, n, min_freq=2):
    tokens = preprocess_text(text)

    ngrams = Counter()
    context_counts = Counter()
    
    # Generate n-grams and context counts
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        context = tuple(tokens[i:i + n - 1])
        ngrams[ngram] += 1
        context_counts[context] += 1
            
    # Calculate conditional probability and filter by min frequency
    data = []
    for ngram, freq in ngrams.items():
        if freq >= min_freq:  # Only include ngrams with frequency >= min_freq
            context = ngram[:-1]
            context_freq = context_counts[context]
            conditional_prob = freq / context_freq if context_freq > 0 else 0
            if n == 2:
                data.append([ngram[0], ngram[1], freq, context_freq, conditional_prob])
            elif n == 3:
                data.append([ngram[0], ngram[1], ngram[2], freq, context_freq, conditional_prob])
    
    # Create DataFrame and sort by frequency in descending order
    columns = ["Term 1", "Term 2", "Frequency", "Context Frequency", "Conditional Probability"] if n == 2 else \
              ["Term 1", "Term 2", "Term 3", "Frequency", "Context Frequency", "Conditional Probability"]
    df = pd.DataFrame(data, columns=columns)
    df = df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    return df

# Streamlit interface
st.title("Bigram and Trigram Generator")

uploaded_files = st.file_uploader("Upload your corpus files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load and process each uploaded corpus file
        df = pd.read_csv(uploaded_file)
        st.write(f"Processing file: {uploaded_file.name}")
        
        # Combine text data from the file into one string for a unified corpus
        text = " ".join(df[df.columns[0]].tolist())  # assuming text is in the first column
        
        # Generate bigrams and trigrams
        bigrams_df = calculate_ngrams(text, 2)
        trigrams_df = calculate_ngrams(text, 3)
        
        # Display results in Streamlit
        st.write("### Bigrams")
        st.dataframe(bigrams_df)
        st.write("### Trigrams")
        st.dataframe(trigrams_df)
        
        # Save as CSV
        bigrams_df.to_csv(f"Ngrams/{uploaded_file.name}_bigramas.csv", index=False)
        trigrams_df.to_csv(f"Ngrams/{uploaded_file.name}_trigramas.csv", index=False)
        
        st.download_button("Download Bigrams CSV", data=bigrams_df.to_csv(index=False), file_name=f"bigrams_{uploaded_file.name}")
        st.download_button("Download Trigrams CSV", data=trigrams_df.to_csv(index=False), file_name=f"trigrams_{uploaded_file.name}")

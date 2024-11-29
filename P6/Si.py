import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import re
import pickle

# Load SEL Lexicon
def load_sel(file_path):
    lexicon_sel = {}
    with open(file_path, 'r') as input_file:
        next(input_file)  # Skip header
        for line in input_file:
            parts = line.strip().split("\t")
            word, emotion, value = parts[0], parts[6], float(parts[5])
            if word not in lexicon_sel:
                lexicon_sel[word] = {}
            lexicon_sel[word][emotion] = value
    return lexicon_sel

# Extract SEL features
def get_sel_features(corpus, lexicon_sel):
    features = []
    for text in corpus:
        emotions = {emotion: 0.0 for emotion in ["Alegría", "Tristeza", "Enojo", "Repulsión", "Miedo", "Sorpresa"]}
        words = re.findall(r'\w+', text.lower())
        for word in words:
            if word in lexicon_sel:
                for emotion, value in lexicon_sel[word].items():
                    emotions[emotion] += value
        positive = emotions["Alegría"] + emotions["Sorpresa"]
        negative = sum(emotions.values()) - positive
        features.append([positive, negative])
    return np.array(features)

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and prepare dataset
def load_dataset(file_path, lexicon_sel):
    df = pd.read_excel(file_path)
    df['Combined'] = (df['Title'] + " " + df['Opinion']).apply(preprocess_text)
    X = df['Combined']
    y = df['Polarity']

    # Extract SEL features
    sel_features = get_sel_features(X, lexicon_sel)
    
    return X, sel_features, y

# Vectorize and balance dataset
def vectorize_and_balance(X_train, y_train, vectorizer_type="tfidf"):
    # Vectorize text data
    if vectorizer_type == "binarized":
        vectorizer = CountVectorizer(binary=True)
    elif vectorizer_type == "frequency":
        vectorizer = CountVectorizer()
    else:  # TF-IDF
        vectorizer = TfidfVectorizer()
    
    X_train_vec = vectorizer.fit_transform(X_train)

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=0)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)

    return X_train_balanced, y_train_balanced, vectorizer

# Train and evaluate model
def train_and_evaluate(X_train, y_train, model_type="logistic", cv_folds=5):
    # Select model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "svm":
        model = SVC()
    else:  # Random Forest
        model = RandomForestClassifier()

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    results = cross_validate(model, X_train, y_train, cv=cv, scoring="f1_macro", return_train_score=True)
    
    print(f"Model: {model_type}, Mean F1-macro: {np.mean(results['test_score'])}")
    return model

# Main script
if __name__ == "__main__":
    # Load SEL lexicon
    lexicon_sel_file = 'P6\SEL_full.txt'
    lexicon_sel = load_sel(lexicon_sel_file)

    # Load dataset
    dataset_file = 'P6\Rest_Mex_2022.xlsx'
    X, sel_features, y = load_dataset(dataset_file, lexicon_sel)

    # Split dataset
    X_train, X_test, y_train, y_test, sel_train, sel_test = train_test_split(
        X, sel_features, y, test_size=0.2, stratify=y, random_state=0
    )

    # Choose vectorization method and balance training data
    X_train_balanced, y_train_balanced, vectorizer = vectorize_and_balance(X_train, y_train, "tfidf")

    # Append SEL features to the vectorized data
    X_train_final = hstack([X_train_balanced, sel_train])
    X_test_final = hstack([vectorizer.transform(X_test), sel_test])

    # Train model
    best_model = train_and_evaluate(X_train_final, y_train_balanced, model_type="logistic")

    # Evaluate on test set
    best_model.fit(X_train_final, y_train_balanced)
    y_pred = best_model.predict(X_test_final)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

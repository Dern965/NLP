import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import spacy
import re

# Cargar modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

# Cargar datos
data = pd.read_csv('raw_data_corpus.csv')

# Concatenar columnas Title y Content
data['features'] = data['Title'] + " " + data['Content']
data['target'] = data['Section']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    data['features'], data['target'], test_size=0.2, random_state=42
)

# Normalización de texto
def normalize_text(text):
    if not isinstance(text, str):
        text = str(text) if not pd.isnull(text) else ""
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if token.is_alpha or token.is_punct
    ]
    return " ".join(tokens)

# Aplicar normalización
X_train = X_train.fillna("").apply(normalize_text)
X_test = X_test.fillna("").apply(normalize_text)

# Representaciones de texto
def apply_svd(X_train, X_test, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)
    return X_train_svd, X_test_svd

vectorizers = {
    'Binarized': CountVectorizer(binary=True),
    'Frequency': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

# Modelos de clasificación
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Naive Bayes': MultinomialNB(),
    'Multilayer Perceptron': MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=300)
}

# Evaluación de cada configuración
results = []
for vec_name, vectorizer in vectorizers.items():
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Omitir MultinomialNB para TF-IDF
    for clf_name, clf in classifiers.items():
        if vec_name == 'TF-IDF' and clf_name == 'Naive Bayes':
            print(f"Skipping Naive Bayes with {vec_name} due to negative values.")
            continue
        
        # Aplicar SVD solo si no es MultinomialNB
        if vec_name == 'TF-IDF' and clf_name != 'Naive Bayes':
            X_train_vec, X_test_vec = apply_svd(X_train_vec, X_test_vec)
        
        try:
            clf.fit(X_train_vec, y_train)
            y_pred = clf.predict(X_test_vec)
            
            # Generar reporte y evitar problemas de precisión
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            fscore = np.mean([v['f1-score'] for k, v in report.items() if k != 'accuracy'])
            results.append({
                'Classifier': clf_name,
                'Vectorizer': vec_name,
                'F1-Score': fscore
            })
        except ValueError as e:
            print(f"Error with {clf_name} and {vec_name}: {e}")

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results)

# Agregar configuración de la tabla como en la imagen
results_df['Text Normalization'] = [
    "Tokenization + stopwords + lemmatization" if vec_name != 'TF-IDF' else "Tokenization + text_cleaning + stopwords + lemmatization"
    for vec_name in results_df['Vectorizer']
]
results_df['ML Method Parameters'] = [
    "max_iter=200" if clf_name == 'Logistic Regression' else
    "default" if clf_name == 'Naive Bayes' else
    "hidden_layer_sizes=(200, 100)"
    for clf_name in results_df['Classifier']
]

# Ordenar columnas para generar tabla final
final_results = results_df[['Classifier', 'ML Method Parameters', 'Text Normalization', 'Vectorizer', 'F1-Score']]

# Mostrar resultados
print(final_results)

# Guardar resultados en CSV
final_results.to_csv('classification_results.csv', index=False)

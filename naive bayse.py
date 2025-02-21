import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def preprocess_text(text):
    
    text = text.lower()  
    
   
    tokens = word_tokenize(text)
    
    
    stop_words = set(stopwords.words('english'))
    

    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    
 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)  


def fetch_text_from_url(url):
    
    response = requests.get(url)
    response.raise_for_status()  
    soup = BeautifulSoup(response.text, 'html.parser')
    
    
    page_text = ' '.join(p.text for p in soup.find_all('p'))
    return page_text


texts = [
    "Natural language processing (NLP) is a subfield of computer science...",
    "Computer network is a set of computers sharing resources located on or provided by network nodes...",
    "Deep learning is a subset of machine learning...",
    "Computer hardware includes the physical parts of a computer...",
    "Most modern deep learning models are based on multi-layered neural networks...",
    "Tokenization is a process used in text analysis...",
    "Computer networks may be classified by many criteria...",
    "The nodes of a computer network can include personal computers...",
    "Determine the parse tree (grammatical analysis) of a given sentence...",
    "Computer architecture requires prioritizing between different goals..."
]

labels = ["NLP", "Network", "deep learning", "hardware", "NLP", "deep learning", "NLP", "Network", "NLP", "hardware"]


vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) 
X = vectorizer.fit_transform(texts)  


label2idx = {"NLP": 0, "Network": 1, "deep learning": 2, "hardware": 3}
y = [label2idx[label] for label in labels]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


nb_model = MultinomialNB()


nb_model.fit(X_train, y_train)


y_pred = nb_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


def predict_label(text, model, vectorizer, label2idx):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    predicted_label_idx = model.predict(vectorized_text)[0]
    idx2label = {v: k for k, v in label2idx.items()}
    return idx2label[predicted_label_idx]  


def count_label_words(text, vectorizer, model, label2idx):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    predicted_label_idx = model.predict(vectorized_text)[0]
    idx2label = {v: k for k, v in label2idx.items()}
    predicted_label = idx2label[predicted_label_idx]

    
    word_counts = vectorized_text.toarray().flatten()
    word_list = vectorizer.get_feature_names_out()
    
    label_word_count = {label: 0 for label in label2idx.keys()}
    
    for i, word in enumerate(word_list):
        if word in processed_text:
            for label in label2idx.keys():
                if label.lower() in word:
                    label_word_count[label] += word_counts[i]
    
    return label_word_count


url = 'https://en.wikipedia.org/wiki/Natural_language_processing' 
new_text = fetch_text_from_url(url)


predicted_label = predict_label(new_text, nb_model, vectorizer, label2idx)


label_word_count = count_label_words(new_text, vectorizer, nb_model, label2idx)


total_words = sum(label_word_count.values())
percentages = {label: (count / total_words) * 100 if total_words > 0 else 0 for label, count in label_word_count.items()}


print(f"Accuracy with Naive Bayes and TF-IDF: {accuracy}")
print(f"Predicted label: {predicted_label}")
for label, percent in percentages.items():
    print(f"Percentage of words associated with {label}: {percent:.2f}%")








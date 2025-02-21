import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def fetch_text_from_url(url):
    
    response = requests.get(url)
    response.raise_for_status() 
    soup = BeautifulSoup(response.text, 'html.parser')
    page_text = ' '.join(p.text for p in soup.find_all('p'))
    return page_text


def preprocess_text(text):
    
    text = text.lower() 
    
   
    tokens = word_tokenize(text)
    
   
    stop_words = set(stopwords.words('english'))
    
    
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)  



texts = [
    """Natural language processing (NLP) is a subfield of computer science and especially artificial intelligence. It is primarily concerned with providing computers with the ability to process data encoded in natural language and is thus closely related to information retrieval, knowledge representation and computational linguistics, a subfield of linguistics. Typically data is collected in text corpora, using either rule-based, statistical or neural-based approaches in machine learning and deep learning.

    Major tasks in natural language processing are speech recognition, text classification, natural-language understanding, and natural-language generation. """,

    "Computer network is a set of computers sharing resources located on or provided by network nodes. Computers use common communication protocols over digital interconnections to communicate with each other. These interconnections are made up of telecommunication network technologies based on physically wired, optical, and wireless radio-frequency methods that may be arranged in a variety of network topologies.",

    "Deep learning is a subset of machine learning that focuses on utilizing neural networks to perform tasks such as classification, regression, and representation learning. The field takes inspiration from biological neuroscience and is centered around stacking artificial neurons into layers and training them to process data.",
    
    "Computer hardware includes the physical parts of a computer, such as the central processing unit (CPU), random access memory (RAM), motherboard, computer data storage, graphics card, sound card, and computer case. It includes external devices such as a monitor, mouse, keyboard, and speakers.",
    
    "Most modern deep learning models are based on multi-layered neural networks such as convolutional neural networks and transformers, although they can also include propositional formulas or latent variables organized layer-wise in deep generative models such as the nodes in deep belief networks and deep Boltzmann machines.",
    
    "Tokenization is a process used in text analysis that divides text into individual words or word fragments. This technique results in two key components: a word index and tokenized text. The word index is a list that maps unique words to specific numerical identifiers, and the tokenized text replaces each word with its corresponding numerical token. These numerical tokens are then used in various deep learning methods.",
    
    "Computer networks may be classified by many criteria, including the transmission medium used to carry signals, bandwidth, communications protocols to organize network traffic, the network size, the topology, traffic control mechanisms, and organizational intent.",
    
    "The nodes of a computer network can include personal computers, servers, networking hardware, or other specialized or general-purpose hosts. They are identified by network addresses and may have hostnames. Hostnames serve as memorable labels for the nodes and are rarely changed after initial assignment. Network addresses serve for locating and identifying the nodes by communication protocols such as the Internet Protocol.",
    
    "Determine the parse tree (grammatical analysis) of a given sentence. The grammar for natural languages is ambiguous and typical sentences have multiple possible analyses: perhaps surprisingly, for a typical sentence there may be thousands of potential parses (most of which will seem completely nonsensical to a human).",
    
    "Computer architecture requires prioritizing between different goals, such as cost, speed, availability, and energy efficiency. The designer must have a good grasp of the hardware requirements and many different aspects of computing, from compilers to integrated circuit design."
]


labels = [
    "NLP", "Network", "deep learning", "hardware",
    "NLP", "deep learning", "NLP",
    "Network", "NLP", "hardware"
]

preprocessed_texts = [preprocess_text(text) for text in texts]


vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))  
X = vectorizer.fit_transform(preprocessed_texts)


label2idx = {"NLP": 0, "Network": 1, "deep learning": 2, "hardware": 3}
y = [label2idx[label] for label in labels]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_model = SVC(kernel='linear', C=1.0)


svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)


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
    
   
    total_words = sum(label_word_count.values())
    if total_words > 0:
        for label in label_word_count:
            label_word_count[label] = (label_word_count[label] / total_words) * 100  
    
    return label_word_count



url = 'https://en.wikipedia.org/wiki/Natural_language_processing'  
new_text = fetch_text_from_url(url)
predicted_label = predict_label(new_text, svm_model, vectorizer, label2idx)


label_word_count = count_label_words(new_text, vectorizer, svm_model, label2idx)


print(f"Predicted label: {predicted_label}")
for label, count in label_word_count.items():
    print(f"Percentage of words associated with {label}: {count:.2f}%")







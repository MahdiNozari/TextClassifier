import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from collections import defaultdict
import requests
from bs4 import BeautifulSoup


def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


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


label2idx = {"NLP": 0, "Network": 1, "deep learning": 2, "hardware": 3}
encoded_labels = [label2idx[label] for label in labels]


train_texts, test_texts, train_labels, test_labels = train_test_split(texts, encoded_labels, test_size=0.2)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextClassificationDataset(train_encodings, train_labels)
test_dataset = TextClassificationDataset(test_encodings, test_labels)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)


model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label2idx))


class_weights = torch.tensor([5.0, 1.0, 1.0, 1.0])  


optimizer = AdamW(model.parameters(), lr=2e-5)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5

word_label_counts = defaultdict(int)
total_word_count = 0  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        
        for idx in range(batch['labels'].shape[0]):
            label = batch['labels'][idx].item()
            text = train_texts[idx] 
            words = preprocess_text(text)  
            total_word_count += len(words)  
            for word in words:
                word_label_counts[label] += 1
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
target_names = list(label2idx.keys())

for label, count in word_label_counts.items():
    percentage = (count / total_word_count) * 100
    print(f"{target_names[label]}:  {percentage:.2f}%")



response = requests.get('https://en.wikipedia.org/wiki/Natural_language_processing')
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')
text_content = soup.get_text()


def predict_with_model(text):
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    encoding = {key: val.to(device) for key, val in encoding.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = target_names[predicted_class]
    return predicted_label

predicted_label = predict_with_model(text_content)
print(f"label for  text: {predicted_label}")
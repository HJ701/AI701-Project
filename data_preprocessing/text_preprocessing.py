# text_preprocessing.py

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    # Remove irrelevant information (e.g., patient IDs, dates)
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
    # Correct spelling errors (optional - can be added if necessary)
    # Standardize terminology (e.g., map synonyms to a standard term)
    # This requires a predefined dictionary of term mappings
    # For this example, we'll assume a simple mapping
    term_mapping = {
        'malocclusion': 'misalignment',
        'overjet': 'overjet',  # No change
        # Add more mappings as needed
    }
    for term, standard_term in term_mapping.items():
        text = re.sub(r'\b{}\b'.format(term), standard_term, text)
    return text

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

def preprocess_clinical_notes(text):
    text = clean_text(text)
    text = normalize_text(text)
    tokens = tokenize_text(text)
    tokens = lemmatize_tokens(tokens)
    processed_text = ' '.join(tokens)
    return processed_text

def encode_texts(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(
        texts,
        return_tensors='pt',  # 'tf' for TensorFlow, 'pt' for PyTorch
        padding=True,
        truncation=True,
        max_length=512
    )
    return inputs

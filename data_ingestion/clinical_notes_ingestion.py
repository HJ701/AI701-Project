# text_preprocessing.py

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer

# Ensure necessary NLTK data files are downloaded (uncomment when running for the first time)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def clean_text(text):
    """
    Cleans the text by removing irrelevant information and standardizing terminology.
    """
    # Remove numbers and standalone symbols
    text = re.sub(r'\b\d+\b', '', text)
    # Standardize terminology
    term_mapping = {
        'malocclusion': 'misalignment',
        'overjet': 'overjet',
        # Add more mappings as needed
    }
    for term, standard_term in term_mapping.items():
        text = re.sub(r'\b{}\b'.format(term), standard_term, text)
    return text

def normalize_text(text):
    """
    Normalizes the text by converting to lowercase and removing punctuation.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text):
    """
    Tokenizes the text into words.
    """
    tokens = word_tokenize(text)
    return tokens

def lemmatize_tokens(tokens):
    """
    Lemmatizes the tokens to their base forms.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

def preprocess_clinical_notes(text):
    """
    Performs full preprocessing on clinical notes.
    """
    text = clean_text(text)
    text = normalize_text(text)
    tokens = tokenize_text(text)
    tokens = lemmatize_tokens(tokens)
    processed_text = ' '.join(tokens)
    return processed_text

def encode_texts(texts):
    """
    Encodes a list of texts into numerical vectors using BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(
        texts,
        return_tensors='pt',  # Use 'tf' if using TensorFlow
        padding=True,
        truncation=True,
        max_length=512
    )
    return inputs

# text_preprocessing.py

import re
import string
import nltk
<<<<<<< HEAD
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
=======
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
from transformers import BertTokenizer, BertModel
# import torch

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Device configuration for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)
bert_model.eval()

def extract_iotn_grade(text):
    """
    Extracts the IOTN Grade from the clinical notes.
    """
    # Patterns like 'IOTN-4', 'IOTN Grade: 3', 'IOTN Grade 5'
    match = re.search(r'iotn[\s\-:]*(grade\s*)?(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(2))
    else:
        return None

def extract_malocclusion_class(text):
    """
    Extracts the Malocclusion Class from the clinical notes.
    """
    # Patterns like 'Class 2 malocclusion', 'CLASS I MALOCCLUSION'
    match = re.search(r'class\s*(i{1,3}|iv|v|\d+)', text, re.IGNORECASE)
    if match:
        class_info = match.group(1).upper()
        # Map Roman numerals or digits to standard classes
        class_mapping = {
            'I': 'Class I',
            'II': 'Class II',
            'III': 'Class III',
            'IV': 'Class IV',
            'V': 'Class V',
            '1': 'Class I',
            '2': 'Class II',
            '3': 'Class III',
            '4': 'Class IV',
            '5': 'Class V',
            # Extend if necessary
        }
        return class_mapping.get(class_info, 'Unknown')
    else:
        return 'Unknown'

def extract_diagnoses(text):
    """
    Extracts diagnoses from the clinical notes based on predefined keywords.
    """
    # Comprehensive list of possible diagnoses to look for
    diagnoses_list = [
        'anterior crossbite',
        'posterior crossbite',
        'bilateral crossbite',
        'crossbite',
        'mild crowding',
        'moderate crowding',
        'severe crowding',
        'crowding',
        'anterior open bite',
        'posterior open bite',
        'open bite',
        'overjet',
        'overbite',
        'deep bite',
        'mild spacing',
        'moderate spacing',
        'severe spacing',
        'spacing',
        'left midline deviation',
        'right midline deviation',
        'midline deviation',
        'left midline shift',
        'right midline shift',
        'midline shift',
        'lower midline shift',
        'upper midline shift',
        'lower midline deviation',
        'upper midline deviation',
        'impacted teeth',
        'missing teeth',
        'lower crowding',
        'upper crowding',
        'lower spacing',
        'upper spacing',
        # Add more diagnoses as needed
    ]
    diagnoses_found = set()
    for diagnosis in diagnoses_list:
        # Use word boundaries to ensure accurate matching
        pattern = r'\b{}\b'.format(re.escape(diagnosis))
        if re.search(pattern, text, re.IGNORECASE):
            diagnoses_found.add(diagnosis)
    return list(diagnoses_found)

def clean_text(text):
    """
    Cleans the clinical notes by removing tooth numbers and unnecessary digits,
    standardizing terminology, and removing extra whitespace.
    """
    # 1. Remove tooth numbers like #22
    text = re.sub(r'#\d+', '', text)
    
    # 2. Remove standalone numbers unless preceded by 'Class' or 'IOTN'
    # Tokenize the text to handle each word separately
    tokens = text.split()
    cleaned_tokens = []
    for i, token in enumerate(tokens):
        # Check if the token is a standalone number
        if re.fullmatch(r'\d+', token):
            # Check if the previous token is 'Class' or starts with 'IOTN'
            if i > 0:
                prev_token = tokens[i-1].lower()
                if prev_token == 'class' or prev_token.startswith('iotn'):
                    cleaned_tokens.append(token)  # Keep the number
                else:
                    continue  # Remove the standalone number
            else:
                continue  # Remove if it's the first token
        else:
            cleaned_tokens.append(token)
    
    # Reconstruct the text from cleaned tokens
    text = ' '.join(cleaned_tokens)
    
    # 3. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Standardize terminology
    term_mapping = {
        # Include all relevant terms and standardize them
        'malocclusion': 'malocclusion',
        'overjet': 'overjet',
        'over bite': 'overbite',
        'overbite': 'overbite',
        'deepbite': 'deep bite',
        'deep bite': 'deep bite',
        'openbite': 'open bite',
        'open bite': 'open bite',
        'anterior open bite': 'anterior open bite',
        'posterior open bite': 'posterior open bite',
        'crossbite': 'crossbite',
        'anterior crossbite': 'anterior crossbite',
        'posterior crossbite': 'posterior crossbite',
        'bilateral crossbite': 'bilateral crossbite',
        'crowding': 'crowding',
        'mild crowding': 'mild crowding',
        'moderate crowding': 'moderate crowding',
        'severe crowding': 'severe crowding',
        'lower crowding': 'lower crowding',
        'upper crowding': 'upper crowding',
        'lower spacing': 'lower spacing',
        'upper spacing': 'upper spacing',
        'spacing': 'spacing',
        'mild spacing': 'mild spacing',
        'moderate spacing': 'moderate spacing',
        'severe spacing': 'severe spacing',
        'midline deviation': 'midline deviation',
        'left midline deviation': 'left midline deviation',
        'right midline deviation': 'right midline deviation',
        'midline shift': 'midline shift',
        'left midline shift': 'left midline shift',
        'right midline shift': 'right midline shift',
        'lower midline shift': 'lower midline shift',
        'upper midline shift': 'upper midline shift',
        'impacted teeth': 'impacted teeth',
        'impacted': 'impacted teeth',
        'missing teeth': 'missing teeth',
        'lower midline deviation': 'lower midline deviation',
        'upper midline deviation': 'upper midline deviation',
        'lower midline shift': 'lower midline shift',
        'upper midline shift': 'upper midline shift',
        # Add more mappings as needed
    }
    
    for term, standard_term in term_mapping.items():
        pattern = r'\b{}\b'.format(re.escape(term))
        text = re.sub(pattern, standard_term, text, flags=re.IGNORECASE)
    
    return text.strip()

def normalize_text(text):
    """
    Normalizes the text by converting to lowercase and removing punctuation.
    """
>>>>>>> feature_extraction_with_classification_fusion_layer
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

<<<<<<< HEAD
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
=======
def preprocess_and_extract_labels(text):
    """
    Extracts labels from the clinical notes and cleans the text.
    """
    # 1. Extract labels before cleaning the text
    labels = {
        'IOTN_Grade': extract_iotn_grade(text),
        'Malocclusion_Class': extract_malocclusion_class(text),
        'Diagnosis': extract_diagnoses(text)
    }
    # 2. Clean and normalize the text for other purposes (if needed)
    cleaned_text = clean_text(text)
    normalized_text = normalize_text(cleaned_text)
    # (Optionally, you can store or return the cleaned and normalized text)
    return labels

def encode_texts_with_bert(texts, batch_size=32):
    """
    Encodes a list of texts using BERT and returns the embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # Get the embeddings from the last hidden state corresponding to [CLS] token
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    return embeddings

def reduce_dimensions(embeddings, n_components=50, pca_model_path=None):
    """
    Reduces the dimensionality of embeddings using PCA.
    If pca_model_path is None, fits a new PCA model and saves it.
    """
    if pca_model_path is None:
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        # Save the PCA model for future use
        joblib.dump(pca, 'pca_model.pkl')
    else:
        # Load the PCA model and transform embeddings
        pca = joblib.load(pca_model_path)
        reduced_embeddings = pca.transform(embeddings)
    return reduced_embeddings

def preprocess_text_data(df, is_training=True):
    """
    Preprocesses the text data in the DataFrame and returns features.
    """
    # Extract labels
    df['Labels'] = df['clinical_notes'].apply(preprocess_and_extract_labels)
    
    # Clean and normalize text (if needed)
    df['cleaned_text'] = df['clinical_notes'].apply(clean_text)
    
    # Encode text using BERT
    texts = df['cleaned_text'].tolist()
    embeddings = encode_texts_with_bert(texts)
    
    # Reduce dimensionality
    if is_training:
        reduced_embeddings = reduce_dimensions(embeddings, n_components=50)
    else:
        reduced_embeddings = reduce_dimensions(embeddings, n_components=50, pca_model_path='pca_model.pkl')
    
    # Convert reduced embeddings to DataFrame
    embedding_df = pd.DataFrame(reduced_embeddings, index=df.index)
    
    # Extract structured labels from df['Labels']
    labels_df = pd.json_normalize(df['Labels'])
    
    # Handle 'Diagnosis' field
    mlb = MultiLabelBinarizer()
    diagnosis_binarized = mlb.fit_transform(labels_df['Diagnosis'])
    diagnosis_df = pd.DataFrame(diagnosis_binarized, columns=mlb.classes_, index=df.index)
    # Save the MultiLabelBinarizer for future use
    joblib.dump(mlb, 'mlb.pkl')
    
    # One-hot encode 'Malocclusion_Class'
    # malocclusion_dummies = pd.get_dummies(labels_df['Malocclusion_Class'], prefix='Malocclusion_Class')
    
    # Concatenate all features
    features_df = pd.concat([
        embedding_df,
        labels_df[['IOTN_Grade']],
        malocclusion_dummies,
        diagnosis_df
    ], axis=1)
    
    return features_df

def preprocess_text_data_new(df):
    """
    Preprocesses the text data in the DataFrame and returns features.
    """
    # Extract labels
    df['Labels'] = df['clinical_notes'].apply(preprocess_and_extract_labels)
    
    # Clean and normalize text (if needed)
    df['cleaned_text'] = df['clinical_notes'].apply(clean_text)
    
    # Encode text using BERT
    texts = df['cleaned_text'].tolist()
    
    # Extract structured labels from df['Labels']
    labels_df = pd.json_normalize(df['Labels'])
    
    # Handle 'Diagnosis' field
    mlb = MultiLabelBinarizer()
    diagnosis_binarized = mlb.fit_transform(labels_df['Diagnosis'])
    diagnosis_df = pd.DataFrame(diagnosis_binarized, columns=mlb.classes_, index=df.index)
    
    # Concatenate all features
    features_df = pd.concat([
        df[['patient_id']],
        df[['set']],
        df[['image_filenames']],
        df[['radiograph_filenames']],
        labels_df[['IOTN_Grade']],
        labels_df[['Malocclusion_Class']],
        diagnosis_df
    ], axis=1)
    
    return features_df
>>>>>>> feature_extraction_with_classification_fusion_layer

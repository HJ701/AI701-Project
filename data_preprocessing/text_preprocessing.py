# text_preprocessing.py

import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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
        'midline shift',
        'impacted teeth',
        'missing teeth',
        'lower crowding',
        'upper crowding',
        'lower spacing',
        'upper spacing',
        'midline shift',
        'midline deviation',
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
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

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

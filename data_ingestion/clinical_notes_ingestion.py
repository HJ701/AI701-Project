<<<<<<< HEAD
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
=======
import os
from pathlib import Path
import logging
from docx import Document

def ingest_clinical_notes(data_path):
    """
    Ingests clinical notes from either .txt or .docx files within the specified directory.

    Parameters:
        data_path (str or Path): Path to the directory containing clinical notes files.

    Returns:
        str or None: The extracted clinical notes text, or None if no appropriate file is found.
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Ensure data_path is a Path object
    data_path = Path(data_path)

    if not data_path.exists() or not data_path.is_dir():
        logger.error(f"The provided data path '{data_path}' does not exist or is not a directory.")
        return None

    # Define keywords and supported extensions for clinical notes files
    keywords = ['clinical_notes', 'diagnosis', 'untitled']
    supported_extensions = ['.txt', '.docx']

    # Lists to hold matched files
    matched_txt_files = []
    matched_docx_files = []

    # Iterate through all files in the directory
    for file in data_path.iterdir():
        if file.is_file():
            filename_lower = file.name.lower()
            if file.suffix.lower() in supported_extensions:
                if any(keyword in filename_lower for keyword in keywords):
                    if file.suffix.lower() == '.txt':
                        matched_txt_files.append(file)
                    elif file.suffix.lower() == '.docx':
                        matched_docx_files.append(file)

    # If no files matched with keywords, attempt to find any .txt or .docx files
    if not matched_txt_files and not matched_docx_files:
        logger.warning("No clinical notes files found with specified keywords. Attempting to find any .txt or .docx files.")
        for file in data_path.iterdir():
            if file.is_file():
                if file.suffix.lower() == '.txt':
                    matched_txt_files.append(file)
                elif file.suffix.lower() == '.docx':
                    matched_docx_files.append(file)

    # If still no files found, return None
    if not matched_txt_files and not matched_docx_files:
        logger.warning("No clinical notes files found in the data directory.")
        return None

    clinical_notes = []

    # Process .txt files first
    for txt_file in matched_txt_files:
        try:
            logger.info(f"Attempting to read .txt file: {txt_file}")
            with txt_file.open('r', encoding='utf-8') as f:
                clinical_notes.append(f.read())
            logger.info(f"Successfully read .txt file: {txt_file}")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {txt_file}. Trying 'ISO-8859-1' encoding.")
            try:
                with txt_file.open('r', encoding='ISO-8859-1') as f:
                    clinical_notes.append(f.read())
                logger.info(f"Successfully read .txt file with 'ISO-8859-1' encoding: {txt_file}")
            except Exception as e:
                logger.error(f"Failed to read .txt file {txt_file} with 'ISO-8859-1' encoding: {e}")
        except Exception as e:
            logger.error(f"Error reading .txt file {txt_file}: {e}")

    # If no .txt files were read, process .docx files
    if not clinical_notes:
        for docx_file in matched_docx_files:
            try:
                logger.info(f"Attempting to read .docx file: {docx_file}")
                doc = Document(docx_file)
                full_text = [para.text for para in doc.paragraphs]
                clinical_notes.append('\n'.join(full_text))
                logger.info(f"Successfully read .docx file: {docx_file}")
            except Exception as e:
                logger.error(f"Error reading .docx file {docx_file}: {e}")

    # Concatenate all clinical notes content
    combined_clinical_notes = '\n'.join(clinical_notes) if clinical_notes else None

    if combined_clinical_notes:
        logger.info("Successfully ingested clinical notes.")
    else:
        logger.warning("Clinical notes were found but could not be read.")

    return combined_clinical_notes
>>>>>>> feature_extraction_with_classification_fusion_layer

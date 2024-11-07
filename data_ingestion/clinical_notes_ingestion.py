import pandas as pd
import regex
# import unidecode
# import beautifulsoup4
import os


def ingest_clinical_notes(data_path):
    
    # List all files in the data directory
    files = os.listdir(data_path)
    clinical_notes_file = None

    # Try to find the clinical notes file
    for file in files:
        if 'txt' in file.lower() or 'txt' in file.lower() or 'clinical_notes' in file.lower():
            clinical_notes_file = os.path.join(data_path, file)
            break

    if clinical_notes_file is None:
        print("No clinical notes file found in the data directory.")
        return None

    # Read the clinical notes
    with open(clinical_notes_file, 'r') as f:
        clinical_notes = f.read()

    return clinical_notes
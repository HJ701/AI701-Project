import os
import pandas as pd
from sklearn.model_selection import train_test_split
from clinical_notes_ingestion import ingest_clinical_notes
from image_ingestion import ingest_images
from radiograph_ingestion import ingest_radiographs

def main():
    data_path = '/Users/hj/OrthoAI/data/'

    # List all patient folders
    patient_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    # Split patient folders into train, validation, and test sets
    train_val_folders, test_folders = train_test_split(
        patient_folders, test_size=0.2, random_state=42)

    train_folders, val_folders = train_test_split(
        train_val_folders, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    sets = {
        'train': train_folders,
        'validation': val_folders,
        'test': test_folders
    }

    data = []

    for set_name, folders in sets.items():
        for patient_folder in folders:
            patient_path = os.path.join(data_path, patient_folder)
            if os.path.isdir(patient_path):
                # Ingest data for each patient
                clinical_notes = ingest_clinical_notes(patient_path)
                images = ingest_images(patient_path)
                radiographs = ingest_radiographs(patient_path)

                # Create a record for each patient
                record = {
                    'patient_id': patient_folder,
                    'set': set_name,
                    'clinical_notes': clinical_notes,
                    'image_filenames': list(images.keys()),
                    'radiograph_filenames': list(radiographs.keys())
                }
                data.append(record)

    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    csv_output_path = '/Users/hj/OrthoAI/data/structured_data.csv'
    df.to_csv(csv_output_path, index=False)

    # Print DataFrame
    print("OrthoAI Structured Data:")
    print(df)

if __name__ == '__main__':
    main()

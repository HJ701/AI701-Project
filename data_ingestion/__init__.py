import os
import pandas as pd
from clinical_notes_ingestion import ingest_clinical_notes
from image_ingestion import ingest_images
from radiograph_ingestion import ingest_radiographs

def main():
    data_path = '/Users/hj/OrthoAI/Datasets/'

    # Create a DataFrame
    data = []

    # Loop over all subdirectories in data_path
    for patient_folder in os.listdir(data_path):
        patient_path = os.path.join(data_path, patient_folder)
        if os.path.isdir(patient_path):
            # Ingest data for each patient
            clinical_notes = ingest_clinical_notes(patient_path)
            images = ingest_images(patient_path)
            radiographs = ingest_radiographs(patient_path)

            # Assuming one patient per dataset folder
            patient_id = patient_folder
            record = {
                'patient_id': patient_id,
                'clinical_notes': clinical_notes,
                'image_filenames': list(images.keys()),
                'radiograph_filenames': list(radiographs.keys())
            }
            data.append(record)

    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    csv_output_path = '/Users/hj/OrthoAI/Datasets/structured_data.csv'
    df.to_csv(csv_output_path, index=False)

    # Print DataFrame
    print("Orthodentist Structured Data:")
    print(df)

if __name__ == '__main__':
    main()

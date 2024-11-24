# main.py

import os
import pandas as pd
import numpy as np
from data_preprocessing import (
    preprocess_and_extract_labels,
    preprocess_dataset,
    preprocess_radiograph_dataset,
)
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer

# Define the output directory for processed samples
output_dir = 'Processed_Samples'
os.makedirs(output_dir, exist_ok=True)

def main():
    # Data Ingestion
    csv_input_path = "Datasets/structured_data.csv"
    df = pd.read_csv(csv_input_path)

    # Apply label extraction to the entire dataset
    print("\nExtracting labels from clinical notes...")
    labels_df = df['clinical_notes'].apply(lambda x: pd.Series(preprocess_and_extract_labels(x)))
    df = pd.concat([df, labels_df], axis=1)

    # Multi-Label Encoding for 'Diagnosis'
    print("\nApplying multi-label encoding to 'Diagnosis'...")
    mlb = MultiLabelBinarizer()
    diagnosis_encoded = mlb.fit_transform(df['Diagnosis'])
    diagnosis_classes = mlb.classes_
    diagnosis_encoded_df = pd.DataFrame(diagnosis_encoded, columns=diagnosis_classes)
    df = pd.concat([df, diagnosis_encoded_df], axis=1)

    # Encode 'Malocclusion_Class' as numeric labels
    print("\nEncoding 'Malocclusion_Class' as numeric labels...")
    malocclusion_classes = df['Malocclusion_Class'].unique().tolist()
    malocclusion_class_to_index = {cls: idx for idx, cls in enumerate(malocclusion_classes)}
    df['Malocclusion_Class_Encoded'] = df['Malocclusion_Class'].map(malocclusion_class_to_index)

    # ------------------- Export Processed Data -------------------
    # Option 1: Export the entire processed DataFrame
    processed_csv_path = os.path.join(output_dir, 'processed_data.csv')
    df.to_csv(processed_csv_path, index=False)
    print(f"\nProcessed data has been saved to {processed_csv_path}")

    # Option 2: Export a sample of the processed DataFrame
    sample_csv_path = os.path.join(output_dir, 'processed_sample.csv')
    df_sample = df[['patient_id', 'clinical_notes', 'IOTN_Grade', 'Malocclusion_Class', 'Diagnosis']].head(10)
    df_sample.to_csv(sample_csv_path, index=False)
    print(f"Sample of processed data has been saved to {sample_csv_path}")
    # -----------------------------------------------------------------

    # Process each set separately
    for set_name in ["train", "validation", "test"]:
        print(f"\nProcessing {set_name} set:")
        subset = df[df["set"] == set_name].reset_index(drop=True)

        # Prepare labels for image and radiograph datasets
        diagnosis_columns = diagnosis_classes.tolist()

        # Preprocess Images
        print("\nPreprocessing Images...")
        image_paths = []
        image_labels = []
        for idx, row in subset.iterrows():
            patient_id = str(row["patient_id"])
            img_filenames = eval(row["image_filenames"])
            # labels for this patient, ensure they are of type float32
            patient_labels = row[diagnosis_columns].values.astype(np.float32)
            for img_filename in img_filenames:
                img_path = os.path.join("Datasets", patient_id, img_filename)
                image_paths.append(img_path)
                image_labels.append(patient_labels)
        print(f"Total images: {len(image_paths)}")
        print(f"Total image labels: {len(image_labels)}")
        is_training = set_name == "train"
        augmentations = ["flip_horizontal", ("rotate", {"angle": 10})]

        # Stack image_labels into a numpy array
        image_labels = np.stack(image_labels)
        print("image_labels shape:", image_labels.shape)
        print("image_labels dtype:", image_labels.dtype)

        image_dataset = preprocess_dataset(
            image_paths=image_paths,
            labels=image_labels,
            is_training=is_training,
            augmentations=augmentations if is_training else None,
        )

        # Print sample image batch shape
        for images_batch, img_labels_batch in image_dataset.take(1):
            print("Image batch shape:", images_batch.shape)
            print("Label batch shape:", img_labels_batch.shape)
            break  # Only need one batch for testing

        # Preprocess Radiographs
        print("\nPreprocessing Radiographs...")
        radiograph_paths = []
        radiograph_labels = []
        for idx, row in subset.iterrows():
            patient_id = str(row["patient_id"])
            rad_filenames = eval(row["radiograph_filenames"])
            # labels for this patient, ensure they are of type float32
            patient_labels = row[diagnosis_columns].values.astype(np.float32)
            for rad_filename in rad_filenames:
                rad_path = os.path.join("Datasets", patient_id, rad_filename)
                radiograph_paths.append(rad_path)
                radiograph_labels.append(patient_labels)
        print(f"Total radiographs: {len(radiograph_paths)}")
        print(f"Total radiograph labels: {len(radiograph_labels)}")

        # Stack radiograph_labels into a numpy array
        radiograph_labels = np.stack(radiograph_labels)
        print("radiograph_labels shape:", radiograph_labels.shape)
        print("radiograph_labels dtype:", radiograph_labels.dtype)

        # Create radiograph dataset
        radiograph_dataset = preprocess_radiograph_dataset(
            image_paths=radiograph_paths,
            labels=radiograph_labels,
            is_training=is_training,
        )

        # Print sample radiograph batch shape
        for radiographs_batch, rad_labels_batch in radiograph_dataset.take(1):
            print("Radiograph batch shape:", radiographs_batch.shape)
            print("Label batch shape:", rad_labels_batch.shape)
            break  # Only need one batch for testing

    # ------------------- Preview Processed Data -------------------
    print("\nPreviewing processed data:")
    preview_columns = ['patient_id', 'IOTN_Grade', 'Malocclusion_Class', 'Diagnosis'] + diagnosis_columns
    print(df[preview_columns].head())
    # -----------------------------------------------------------------

if __name__ == "__main__":
    main()

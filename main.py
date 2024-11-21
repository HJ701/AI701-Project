import os
import pandas as pd
from data_preprocessing import (
    preprocess_clinical_notes,
    encode_texts,
    preprocess_dataset,
    preprocess_radiograph_dataset,
)
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    # Data Ingestion
    csv_input_path = "Datasets/structured_data.csv"
    df = pd.read_csv(csv_input_path)

    # Process each set separately
    for set_name in ["train", "validation", "test"]:
        print(f"\nProcessing {set_name} set:")
        subset = df[df["set"] == set_name].reset_index(drop=True)

        # Preprocess Clinical Notes
        # print("\nPreprocessing Clinical Notes...")
        # subset['processed_notes'] = subset['clinical_notes'].apply(preprocess_clinical_notes)
        # # Encode texts
        # encoded_notes = encode_texts(subset['processed_notes'].tolist())
        # print("Sample encoded text input IDs:", encoded_notes['input_ids'][:1])

        # Preprocess Images
        print("\nPreprocessing Images...")
        image_paths = []
        labels = []  # Adjust this if you have labels
        for idx, row in subset.iterrows():
            # patient_id = row['patient_id']
            patient_id = str(row["patient_id"])
            # Assuming you have a 'label' column
            label = row.get("label", 0)  # Default label to 0 if not present
            img_filenames = eval(row["image_filenames"])
            for img_filename in img_filenames:
                print(f"patient_id: {patient_id}, type: {type(patient_id)}")
                print(f"img_filename: {img_filename}, type: {type(img_filename)}")
                img_path = os.path.join("Datasets/", patient_id, img_filename)
                image_paths.append(img_path)
                labels.append(label)

        # Create image dataset
        is_training = set_name == "train"
        augmentations = ["flip_horizontal", ("rotate", {"angle": 10})]
        image_dataset = preprocess_dataset(
            image_paths=image_paths,
            labels=labels,
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
            label = row.get("label", 0)  # Adjust as needed
            rad_filenames = eval(row["radiograph_filenames"])
            for rad_filename in rad_filenames:
                print(f"patient_id: {patient_id}, type: {type(patient_id)}")
                print(f"img_filename: {img_filename}, type: {type(img_filename)}")
                rad_path = os.path.join("Datasets/", patient_id, rad_filename)
                radiograph_paths.append(rad_path)
                radiograph_labels.append(label)

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

        def visualize_images(images_batch, labels_batch, save_prefix="image"):
            """
            Visualizes a batch of processed images.
            """
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            images = images_batch.numpy()
            labels = labels_batch.numpy()

            plt.figure(figsize=(12, 12))
            for i in range(min(len(images), 9)):  # Display up to 9 images
                plt.subplot(3, 3, i + 1)
                if images[i].shape[-1] == 1:
                    plt.imshow(images[i].squeeze(), cmap="gray")
                else:
                    plt.imshow(images[i])
                plt.title(f"Label: {labels[i]}")
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"tests/samples/{date_time}_{save_prefix}.png")

        visualize_images(images_batch, img_labels_batch, save_prefix="image")
        visualize_images(radiographs_batch, rad_labels_batch, save_prefix="radiograph")

    # # Feature Extraction
    # text_features = text_feature_extraction.extract(preprocessed_text)
    # image_features = image_feature_extraction.extract(preprocessed_images)
    # three_d_features = three_d_feature_extraction.extract(preprocessed_3d)
    # radiograph_features = radiograph_feature_extraction.extract(preprocessed_radiographs)

    # # Model Prediction
    # fused_features = fusion_layer.fuse([
    #     text_features,
    #     image_features,
    #     three_d_features,
    #     radiograph_features
    # ])
    # predictions = classification_layer.predict(fused_features)

    # # Decision Making
    # decisions = thresholding.apply(predictions)

    # # Evaluation
    # performance_metrics.evaluate(decisions)

    # # Deployment
    # api_integration.update(decisions)
    # user_interface.display(decisions)


if __name__ == "__main__":
    main()

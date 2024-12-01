import os

import pandas as pd
import tensorflow as tf

from data_preprocessing.image_preprocessing import (
    augment_image,
    load_and_preprocess_image,
)

# mapping for IOTN and malocclusion classes
# IOTN_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
IOTN_mapping = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
malocclusion_mapping = {"Class I": 0, "Class II": 1, "Class III": 2}


def process_image(image_path, augmentations):
    """
    Process the image by loading and preprocessing it and applying augmentations if provided.

    Args:
        image_path (str): Path to the image file.
        augmentations (list): List of tuples where each tuple contains the augmentation name and its parameters. See augment_image function for details.
    """
    image = load_and_preprocess_image(
        image_path, target_size=(224, 224), preserve_aspect_ratio=True
    )
    if augmentations:
        image = augment_image(image, augmentations)
    return image


def process_sample(**kwargs):
    """
    Process a sample by loading and preprocessing the image and radiograph, and encoding the labels.

    Args:
        **kwargs: Dictionary containing the sample information and augmentations.

    Returns:
        Tuple[Tensor, Tensor]: Tuple containing the processed image and radiograph.
        Tuple[Tensor, Tensor, Tuple]: Tuple containing the encoded labels.
    """
    image = process_image(kwargs["image_path"], augmentations=kwargs["augmentations"])
    radiograph = process_image(kwargs["radiograph_path"], augmentations=None)
    labels = [
        v
        for k, v in kwargs.items()
        if k not in ["image_path", "radiograph_path", "augmentations"]
    ]

    label1 = tf.convert_to_tensor(
        int(labels[0]), dtype=tf.int32
    )  # Multiclass target 1 (shape: (batch_size,))
    label2 = tf.convert_to_tensor(
        int(labels[1]), dtype=tf.int32
    )  # Multiclass target 2 (shape: (batch_size,))
    label3 = tf.convert_to_tensor(
        labels[2:], dtype=tf.float32
    )  # Multilabel target 3 (shape: (batch_size, num_classes))
    return (image, radiograph), (label1, label2, label3)


def get_dataset(csv_path, imgs_path, batch_size, split="train", augmentations=None):
    """
    Create a tf.data.Dataset from the CSV file at the given path.

    Args:
        csv_path (str): Path to the CSV file.
        imgs_path (str): Path to the directory containing the images folders.
        batch_size (int): Batch size for the dataset.
        split (str): Split of the dataset. One of 'train', 'validation', or 'test'.
        augmentations (list): List of tuples where each tuple contains the augmentation name and its parameters. See augment_image function for details.

    Returns:
        tf.data.Dataset: Dataset object containing the processed samples.
    """
    assert split in ["train", "validation", "test"], "Invalid split"
    df = pd.read_csv(csv_path)
    df = df.loc[df["set"] == split].drop(columns=["set"])
    print(f"Loaded {split} data with {len(df)} samples")
    n_patients = len(df["patient_id"].unique())
    print(f"Number of unique patients: {n_patients}")

    # prepare image and radiograph paths
    df["patient_id"] = df["patient_id"].astype(str)
    df["image_path"] = df.apply(
        lambda x: os.path.join(imgs_path, x["patient_id"], x["image_path"]), axis=1
    )
    df["radiograph_path"] = df.apply(
        lambda x: os.path.join(imgs_path, x["patient_id"], x["radiograph_path"]), axis=1
    )
    df = df.drop(columns=["patient_id"])
    # encode IOTN and malocclusion class
    df["IOTN_grade"] = df["IOTN_grade"].map(IOTN_mapping)
    df["malocclusion_class"] = df["malocclusion_class"].map(malocclusion_mapping)

    dataset = tf.data.Dataset.from_tensor_slices(dict(df))

    if split == "train":
        dataset = dataset.shuffle(buffer_size=1000)
    else:
        assert (
            augmentations is None
        ), "augmentations must be None if is_training is False"

    dataset = dataset.map(
        lambda x: process_sample(**dict(x, augmentations=augmentations)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# test
if __name__ == "__main__":
    from imageio import imwrite
    from glob import glob
    dataset = get_dataset(
        "Processed_Samples/consolidated_data.csv",
        "../orthoai_patient_records/orthoai_patient_records/",
        batch_size=32,
        split="train",
        augmentations=[("flip_horizontal", {})],
    )
    for (images, radiographs), (label1, label2, label3, pid) in dataset:
        for i in range(images.shape[0]):
            cur_pid = pid[i].numpy().decode('utf-8')
            existing_files = sorted(glob(f"data/samples/img/{cur_pid}_*"))
            last_idx = existing_files[-1].split("_")[-1].split(".")[0] if existing_files else -1
            idx = int(last_idx) + 1
            imwrite(f"data/samples/img/{cur_pid}_{idx}.png", np.uint8(images[i].numpy() * 255))
            imwrite(f"data/samples/rad/{cur_pid}.png", np.uint8(radiographs[i].numpy() * 255))

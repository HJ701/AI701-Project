"""
Image preprocessing functions.
"""

# pylint: disable=no-member
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_image(
    image_path, target_size=(224, 224), preserve_aspect_ratio=True
):
    """
    Reads an image from a file, decodes it into a dense tensor,
    converts it into a floating-point tensor, and resizes it to a fixed size.

    Args:
        image_path: Path to the image file.
        target_size: The size to which the image will be resized. Default is (224, 224).
        preserve_aspect_ratio: If True, the aspect ratio of the image will be preserved
        when resizing. Default is True.

    Returns:
        Processed image tensor.
    """
    # Read the image file
    image = tf.io.read_file(image_path)
    # Decode the image into a tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert to floating-point tensors and rescale pixel values to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    if preserve_aspect_ratio:
        # Resize the image while preserving the aspect ratio
        image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    else:
        # Resize the image to a standard size
        image = tf.image.resize(image, size=target_size)
    return image


def _flip_horizontal(image):
    """
    Flips the image horizontally with a 50% chance.
    """
    if tf.random.uniform([]) > 0.5:
        return tf.image.flip_left_right(image)
    return image


def _flip_vertical(image):
    """
    Flips the image vertically with a 50% chance.
    """
    if tf.random.uniform([]) > 0.5:
        return tf.image.flip_up_down(image)
    return image


def flip_vertical(image):
    """
    Flips the image vertically.
    """
    return tf.image.flip_up_down(image)

<<<<<<< HEAD
=======
def rotate_90deg(image):
    """
    Rotates the image by 90 degrees.
    """
    return tf.image.rot90(image)

>>>>>>> feature_extraction_with_classification_fusion_layer

# this is a workaround to avoid reinitializing the layer every time
rot_layer = tf.keras.layers.RandomRotation(factor=0.1, fill_mode="constant")


def _rotate(image, angle):
    """
    Rotates the image by a random angle within [-angle, angle] with a 50% chance.

    Args:
        image: Image tensor.
        angle: Angle range in degrees.
    """
    if tf.random.uniform([]) > 0.5:
        return image
    # Convert angle range to a fraction of 180 degrees
    angle /= 180

    random_angle = tf.random.uniform([], -angle, angle)
    rot_layer.factor = (random_angle, random_angle)
    return rot_layer(image)


def _show_image(image, label=None):
    """
    For debugging purposes.
    """
    if not isinstance(image, list):
        image = [image]
        label = [label] if label is not None else None

    n = len(image)
    c = np.sqrt(n)
    c = int(c) if c.is_integer() else int(c) + 1
    r = n // c
    _, axes = plt.subplots(r, c, figsize=(3 * c, 3 * r))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, ax in enumerate(axes.flat):
        if i < n:
            if image[i].shape[-1] == 1:
                ax.imshow(image[i].numpy().squeeze(), cmap="gray")
            else:
                ax.imshow(image[i].numpy())
            ax.axis("off")
            if label is not None:
                ax.set_title(label[i])
    plt.show()


<<<<<<< HEAD
def _augment_image(image, augmentations=None):
=======
def augment_image(image, augmentations=None):
>>>>>>> feature_extraction_with_classification_fusion_layer
    """
    Applies data augmentation techniques to the image.
    Currently supported augmentations: flip_horizontal, flip_vertical, rotate.

    Args:
        image: Image tensor.
        augmentations: List of augmentation functions to apply to the image. The list
            format is [(augmentation_name, params), ...], where augmentation_name is the
            name of the augmentation function and params is a dictionary of parameters for
            the augmentation function. If the augmentation function does not require any
            parameters, just pass the function name as a string.
            For example, ['flip_horizontal', ('rotate', {'angle': 0.1})].

                Augmentation parameters are as follows:
                - flip_horizontal: No parameters
                - flip_vertical: No parameters
                - rotate: {'angle': float} where float is the angle range in degrees.

    Returns:
        Augmented image tensor.
    """
    if augmentations is None:
        return image

    assert isinstance(augmentations, list), "augmentations must be a list."
    assert isinstance(image, tf.Tensor), "image must be a TensorFlow tensor."

    aug_name_dict = {
        "flip_horizontal": _flip_horizontal,
        "flip_vertical": _flip_vertical,
        "rotate": _rotate,
    }
    for augmentation in augmentations:
        if isinstance(augmentation, str):
            aug_name = augmentation
            params = {}
        else:
            aug_name, params = augmentation
        assert (
            aug_name in aug_name_dict
        ), f"{aug_name} is not a valid augmentation.\
        Supported augmentations are: {list(aug_name_dict.keys())}"
        image = aug_name_dict[aug_name](image, **params)
    return image


def preprocess_dataset(
    image_paths,
    labels,
    batch_size=32,
    shuffle_buffer_size=1000,
    is_training=True,
    augmentations=None,
):
    """
    Creates a TensorFlow dataset for image preprocessing with optional data augmentation.

    Args:
        image_paths: List of image file paths.
        labels: List of image labels.
        batch_size: Batch size. Default is 32.
        shuffle_buffer_size: Buffer size for shuffling the dataset. Default is 1000.
        is_training: If True, the dataset is shuffled and augmented. Default is True.
        augmentations: List of augmentation functions to apply to the image. The list
            format is [(augmentation_name, params), ...], where augmentation_name is the
            name of the augmentation function and params is a dictionary of parameters for
            the augmentation function. If the augmentation function does not require any
            parameters, just pass the function name as a string.
            For example, ['flip_horizontal', ('rotate', {'angle': 0.1})].

                Augmentation parameters are as follows:
                - flip_horizontal: No parameters
                - flip_vertical: No parameters
                - rotate: {'angle': float} where float is the angle range in degrees.
    """
    # Create a TensorFlow Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle the dataset if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    else:
        assert (
            augmentations is None
        ), "augmentations must be None if is_training is False."

    def process_image(image_path, label):
        image = load_and_preprocess_image(
            image_path, target_size=(224, 224), preserve_aspect_ratio=True
        )
        if is_training and augmentations:
<<<<<<< HEAD
            image = _augment_image(image, augmentations)
=======
            image = augment_image(image, augmentations)
>>>>>>> feature_extraction_with_classification_fusion_layer
        return image, label

    # Map the processing function to each item
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch to improve performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

<<<<<<< HEAD
    return dataset
=======
    return dataset
>>>>>>> feature_extraction_with_classification_fusion_layer

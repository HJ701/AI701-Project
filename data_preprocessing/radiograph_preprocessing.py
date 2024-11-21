"""
Radiograph preprocessing functions.
"""

import tensorflow as tf


def load_and_preprocess_radiograph(
    image_path, target_size=(224, 224), preserve_aspect_ratio=True, include_edges=False
):
    """
    Reads a radiograph image, enhances contrast, (optionally) detects edges, and resizes
    it.

    Args:
        image_path: Path to the radiograph image file.
        target_size: The size to which the image will be resized. Default is (224, 224).
        preserve_aspect_ratio: If True, the aspect ratio of the image will be preserved
        when resizing. Default is True.
        include_edges: If True, the edges will be detected and combined with the image.
        Default is False.

    Returns:
        Processed image tensor.
    """
    # Read the image file
    image = tf.io.read_file(image_path)
    # Decode the image into a tensor (assuming grayscale radiographs)
    image = tf.image.decode_jpeg(image, channels=1)
    # Convert to floating-point tensors and rescale pixel values to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Enhance contrast
    image = enhance_contrast(image)
    # Detect edges
    if include_edges:
        edges = detect_edges(image)
        # Combine image and edges (optional)
        image = tf.add(image, edges)
    if preserve_aspect_ratio:
        # Resize the image while preserving the aspect ratio
        image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    else:
        # Resize the image to a standard size
        image = tf.image.resize(image, size=target_size)
    return image


def enhance_contrast(image):
    """
    Enhances the contrast of the image using histogram equalization.

    Args:
        image: Input image tensor.

    Returns:
        Image tensor with enhanced contrast.
    """
    # Apply adaptive histogram equalization
    image = tf.image.adjust_contrast(image, contrast_factor=2.0)
    return image


def detect_edges(image):
    """
    Detects edges in the image using the Sobel operator.
    """
    # Expand dims to add batch dimension: [height, width, channels] -> [1, height, width, channels]
    image = tf.expand_dims(image, axis=0)

    # Apply Sobel edge detection
    edges = tf.image.sobel_edges(image)  # Output shape: [1, height, width, channels, 2]

    # Remove the batch dimension
    edges = tf.squeeze(edges, axis=0)  # Now shape: [height, width, channels, 2]

    # Compute the magnitude of the edges
    edges = tf.sqrt(
        tf.reduce_sum(tf.square(edges), axis=-1)
    )  # Now shape: [height, width, channels]

    return edges


def preprocess_radiograph_dataset(
    image_paths, labels, batch_size=32, shuffle_buffer_size=1000, is_training=True
):
    """
    Creates a TensorFlow dataset for radiograph preprocessing.
    """
    # Create a TensorFlow Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle the dataset if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    def process_image(image_path, label):
        image = load_and_preprocess_radiograph(
            image_path, target_size=(224, 224), preserve_aspect_ratio=True
        )
        return image, label

    # Map the processing function to each item
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch to improve performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


class RadiographFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(RadiographFeatureExtractor, self).__init__()
        # Load pre-trained ResNet50 without the top classifier layer
        self.base_model = ResNet50(include_top=False, weights="imagenet", pooling=None)

        # Our images are converted to RGB in preprocessing

        # Adaptive Pooling equivalent in TensorFlow
        self.adaptive_pool = layers.GlobalAveragePooling2D()

        # Fully Connected Layers for Dimensionality Reduction
        self.fc_reduction = models.Sequential(
            [
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(2048),
            ]
        )

    def call(self, x, training=False):
        features = self.base_model(
            x, training=training
        )  # Shape: [batch_size, H, W, 2048]

        # Adaptive Pooling
        pooled_features = self.adaptive_pool(features)  # Shape: [batch_size, 2048]

        # Fully Connected Layers for Dimensionality Reduction
        reduced_features = self.fc_reduction(
            pooled_features, training=training
        )  # Shape: [batch_size, 2048]

        return reduced_features


# test the RadiographFeatureExtractor model
if __name__ == "__main__":
    # Create an instance of the RadiographFeatureExtractor
    feature_extractor = RadiographFeatureExtractor()

    # Generate a random grayscale image tensor
    random_image = tf.random.normal((1, 224, 224, 3))

    # Obtain the reduced features
    reduced_features = feature_extractor(random_image)

    # Display the shape of the reduced features
    print("Shape of reduced features:", reduced_features.shape)
    # Expected output: Shape of reduced features: (1, 2048)

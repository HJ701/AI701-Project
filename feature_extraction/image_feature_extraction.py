import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B3, EfficientNetV2B3


class IntraoralFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(IntraoralFeatureExtractor, self).__init__()
        # Load pre-trained EfficientNetV2-B3 without the top classifier layer
        self.base_model = EfficientNetV2B3(
            include_top=False, weights="imagenet", pooling=None
        )

        # Channel Attention Module
        self.channel_attention = models.Sequential(
            [
                layers.Conv2D(1536, kernel_size=1, padding="same"),
                layers.Activation("sigmoid"),
            ]
        )

        # Spatial Attention Module
        self.spatial_attention = models.Sequential(
            [
                layers.Conv2D(1536, kernel_size=7, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(1536, kernel_size=1, padding="same"),
                layers.Activation("sigmoid"),
            ]
        )

        # Adaptive Pooling equivalent in TensorFlow
        self.adaptive_pool = layers.GlobalAveragePooling2D()

        # Fully Connected Layers for Dimensionality Reduction
        self.fc_reduction = models.Sequential(
            [
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1536),
            ]
        )

    def call(self, x, training=False):
        features = self.base_model(
            x, training=training
        )  # Output shape: [batch_size, H, W, 1536]

        # Channel Attention
        channel_attn = self.channel_attention(
            features
        )  # Shape: [batch_size, H, W, 1536]
        features = layers.multiply([features, channel_attn])  # Apply channel attention

        # Spatial Attention
        spatial_attn = self.spatial_attention(
            features
        )  # Shape: [batch_size, H, W, 1536]
        features = layers.multiply([features, spatial_attn])  # Apply spatial attention

        # Adaptive Pooling
        pooled_features = self.adaptive_pool(features)  # Shape: [batch_size, 1536]

        # Fully Connected Layers for Dimensionality Reduction
        reduced_features = self.fc_reduction(
            pooled_features, training=training
        )  # Shape: [batch_size, 1536]

        return reduced_features


# test the IntraoralFeatureExtractor model
if __name__ == "__main__":
    # Create an instance of the IntraoralFeatureExtractor
    feature_extractor = IntraoralFeatureExtractor()

    # Generate a random image tensor
    random_image = tf.random.normal((1, 224, 224, 3))

    # Obtain the reduced features
    reduced_features = feature_extractor(random_image)

    # Display the shape of the reduced features
    print("Shape of reduced features:", reduced_features.shape)
    # Expected output: Shape of reduced features: (1, 1536)

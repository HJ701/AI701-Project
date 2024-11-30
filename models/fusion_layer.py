# models/fusion_layer_tf.py

import tensorflow as tf
from tensorflow.keras import layers, models


class FusionLayer(tf.keras.Model):
    def __init__(self, projection_dim=512, num_heads=8, dropout_rate=0.5, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Linear layers to project features to a common dimension
        self.proj_intraoral = layers.Dense(projection_dim, activation=None)
        self.proj_radiograph = layers.Dense(projection_dim, activation=None)
        self.proj_text = layers.Dense(projection_dim, activation=None)

        # Attention Mechanism
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )

        # Fully Connected Layers for Dimensionality Reduction
        self.fc = models.Sequential(
            [
                layers.Dense(projection_dim, activation="relu"),
                layers.Dropout(dropout_rate),
            ]
        )

    def build(self, input_shape):
        # No weights or initialization needed
        super(FusionLayer, self).build(input_shape)

    def call(self, intraoral_feat, radiograph_feat, training=False):
        # Project features
        intraoral_proj = self.proj_intraoral(intraoral_feat)  # Shape: [batch_size, 512]
        radiograph_proj = self.proj_radiograph(
            radiograph_feat
        )  # Shape: [batch_size, 512]

        # Stack projected features for attention
        # Expand dims to add sequence length (3)
        features = tf.stack(
            [intraoral_proj, radiograph_proj], axis=1
        )  # Shape: [batch_size, 2, 512]

        # Apply attention
        attn_output = self.attention(
            features, features, features, training=training
        )  # Shape: [batch_size, 2, 512]

        # Aggregate features (e.g., mean pooling)
        fused = tf.reduce_mean(attn_output, axis=1)  # Shape: [batch_size, 512]

        # Pass through FC layers
        fused = self.fc(fused, training=training)  # Shape: [batch_size, 512]

        return fused

    def get_config(self):
        # Return all arguments required to reconstruct the model
        config = super(FusionLayer, self).get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Test the FusionLayer
if __name__ == "__main__":
    # Create an instance of the FusionLayer
    fusion_layer = FusionLayer()

    # Generate random feature tensors
    intraoral_feat = tf.random.normal((32, 2048))
    radiograph_feat = tf.random.normal((32, 2048))
    text_feat = tf.random.normal((32, 768))

    # Obtain the fused features
    fused_features = fusion_layer(intraoral_feat, radiograph_feat, text_feat)

    # Display the shape of the fused features
    print("Shape of fused features:", fused_features.shape)
    # Expected output: Shape of fused features: (32, 512)

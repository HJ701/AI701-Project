# models/fusion_layer_tf.py

import tensorflow as tf
from tensorflow.keras import layers, models

class FusionLayer(tf.keras.Model):
    def __init__(self, feature_dims, projection_dim=512, num_heads=8, dropout_rate=0.5):
        super(FusionLayer, self).__init__()
        # Linear layers to project features to a common dimension
        self.proj_intraoral = layers.Dense(projection_dim, activation=None)
        self.proj_radiograph = layers.Dense(projection_dim, activation=None)
        self.proj_text = layers.Dense(projection_dim, activation=None)
        
        # Attention Mechanism
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        
        # Fully Connected Layers for Dimensionality Reduction
        self.fc = models.Sequential([
            layers.Dense(projection_dim, activation='relu'),
            layers.Dropout(dropout_rate)
        ])
    
    def call(self, intraoral_feat, radiograph_feat, text_feat, training=False):
        # Project features
        intraoral_proj = self.proj_intraoral(intraoral_feat)  # Shape: [batch_size, 512]
        radiograph_proj = self.proj_radiograph(radiograph_feat)  # Shape: [batch_size, 512]
        text_proj = self.proj_text(text_feat)  # Shape: [batch_size, 512]
        
        # Stack projected features for attention
        # Expand dims to add sequence length (3)
        features = tf.stack([intraoral_proj, radiograph_proj, text_proj], axis=1)  # Shape: [batch_size, 3, 512]
        
        # Apply attention
        attn_output = self.attention(features, features, features, training=training)  # Shape: [batch_size, 3, 512]
        
        # Aggregate features (e.g., mean pooling)
        fused = tf.reduce_mean(attn_output, axis=1)  # Shape: [batch_size, 512]
        
        # Pass through FC layers
        fused = self.fc(fused, training=training)  # Shape: [batch_size, 512]
        
        return fused
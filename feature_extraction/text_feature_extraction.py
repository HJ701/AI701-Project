import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFBertModel


class TextFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.bert_model = TFBertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # Optionally, freeze BERT layers to prevent training
        self.bert_model.trainable = False

        # Additional layers can be added here if needed
        self.fc = layers.Dense(512, activation="relu")  # Project to 512 dimensions
        self.dropout = layers.Dropout(0.5)

    def call(self, input_ids, attention_mask, training=False):
        output = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, training=training
        )
        cls_embeddings = output.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]
        projected = self.fc(
            cls_embeddings, training=training
        )  # Shape: [batch_size, 512]
        projected = self.dropout(projected, training=training)
        return projected  # Shape: [batch_size, 512]


# Test the TextFeatureExtractor model
if __name__ == "__main__":
    # Create an instance of the TextFeatureExtractor
    feature_extractor = TextFeatureExtractor()

    # Generate a random batch of input_ids and attention_mask
    input_ids = tf.random.uniform((32, 128), maxval=28996, dtype=tf.int32)
    attention_mask = tf.random.uniform((32, 128), maxval=2, dtype=tf.int32)

    # Obtain the projected features
    projected_features = feature_extractor(input_ids, attention_mask)

    # Display the shape of the projected features
    print("Shape of projected features:", projected_features.shape)
    # Expected output: Shape of projected features: (32, 512)

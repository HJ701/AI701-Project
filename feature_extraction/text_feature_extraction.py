import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import TFBertModel

class TextFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.bert_model = TFBertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        # Optionally, freeze BERT layers to prevent training
        self.bert_model.trainable = False
        
        # Additional layers can be added here if needed
        self.fc = layers.Dense(512, activation='relu')  # Project to 512 dimensions
        self.dropout = layers.Dropout(0.5)

    def call(self, input_ids, attention_mask, training=False):
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, training=training)
        cls_embeddings = output.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]
        projected = self.fc(cls_embeddings, training=training)  # Shape: [batch_size, 512]
        projected = self.dropout(projected, training=training)
        return projected  # Shape: [batch_size, 512]

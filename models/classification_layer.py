# models/classification_layer_tf.py

import tensorflow as tf
from tensorflow.keras import layers, models

class ClassificationLayer(tf.keras.Model):
    def __init__(self, input_dim, num_classes_iotn, num_classes_malocclusion, num_classes_subclass):
        """
        Args:
            input_dim (int): Dimension of the fused feature vector.
            num_classes_iotn (int): Number of classes for IOTN Grade.
            num_classes_malocclusion (int): Number of classes for Malocclusion Class.
            num_classes_subclass (int): Number of subclass diagnoses (multi-label).
        """
        super(ClassificationLayer, self).__init__()
        # Classification head for IOTN Grade
        self.classifier_iotn = models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes_iotn)
        ])
        
        # Classification head for Malocclusion Class
        self.classifier_malocclusion = models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes_malocclusion)
        ])
        
        # Classification head for Subclass Diagnoses (multi-label)
        self.classifier_subclass = models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes_subclass)
        ])

    def call(self, inputs):
        """
        Forward pass for multi-task classification.

        Args:
            inputs (Tensor): Fused feature tensor of shape [batch_size, input_dim].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 
                - outputs_iotn: [batch_size, num_classes_iotn]
                - outputs_malocclusion: [batch_size, num_classes_malocclusion]
                - outputs_subclass: [batch_size, num_classes_subclass]
        """
        outputs_iotn = self.classifier_iotn(inputs)
        outputs_malocclusion = self.classifier_malocclusion(inputs)
        outputs_subclass = self.classifier_subclass(inputs)
        return outputs_iotn, outputs_malocclusion, outputs_subclass

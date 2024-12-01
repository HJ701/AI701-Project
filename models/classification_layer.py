# models/classification_layer_tf.py

import tensorflow as tf
from tensorflow.keras import layers, models


class ClassificationLayer(tf.keras.Model):
    def __init__(
        self, num_classes_iotn, num_classes_malocclusion, num_classes_subclass
    ):
        """
        Args:
            num_classes_iotn (int): Number of classes for IOTN Grade.
            num_classes_malocclusion (int): Number of classes for Malocclusion Class.
            num_classes_subclass (int): Number of subclass diagnoses (multi-label).
        """
        super(ClassificationLayer, self).__init__()
        # Classification head for IOTN Grade
        self.classifier_iotn = models.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes_iotn),
            ]
        )

        # Classification head for Malocclusion Class
        self.classifier_malocclusion = models.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes_malocclusion),
            ]
        )

        # Classification head for Subclass Diagnoses (multi-label)
        self.classifier_subclass = models.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes_subclass),
            ]
        )

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


# Test the ClassificationLayer
if __name__ == "__main__":
    # Create an instance of the ClassificationLayer
    classification_layer = ClassificationLayer(
        num_classes_iotn=3, num_classes_malocclusion=4, num_classes_subclass=5
    )

    # Generate a random batch of fused features
    fused_features = tf.random.normal((32, 512))

    # Obtain the classification outputs
    outputs_iotn, outputs_malocclusion, outputs_subclass = classification_layer(
        fused_features
    )

    # Display the shape of the classification outputs
    print("Shape of IOTN Grade outputs:", outputs_iotn.shape)
    print("Shape of Malocclusion Class outputs:", outputs_malocclusion.shape)
    print("Shape of Subclass Diagnoses outputs:", outputs_subclass.shape)
    # Expected output:
    # Shape of IOTN Grade outputs: (32, 3)
    # Shape of Malocclusion Class outputs: (32, 4)
    # Shape of Subclass Diagnoses outputs: (32, 5)

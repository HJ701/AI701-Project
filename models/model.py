"""
Integrated model for the project
"""

import tensorflow as tf

from feature_extraction import IntraoralFeatureExtractor, RadiographFeatureExtractor
from models import ClassificationLayer, FusionLayer


class OrthoNet(tf.keras.Model):
    """
    OrthoNet model for multi-task classification. The model consists of two feature
    extractors and has three classification heads for IOTN Grade, Malocclusion Class,
    and Subclass Diagnoses. For Subclass Diagnoses, we use a multi-label classification
    approach.

    Args:
        n_classes_iotn (int): Number of classes for IOTN Grade.
        n_classes_malocclusion (int): Number of classes for Malocclusion Class.
        n_classes_subclass (int): Number of subclass diagnoses (multi-label).
        projection_dim (int): Dimensionality of the projected features.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate for the model.
    """

    def __init__(
        self,
        n_classes_iotn,
        n_classes_malocclusion,
        n_classes_subclass,
        projection_dim=512,
        num_heads=8,
        dropout_rate=0.5,
        **kwargs,
    ):
        super(OrthoNet, self).__init__(**kwargs)
        self.n_classes_iotn = n_classes_iotn
        self.n_classes_malocclusion = n_classes_malocclusion
        self.n_classes_subclass = n_classes_subclass
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.image_feature_extractor = IntraoralFeatureExtractor()
        self.radio_feature_extractor = RadiographFeatureExtractor()
        self.fusion = FusionLayer(
            projection_dim=projection_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.classification = ClassificationLayer(
            n_classes_iotn, n_classes_malocclusion, n_classes_subclass
        )

    def get_config(self):
        # Return all arguments required to reconstruct the model
        config = super(OrthoNet, self).get_config()
        config.update(
            {
                "n_classes_iotn": self.n_classes_iotn,
                "n_classes_malocclusion": self.n_classes_malocclusion,
                "n_classes_subclass": self.n_classes_subclass,
                "projection_dim": self.projection_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        images, radiographs = inputs
        # Image feature extraction
        image_features = self.image_feature_extractor(images)
        # Radiograph feature extraction
        radiograph_features = self.radio_feature_extractor(radiographs)
        # Fusion
        fused_features = self.fusion(image_features, radiograph_features)
        # Classification
        outputs_iotn, outputs_malocclusion, outputs_subclass = self.classification(
            fused_features
        )

        # iotn and malocclusion outputs are single-label classification
        # outputs_iotn = tf.nn.softmax(outputs_iotn)
        outputs_iotn = tf.nn.sigmoid(outputs_iotn)
        outputs_malocclusion = tf.nn.softmax(outputs_malocclusion)
        # subclass outputs are multi-label classification
        outputs_subclass = tf.nn.sigmoid(outputs_subclass)

        return outputs_iotn, outputs_malocclusion, outputs_subclass


# Test the OrthoNet model
if __name__ == "__main__":
    # Create an instance of the OrthoNet model
    ortho_net = OrthoNet(
        n_classes_iotn=5, n_classes_malocclusion=3, n_classes_subclass=10
    )

    # Generate random image and radiograph tensors
    random_image = tf.random.normal((1, 224, 224, 3))
    random_radiograph = tf.random.normal((1, 224, 224, 3))

    # Obtain the model predictions
    outputs_iotn, outputs_malocclusion, outputs_subclass = ortho_net(
        random_image, random_radiograph
    )

    # Display the shape of the outputs
    print("Shape of IOTN outputs:", outputs_iotn.shape)
    print("Shape of Malocclusion outputs:", outputs_malocclusion.shape)
    print("Shape of Subclass outputs:", outputs_subclass.shape)

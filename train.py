import os
import pickle
from argparse import ArgumentParser
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import metrics
import keras

from data_preprocessing import get_dataset
from models import OrthoNet


def parse_args():
    """
    Parse the arguments for training the model from the command line.

    Example:
    python train.py -batch_size 32 -epochs 10 -lr 1e-4 -optimizer adam -exp_dir experiments/ -exp_name exp
    """
    parser = ArgumentParser(description="Train OrthoNet on the dataset")
    parser.add_argument(
        "-batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "-epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "-lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "-optimizer", type=str, default="adam", help="Optimizer for training"
    )
    parser.add_argument(
        "-exp_dir", type=str, default="experiments/", help="Experiment directory"
    )
    parser.add_argument("-exp_name", type=str, default="exp", help="Experiment name")
    return parser.parse_args()


optimizer_map = {
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
}

@keras.saving.register_keras_serializable()
def categorical_focal_loss_with_one_hot(num_classes, gamma=2.0, alpha=0.25, from_logits=True):
    """
    A wrapper for CategoricalFocalCrossentropy that converts sparse labels to one-hot encoding.

    Args:
        num_classes (int): Number of classes.
        gamma (float): Focusing parameter for focal loss.
        alpha (float): Balancing parameter for focal loss.
        from_logits (bool): Whether predictions are logits or probabilities.

    Returns:
        A loss function that converts sparse labels to one-hot and computes the focal loss.
    """
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
        gamma=gamma, alpha=alpha, from_logits=from_logits
    )

    def loss_fn(y_true, y_pred):
        # Convert sparse labels to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        # Compute the focal loss
        return focal_loss(y_true_one_hot, y_pred)

    return loss_fn


@keras.saving.register_keras_serializable()
class CategoricalFocalLossWithOneHot(tf.keras.losses.Loss):
    def __init__(self, num_classes, gamma=2.0, alpha=0.25, from_logits=True, name="categorical_focal_loss"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=gamma, alpha=alpha, from_logits=from_logits
        )

    def call(self, y_true, y_pred):
        # Convert sparse labels to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        # Compute the focal loss
        return self.focal_loss(y_true_one_hot, y_pred)

    def get_config(self):
        # Include all arguments needed to reconstruct the object
        config = {
            "num_classes": self.num_classes,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits,
            "name": self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct the object using the provided configuration
        return cls(**config)

class SaveBestModelOnCombinedMetric(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor_metrics):
        """
        Callback to save the model with the best performance based on the sum of specified metrics.

        Args:
            filepath (str): Path to save the model.
            monitor_metrics (list): List of metric names to sum for monitoring.
            mode (str): "max" to save the model with the highest combined metric, "min" for the lowest.
        """
        super(SaveBestModelOnCombinedMetric, self).__init__()
        self.filepath = filepath
        self.monitor_metrics = monitor_metrics
        self.best = -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Compute the sum of the specified metrics
        combined_metric = sum([logs.get(metric, 0) for metric in self.monitor_metrics])
        if not isinstance(combined_metric, (int, float)):
            combined_metric = combined_metric.numpy().sum()
        if combined_metric > self.best:
            self.best = combined_metric
            self.model.save(self.filepath)
            # self.model.save_weights(self.filepath)
            print(f"Epoch {epoch + 1}: saving model with combined metric = {combined_metric:.4f}")



def main():
    """
    Main function to train the model.
    """
    args = parse_args()

    # create experiment directory
    exp_name = f"{args.exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"  # add timestamp to experiment name
    exp_dir = os.path.join(args.exp_dir, exp_name)
    os.makedirs(
        exp_dir, exist_ok=False
    )  # create experiment directory and fail if it already exists

    # Load the dataset
    train_ds = get_dataset(
        csv_path="Processed_Samples/consolidated_full_data.csv",
        imgs_path="../orthoai_patient_records/orthoai_patient_records/",
        split="train",
        batch_size=args.batch_size,
    )
    val_ds = get_dataset(
        csv_path="Processed_Samples/consolidated_full_data.csv",
        imgs_path="../orthoai_patient_records/orthoai_patient_records/",
        split="validation",
        batch_size=args.batch_size,
    )

    # Initialize the model (1 class means binary classification)
    model = OrthoNet(n_classes_iotn=1, n_classes_malocclusion=3, n_classes_subclass=16)#, projection_dim=128,
        # num_heads=4, dropout_rate=0.5)

    # get the optimizer
    if not args.optimizer in optimizer_map:
        raise ValueError(
            f"Invalid optimizer: {args.optimizer}. Use one of {list(optimizer_map.keys())}"
        )
    optimizer = optimizer_map[args.optimizer](learning_rate=args.lr)

    class F1ScoreMetric(metrics.Metric):
        def __init__(self, num_classes, name="f1_score", **kwargs):
            super(F1ScoreMetric, self).__init__(name=name, **kwargs)
            self.num_classes = num_classes
            self.f1_metric = tf.keras.metrics.F1Score()

        def update_state(self, y_true, y_pred, sample_weight=None):
            # Convert y_true to one-hot encoding if it's not already
            if len(y_true.shape) == 1 or y_true.shape[-1] != self.num_classes:
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)

            # Update state for F1 metric
            self.f1_metric.update_state(y_true, y_pred, sample_weight)

        def result(self):
            return self.f1_metric.result()

        def reset_states(self):
            self.f1_metric.reset_states()

    class F1ScoreBinary(metrics.Metric):
        def __init__(self, name="f1_score", threshold=0.5, **kwargs):
            super(F1ScoreBinary, self).__init__(name=name, **kwargs)
            self.threshold = threshold
            self.true_positives = self.add_weight(name="tp", initializer="zeros")
            self.false_positives = self.add_weight(name="fp", initializer="zeros")
            self.false_negatives = self.add_weight(name="fn", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            # Ensure consistent data types
            y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
            y_pred = tf.cast(y_pred >= self.threshold, tf.float32)  # Threshold y_pred and cast to float32

            # Update true positives, false positives, and false negatives
            self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred))
            self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
            self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

        def result(self):
            precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
            recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
            f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            return f1

        def reset_states(self):
            self.true_positives.assign(0)
            self.false_positives.assign(0)
            self.false_negatives.assign(0)

    # Compile the model. Loss functions and metrics could be changed as needed.
    model.compile(
        optimizer=optimizer,
        loss=[
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
        ],
        metrics=[
            F1ScoreBinary(),
            F1ScoreMetric(num_classes=3),
            F1ScoreBinary(),
        ],
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
            # tf.keras.callbacks.ModelCheckpoint(
            #     os.path.join(exp_dir, "model.keras"), save_best_only=True, monitor="val_binary_accuracy", mode="max"
            # ),
            SaveBestModelOnCombinedMetric(filepath=os.path.join(exp_dir, "model.keras"),
                                          monitor_metrics=["val_binary_accuracy", "val_accuracy", "val_binary_accuracy_1"]),
        ],
    )

    # Save the training history
    with open(os.path.join(exp_dir, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()

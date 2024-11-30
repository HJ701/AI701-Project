import os
import pickle
from argparse import ArgumentParser
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import metrics

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
        path="Processed_Samples/consolidated_data.csv",
        split="train",
        batch_size=args.batch_size,
    )
    val_ds = get_dataset(
        path="Processed_Samples/consolidated_data.csv",
        split="validation",
        batch_size=args.batch_size,
    )

    # Initialize the model (1 class means binary classification)
    model = OrthoNet(n_classes_iotn=1, n_classes_malocclusion=3, n_classes_subclass=13)

    # get the optimizer
    if not args.optimizer in optimizer_map:
        raise ValueError(
            f"Invalid optimizer: {args.optimizer}. Use one of {list(optimizer_map.keys())}"
        )
    optimizer = optimizer_map[args.optimizer](learning_rate=args.lr)

    # Compile the model. Loss functions and metrics could be changed as needed.
    model.compile(
        optimizer=optimizer,
        loss=[
            tf.keras.losses.BinaryCrossentropy(),
            tf.keras.losses.SparseCategoricalCrossentropy(),
            tf.keras.losses.BinaryCrossentropy(),
        ],
        metrics=[
            metrics.BinaryAccuracy(threshold=0.5),
            "accuracy",
            metrics.BinaryAccuracy(threshold=0.5),
        ],
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(exp_dir, "model.keras"), save_best_only=True
            ),
        ],
    )

    # Save the training history
    with open(os.path.join(exp_dir, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()

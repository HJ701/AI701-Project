import itertools
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_preprocessing import get_dataset, malocclusion_mapping
from models import OrthoNet


def to_one_hot(y_true, num_classes):
    return np.eye(num_classes)[y_true]


def roc_auc_with_missing(y_true, y_pred):
    auc_scores = 0.0
    den = y_pred.shape[1]
    y_true = to_one_hot(y_true, y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        if np.all(y_true[:, i] == 0):  # If the class is missing in y_true
            den -= 1
        else:
            auc_scores += roc_auc_score(y_true[:, i], y_pred[:, i])
    return auc_scores / den


def parse_args():
    '''
    Extracts the arguments from the command line.
    Example:
    python test.py -batch_size 32 -model_path experiments/exp/model.keras
    '''
    parser = ArgumentParser(description="Evaluate OrthoNet on the test set")
    parser.add_argument(
        "-batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "-model_path",
        type=str,
        default="experiments/exp/model.keras",
        help="Path to the model",
    )
    parser.add_argument(
        "-split",
        type=str,
        default="test",
        help="Split to evaluate the model on (train, validation, test)",
    )
    return parser.parse_args()


def convert_to_one_hot(y_true, num_classes):
    return tf.one_hot(y_true, depth=num_classes)


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


if __name__ == "__main__":
    args = parse_args()

    # Evaluate the model
    test_ds = get_dataset(
        csv_path="Processed_Samples/consolidated_full_data.csv",
        imgs_path="../orthoai_patient_records/orthoai_patient_records/",
        split=args.split,
        batch_size=args.batch_size,
    )

    # use the best model
    if args.model_path.endswith(".h5"):
        model = OrthoNet(n_classes_iotn=1, n_classes_malocclusion=3, n_classes_subclass=16)
        model.load_weights(args.model_path)
    else:
        assert args.model_path.endswith(".keras")
        model = tf.keras.models.load_model(args.model_path)

    # get labels for the test set
    y_iotn, y_malocclusion, y_subclass = [], [], []
    for _, (y1, y2, y3) in test_ds:
        y_iotn.extend(y1.numpy())
        y_malocclusion.extend(y2.numpy())
        y_subclass.extend(y3.numpy())

    # to np arrays
    y_iotn = tf.convert_to_tensor(y_iotn).numpy()
    y_malocclusion = tf.convert_to_tensor(y_malocclusion).numpy()
    y_subclass = tf.convert_to_tensor(y_subclass).numpy()

    # get predictions
    y_pred_iotn, y_pred_malocclusion, y_pred_subclass = model.predict(test_ds)
    y_thresh_iotn = (y_pred_iotn > 0.5).astype(int)
    y_thresh_malocclusion = tf.argmax(y_pred_malocclusion, axis=1).numpy()
    y_thresh_subclass = (y_pred_subclass > 0.5).astype(int)

    # accuracy
    acc_iotn = accuracy_score(y_iotn, y_thresh_iotn)
    acc_malocclusion = accuracy_score(y_malocclusion, y_thresh_malocclusion)
    # per subclass accuracy
    acc_subclass = []
    for i in range(y_subclass.shape[1]):
        acc_subclass.append(accuracy_score(y_subclass[:, i], y_thresh_subclass[:, i]))

    # f1 score
    f1_iotn = f1_score(y_iotn, y_thresh_iotn, average="macro")
    f1_malocclusion = f1_score(y_malocclusion, y_thresh_malocclusion, average="macro")
    f1_subclass = f1_score(
        y_subclass, y_thresh_subclass, average=None, zero_division=np.nan
    )

    # precision
    precision_iotn = precision_score(
        y_iotn, y_thresh_iotn, average="macro", zero_division=np.nan
    )
    precision_malocclusion = precision_score(
        y_malocclusion, y_thresh_malocclusion, average="macro", zero_division=np.nan
    )
    precision_subclass = precision_score(
        y_subclass, y_thresh_subclass, average=None, zero_division=np.nan
    )

    # recall
    recall_iotn = recall_score(
        y_iotn, y_thresh_iotn, average="macro", zero_division=np.nan
    )
    recall_malocclusion = recall_score(
        y_malocclusion, y_thresh_malocclusion, average="macro", zero_division=np.nan
    )
    recall_subclass = recall_score(
        y_subclass, y_thresh_subclass, average=None, zero_division=np.nan
    )

    # roc auc
    auc_iotn = roc_auc_score(y_iotn, y_pred_iotn)
    auc_malocclusion = roc_auc_with_missing(y_malocclusion, y_pred_malocclusion)
    valid_cols_subclass = np.any(y_subclass, axis=0)
    auc_subclass = roc_auc_score(
        y_subclass[:, valid_cols_subclass],
        y_pred_subclass[:, valid_cols_subclass],
        average=None,
    )

    # metrics to a pandas dataframe, subclass metrics will be saved to another dataframe
    df = pd.DataFrame(columns=["Metric", "IOTN Grade", "Malocclusion Class"])

    metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]
    iotn_metrics = [acc_iotn, f1_iotn, precision_iotn, recall_iotn, auc_iotn]
    malocclusion_metrics = [
        acc_malocclusion,
        f1_malocclusion,
        precision_malocclusion,
        recall_malocclusion,
        auc_malocclusion,
    ]

    df["Metric"] = metrics
    df["IOTN Grade"] = iotn_metrics
    df["Malocclusion Class"] = malocclusion_metrics

    subclass_metrics = pd.DataFrame(
        columns=["Subclass", "F1 Score", "Precision", "Recall", "ROC AUC"]
    )

    data = pd.read_csv("Processed_Samples/consolidated_full_data.csv")
    subclasses = data.columns[6:]
    subclass_metrics["Subclass"] = subclasses
    subclass_metrics["F1 Score"] = f1_subclass
    subclass_metrics["Precision"] = precision_subclass
    subclass_metrics["Recall"] = recall_subclass
    idx = 0
    for i in range(len(subclass_metrics)):
        if valid_cols_subclass[i]:
            subclass_metrics.loc[i, "ROC AUC"] = auc_subclass[idx]
            idx += 1

    # save metrics to csv
    exp_path = os.path.dirname(args.model_path)
    df.to_csv(os.path.join(exp_path, "test_results_iotn_malocclusion.csv"), index=False)
    subclass_metrics.to_csv(
        os.path.join(exp_path, "test_results_subclasses.csv"), index=False
    )

    # confusion matrces
    confusion_iotn = confusion_matrix(y_iotn, y_thresh_iotn, labels=[0, 1])
    confusion_malocclusion = confusion_matrix(
        y_malocclusion,
        y_thresh_malocclusion,
        labels=list(malocclusion_mapping.values()),
    )
    confusion_subclass = []
    for i in range(y_subclass.shape[1]):
        confusion_subclass.append(
            confusion_matrix(y_subclass[:, i], y_thresh_subclass[:, i], labels=[0, 1])
        )

    # plot and save confusion matrices
    os.makedirs(os.path.join(exp_path, "confusion_matrices"), exist_ok=False)
    plot_confusion_matrix(
        confusion_iotn,
        classes=[0, 1],
        title="IOTN Grade Confusion Matrix",
    )
    plt.savefig(os.path.join(exp_path, "confusion_matrices", "iotn_grade.png"))
    plt.close()

    plot_confusion_matrix(
        confusion_malocclusion,
        classes=list(malocclusion_mapping.values()),
        title="Malocclusion Class Confusion Matrix",
    )
    plt.savefig(os.path.join(exp_path, "confusion_matrices", "malocclusion_class.png"))
    plt.close()

    for i, conf in enumerate(confusion_subclass):
        plot_confusion_matrix(
            conf, classes=[0, 1], title=f"Subclass {subclasses[i]} Confusion Matrix"
        )
        plt.savefig(
            os.path.join(
                exp_path, "confusion_matrices", f"subclass_{subclasses[i]}.png"
            )
        )
        plt.close()

'''
2.2017
val_accuracy: 0.6000 - val_binary_accuracy: 0.7333 - val_binary_accuracy_1: 0.8683 - val_binary_crossentropy_loss: 0.5354 - val_loss: 2.9944 - val_sparse_categorical_crossentropy_loss: 1.3520
'''
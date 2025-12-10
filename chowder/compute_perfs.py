"""Compute Chowder performance metrics from predictions on IBDColEpi."""

from pathlib import Path
import argparse
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Chowder performance metrics from predictions on IBDColEpi."
    )
    parser.add_argument(
        "--preds_path",
        type=str,
        default="./chowder_preds/ibdcolepi_preds.csv",
        help="Path to CSV file containing predictions.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Threshold to convert probabilities to binary predictions.",
    )
    parser.add_argument(
        "--cm_normalize",
        type=str,
        default=None,
        choices=[None, "true", "pred", "all"],
        help="Normalization for confusion matrix.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./chowder_perfs",
        help="Directory to save performance metrics and confusion matrix.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    preds_path = Path(args.preds_path)
    save_dir = Path(args.save_dir)
    assert preds_path.exists(), f"Predictions file not found at {preds_path}"
    save_dir.mkdir(parents=True, exist_ok=True)

    df_preds = pd.read_csv(preds_path, index_col=0)
    df_preds["preds"] = (df_preds["probs"] >= args.thr).astype(int)
    df_preds = df_preds.dropna()
    logger.info(f"Total samples: {len(df_preds['label'])}")
    for lbl in np.unique(df_preds["label"]):
        print(
            f"Class {lbl}: {(df_preds['label']==lbl).sum()} samples, {100*(df_preds['label']==lbl).sum()/len(df_preds['label']):.2f}%")

    # Compute accuracy
    acc = accuracy_score(df_preds["label"], df_preds["preds"])
    balanced_acc = balanced_accuracy_score(df_preds["label"], df_preds["preds"])
    # Compute confusion matrix
    cm = confusion_matrix(df_preds["label"], df_preds["preds"], normalize=args.cm_normalize)
    # Classification report (precision, recall, f1)
    report = classification_report(df_preds["label"], df_preds["preds"])
    # ROC AUC
    roc_auc = roc_auc_score(df_preds["label"], df_preds["probs"])

    logger.info(f"Accuracy: {acc}")
    logger.info(f"Balanced Accuracy: {balanced_acc}")
    logger.info(f"ROC AUC: {roc_auc}")
    logger.info(f"Classification Report:\n{report}")

    with open(save_dir / "performance_metrics.txt", "w") as f:
        f.write("Performance Metrics:\n")
        f.write("Using threshold on probabilities: {:.2f}\n".format(args.thr))
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Balanced Accuracy: {balanced_acc}\n")
        f.write(f"ROC AUC: {roc_auc}\n")
        f.write(f"Classification Report:\n{report}\n")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(df_preds["label"]))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (threshold={args.thr})")
    plt.savefig(save_dir / "confusion_matrix.png")
    
    logger.info(f"Performance metrics and confusion matrix saved to {args.save_dir}")

if __name__ == "__main__":
    main()
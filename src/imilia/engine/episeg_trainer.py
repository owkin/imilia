"""Cross-validation trainer for logistic regression on IBDColEPI data."""

import gc
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm


class EpiSegTrainer:
    """Trainer for cross-validation with logistic regression."""

    def __init__(
        self,
        n_folds: int = 5,
        c_values: list[float] | None = None,
        random_state: int = 0,
        max_iter: int = 1000,
    ) -> None:
        """Initialize the trainer.

        Args:
            n_folds: Number of cross-validation folds
            c_values: List of C values for logistic regression regularization.
                     If None, defaults to logspace(-3, 0, 4)
            random_state: Random state for reproducibility
            max_iter: Maximum iterations for logistic regression
        """
        self.n_folds = n_folds
        self.c_values = c_values if c_values is not None else np.logspace(-3, 0, 4).tolist()
        self.random_state = random_state
        self.max_iter = max_iter

    def cross_validate(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        image_index: torch.Tensor,
        n_train_images: int,
    ) -> dict[str, Any]:
        """Perform cross-validation and return results.

        Args:
            xs: Feature matrix of shape (n_samples, n_features)
            ys: Label vector of shape (n_samples,)
            image_index: Index of the image each sample comes from
            n_train_images: Total number of training images

        Returns:
            Dictionary containing:
                - 'mean_precision_all': Mean precision curves for each C value
                - 'std_precision_all': Std precision curves for each C value
                - 'mean_recall': Recall values for the curves
                - 'average_precisions': Average precision scores for each C value
        """
        n_c = len(self.c_values)
        n_points = 1000
        mean_precision_all = np.zeros((n_c, n_points))
        std_precision_all = np.zeros((n_c, n_points))
        mean_recall = np.linspace(0, 1, n_points)
        average_precisions = []

        for C_idx, C in enumerate(self.c_values):
            mean_precision = np.zeros(n_points)
            std_precision = np.zeros(n_points)
            mean_ap = 0.0

            for fold in tqdm(range(self.n_folds), desc=f"Cross Validation for C={C:.2e}"):
                # Define validation images for this fold
                fold_start = fold * (n_train_images // self.n_folds)
                fold_end = (fold + 1) * (n_train_images // self.n_folds)
                # Split images w.r.t. image index
                unique_images = np.unique(image_index.numpy())
                val_images = unique_images[fold_start:fold_end]

                train_selection = ~torch.isin(image_index, torch.tensor(val_images))
                fold_x = xs[train_selection].numpy()
                fold_y = (ys[train_selection] > 0).long().numpy()
                val_x = xs[~train_selection].numpy()
                val_y = (ys[~train_selection] > 0).long().numpy()

                # Train logistic regression
                clf = LogisticRegression(random_state=self.random_state, C=C, max_iter=self.max_iter, n_jobs=1).fit(
                    fold_x, fold_y
                )

                # Predict on validation set
                output = clf.predict_proba(val_x)
                precision, recall, _ = precision_recall_curve(val_y, output[:, 1])
                ap = average_precision_score(val_y, output[:, 1])
                mean_ap += ap

                # Interpolate precision to the mean_recall points
                interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                mean_precision += interp_precision
                std_precision += interp_precision**2
                # Cleanup to avoid script getting "killed" due to memory issues
                del clf, fold_x, fold_y, val_x, val_y, output, precision, recall, interp_precision
                gc.collect()

            mean_precision_all[C_idx] = mean_precision / self.n_folds
            std_precision_all[C_idx] = np.sqrt(std_precision / self.n_folds - mean_precision_all[C_idx] ** 2)
            # Ensure non-negative std (numerical precision issues)
            std_precision_all[C_idx] = np.maximum(std_precision_all[C_idx], 0)
            average_precisions.append(mean_ap / self.n_folds)

        return {
            "c_values": self.c_values,
            "mean_precision_all": mean_precision_all,
            "std_precision_all": std_precision_all,
            "mean_recall": mean_recall,
            "average_precisions": average_precisions,
        }

    def train_final_model(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        c: float = 0.001,
        verbose: int = 0,
    ) -> LogisticRegression:
        """Train the final model on all data.

        Args:
            xs: Feature matrix of shape (n_samples, n_features)
            ys: Label vector of shape (n_samples,)
            c: Regularization parameter C for logistic regression
            verbose: Verbosity level for logistic regression

        Returns:
            Trained LogisticRegression model
        """
        clf = LogisticRegression(
            random_state=self.random_state,
            C=c,
            max_iter=self.max_iter,
            verbose=verbose,
        ).fit(xs.numpy(), (ys > 0).long().numpy())

        return clf

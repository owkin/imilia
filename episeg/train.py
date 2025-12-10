#!/usr/bin/env python3
"""Training script for IBDColEPI epithelium segmentation."""

# python /home/sagemaker-user/episeg/scripts/train.py --image-dir /home/sagemaker-user/custom-file-systems/efs/fs-09913c1f7db79b6fd/PROJECT_IBDCOLEPI/patch-dataset-HE/Trainset/Images_tif --label-dir /home/sagemaker-user/custom-file-systems/efs/fs-09913c1f7db79b6fd/PROJECT_IBDCOLEPI/patch-dataset-HE/Trainset/Labels_tif --output-dir /home/sagemaker-user/custom-file-systems/efs/fs-09913c1f7db79b6fd/histo/epithelium_segmentation/epi_seg_models --n-images-per-wsi 2 --plot  # noqa E501

import argparse
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from imilia.data.loaders import IBDColEPIDataLoader
from imilia.data.paths import TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR
from imilia.data.utils import extract_features_and_labels
from imilia.engine.episeg_trainer import EpiSegTrainer
from imilia.models import PATCH_SIZE, H0miniModelWrapper


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train IBDColEPI epithelium segmentation model")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(TRAIN_IMAGE_DIR),
        help="Directory containing training images",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=str(TRAIN_LABEL_DIR),
        help="Directory containing training labels",
    )
    parser.add_argument(
        "--n-train-images",
        type=int,
        default=100,
        help="Number of training images to use",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Step size for sampling images",
    )
    parser.add_argument(
        "--n-images-per-wsi",
        type=int,
        default=None,
        help=(
            "Number of images to sample per WSI"
            "(defaults to None, in which case '--n-train-images' and '--step' are used)"
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1022,
        help="Image size for processing",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display precision-recall plot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/episeg/",
        help="Directory to save model and plots",
    )
    return parser.parse_args()


def filter_patches(
    xs: torch.Tensor, ys: torch.Tensor, image_index: torch.Tensor, threshold: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter patches to keep only clear positive or negative labels.

    Args:
        xs: Feature matrix
        ys: Label vector
        image_index: Image index vector
        threshold: Threshold for filtering (keep patches with label < threshold or > 1-threshold)

    Returns:
        Filtered (xs, ys, image_index) tensors
    """
    idx = (ys < threshold) | (ys > 1.0 - threshold)
    xs_filtered = xs[idx]
    ys_filtered = ys[idx]
    image_index_filtered = image_index[idx]

    return xs_filtered, ys_filtered, image_index_filtered


def plot_precision_recall_curves(results: dict, output_path: Path | None = None) -> None:
    """Plot precision-recall curves from cross-validation results.

    Args:
        results: Dictionary from cross_validate() containing precision/recall data
        output_path: Optional path to save the plot
    """
    mean_precision_all = results["mean_precision_all"]
    std_precision_all = results["std_precision_all"]
    mean_recall = results["mean_recall"]
    average_precisions = results["average_precisions"]
    c_values = results.get("c_values", [])

    plt.figure(figsize=(10, 8))
    for C_idx, (C, ap) in enumerate(zip(c_values, average_precisions, strict=False)):
        plt.plot(
            mean_recall,
            mean_precision_all[C_idx],
            label=f"C={C:.2e}, AP={ap:.3f}",
        )
        plt.fill_between(
            mean_recall,
            mean_precision_all[C_idx] - std_precision_all[C_idx],
            mean_precision_all[C_idx] + std_precision_all[C_idx],
            alpha=0.2,
        )

    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.title("Mean Precision-Recall Curve across folds")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved precision-recall plot to {output_path}")
    else:
        plt.show()


def get_best_c_and_ap(results: dict) -> tuple[float, float]:
    """Get the best C parameter and corresponding average precision from results.
    If multiple C values have very similar average precision, return the smallest C.
    Similar is defined as within 1e-3 of the best average precision.

    Args:
        results: Dictionary from cross_validate() containing average precisions
    Returns:
        Tuple of (best_c_param, best_ap)
    """
    average_precisions = results["average_precisions"]
    best_ap = max(average_precisions)
    # Find all C values with AP within 1e-3 of best_ap
    similar_c_indices = [i for i, ap in enumerate(average_precisions) if abs(ap - best_ap) < 1e-3]
    # Select the smallest C among them
    best_c_index = min(similar_c_indices, key=lambda i: results["c_values"][i])
    best_c_param = results["c_values"][best_c_index]
    return best_c_param, best_ap


def main() -> None:
    """Main training script."""
    args = parse_args()

    # Setup device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir = output_dir / curr_time
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load feature extraction model
    logger.info("Loading feature extraction model...")
    h0mini_model = H0miniModelWrapper(device)

    # Load dataset
    dataset = IBDColEPIDataLoader(
        args.image_dir,
        args.label_dir,
        image_size=args.image_size,
    )
    logger.info(f"Dataset loaded with size: {len(dataset)} images")

    # Extract features and labels
    xs, ys, image_index = extract_features_and_labels(
        h0mini_model,
        dataset,
        n_images=args.n_train_images,
        step=args.step,
        device=device,
        patch_size=PATCH_SIZE,
        n_images_per_wsi=args.n_images_per_wsi,
    )
    if args.n_images_per_wsi is not None:
        # Update n_train_images based on unique images used
        args.n_train_images = len(np.unique(image_index.numpy()))
        args.step = None  # Not used anymore

    print(f"Design matrix shape: {xs.shape}")
    print(f"Label matrix shape: {ys.shape}")

    # Filter patches
    print("Filtering patches...")
    xs, ys, image_index = filter_patches(xs, ys, image_index)
    print(f"After filtering - Design matrix shape: {xs.shape}")
    print(f"After filtering - Label matrix shape: {ys.shape}")

    # Initialize trainer
    trainer = EpiSegTrainer(
        n_folds=args.n_folds,
        c_values=np.logspace(-2, 1, 4),
        random_state=0,
        max_iter=500,
    )

    # Run cross-validation
    print("Running cross-validation...")
    results = trainer.cross_validate(xs, ys, image_index, args.n_train_images)

    # Print results
    print("\n" + "=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)
    for C, ap in zip(results["c_values"], results["average_precisions"], strict=False):
        print(f"C = {C:.2e}: Average Precision = {ap:.4f}")
    print("=" * 60)

    # Plot results
    if args.plot:
        plot_path = output_dir / "precision_recall_curves.png"
        plot_precision_recall_curves(results, output_path=plot_path)

    # Train final model
    best_c_param, best_ap = get_best_c_and_ap(results)
    print(f"Training final model with C = {best_c_param:.2e}... (average precision = {best_ap:.4f})")
    final_model = trainer.train_final_model(xs, ys, c=best_c_param, verbose=0)

    # Save final model
    model_path = output_dir / "model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved final model to {model_path}")

    # Save results summary
    results_path = output_dir / "results.txt"
    with open(results_path, "w") as f:
        f.write("Training Configuration:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("Cross-Validation Results\n")
        f.write("=" * 60 + "\n")
        for C, ap in zip(results["c_values"], results["average_precisions"], strict=False):
            f.write(f"C = {C:.2e}: Average Precision = {ap:.4f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nFinal model trained with C = {best_c_param} (average precision: {best_ap:.4f})\n")
        f.write(f"Model saved to: {model_path}\n")
    print(f"Saved results summary to {results_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

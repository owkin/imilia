"""Inference script for epithelium segmentation and performance computation.
Intended for the IBDColEPI test dataset (pre-extracted patches).
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from imilia.data.utils import extract_features_and_labels
from imilia.data.loaders import IBDColEPIDataLoader
from imilia.models import PATCH_SIZE, H0miniModelWrapper
from imilia.data.paths import TEST_IMAGE_DIR, TEST_LABEL_DIR


plt.rcParams.update({"font.family": "sans-serif"})

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on IBDColEPI dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained logistic regression model",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        help="Directory containing test labels (optional, for visualization). Currently not used in inference.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1022,
        help="Image size for processing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Indices of images to process (default: all)",
    )
    args = parser.parse_args()
    return args


def get_epithelium_mask_from_predictions(
    predictions: np.ndarray,
    image_shape: tuple[int, int] | torch.Size,
    patch_size: int = PATCH_SIZE,
    threshold: float = 0.75,
) -> np.ndarray:
    """Get epithelium mask from raw predictions."""
    if isinstance(image_shape, torch.Size):
        image_shape = tuple(image_shape)
    h_patches, w_patches = [int(d / patch_size) for d in image_shape]
    epi_score = predictions.reshape(h_patches, w_patches)
    epi_score_binary = (epi_score > threshold).astype(np.uint8) * 255
    epi_score_image = Image.fromarray(epi_score_binary)
    epi_score_resized = epi_score_image.resize(image_shape, Image.NEAREST)
    return np.array(epi_score_resized)


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray, ap_score: float, output_dir: Path) -> None:
    """Plot precision-recall curve."""
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend([f"Average Precision: {ap_score:.4f}"], loc="lower left", fontsize=12)
    plt.savefig(output_dir / "precision_recall_curve.png")
    plt.close()


def run_inference(
    model_path: str | Path | LogisticRegression,
    dataset: IBDColEPIDataLoader,
    indices: list[int] | None = None,
    device: torch.device | None = None,
    output_dir: Path | None = None,
    mask_name: str | None = None,
    overlay_name: str | None = None,
) -> None:
    """Run inference on test images.

    Args:
        model_path: Path to saved logistic regression model
        dataset: Dataset to run inference on
        indices: List of indices to process. If None, processes all images
        device: Device to run inference on. If None, uses CUDA if available
        output_dir: Directory to save predictions. If None, doesn't save
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    feature_model = H0miniModelWrapper(device)
    if isinstance(model_path, (str, Path)):
        clf = joblib.load(model_path)
    else:
        raise ValueError("model_path must be a string or Path to the saved model.")

    if indices is None:
        # Process all images
        indices = list(range(len(dataset)))

    x, y, image_indices = extract_features_and_labels(
        feature_model,
        dataset,
        n_images=len(indices),
        step=1,
        device=device,
        patch_size=PATCH_SIZE,
        n_images_per_wsi=None,
    )
    x = x.numpy()
    y = (y > 0.5).long().numpy()
    image_indices = image_indices.numpy()

    # Predict on test set
    predictions = clf.predict_proba(x)
    precision, recall, _ = precision_recall_curve(y, predictions[:, 1])
    ap = average_precision_score(y, predictions[:, 1])

    logger.info(f"Average Precision: {ap:.4f}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot and save precision-recall curve
        plot_precision_recall_curve(precision, recall, ap_score=ap, output_dir=output_dir)

        # Visualization
        for idx in indices:
            test_image, test_label = dataset[idx]
            epithelium_mask = get_epithelium_mask_from_predictions(
                predictions[image_indices == idx, 1], test_image.shape[1:]
            )

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                if mask_name:
                    output_path = output_dir / mask_name
                else:
                    output_path = output_dir / f"mask_{idx:04d}.png"
                if overlay_name:
                    composite_path = output_dir / overlay_name
                else:
                    composite_path = output_dir / f"overlay_{idx:04d}.png"
                Image.fromarray(epithelium_mask).save(output_path)
                # plot mask overlaid on original image and save it
                original_image = TF.to_pil_image(test_image).convert("RGBA")
                label_image = TF.to_pil_image(test_label)
                label_mask = np.array(label_image)
                # create overlay as a green mask with alpha channel
                overlay = Image.new("RGBA", original_image.size, (0, 255, 0, 0))  # Transparent
                overlay_array_mask = np.array(overlay)
                overlay_array_gt = np.array(overlay)
                overlay_array_mask[epithelium_mask > 0] = [255, 0, 0, 100]  # Semi-transparent red
                overlay_array_gt[label_mask > 0] = [0, 255, 0, 100]  # Semi-transparent green
                overlay_mask = Image.fromarray(overlay_array_mask)
                overlay_gt = Image.fromarray(overlay_array_gt)
                composite_mask = Image.alpha_composite(original_image, overlay_mask)
                composite_gt = Image.alpha_composite(original_image, overlay_gt)
                composite_mask.save(composite_path)
                composite_gt.save(str(composite_path).replace("overlay_", "overlay_gt_"))


def main() -> None:
    """Main entry point for inference script."""
    
    args = parse_args()

    if not args.image_dir:
        logger.warning("No image directory provided. Using default test image directory.")
        args.image_dir = TEST_IMAGE_DIR
        args.label_dir = TEST_LABEL_DIR

    # Create dataset
    print("Loading dataset...")
    dataset = IBDColEPIDataLoader(
        args.image_dir,
        args.label_dir if args.label_dir else None,
        image_size=args.image_size,
    )
    print(f"Dataset size: {len(dataset)} images")

    # Run inference
    run_inference(
        args.model_path,
        dataset,
        indices=args.indices,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()

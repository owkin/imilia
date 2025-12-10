"""Load features."""

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from imilia.data.loaders import IBDColEPIDataLoader
from imilia.models import PATCH_SIZE, patch_quantization_model


def extract_features_and_labels(
    model: nn.Module,
    dataset: IBDColEPIDataLoader,
    n_images: int,
    step: int,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    n_images_per_wsi: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features and prepare labels for training.

    Args:
        model: Feature extraction model
        dataset: Dataset to extract features from
        n_images: Number of images to use
        step: Step size for sampling images
        device: Device to run inference on
        patch_size: Size of patches for label quantization
        n_images_per_wsi: Number of images to sample per WSI (if None, use n_images and step)
    Returns:
        Tuple of (features, labels, image_indices) tensors
    """

    if n_images_per_wsi is not None:
        print(f"Sampling {n_images_per_wsi} images per WSI... Ignoring 'n_images' and 'step'.")
        wsi_ids = [img_name.split("_HE")[0] for img_name in dataset.image_files]
        unique_wsi_ids = np.unique(wsi_ids)
        indices: list[int] = []
        for wsi_id in unique_wsi_ids:
            img_indices = [i for i, name in enumerate(dataset.image_files) if wsi_id + "_HE" in name]
            # Randomly sample n_images_per_wsi indices (or all if fewer available)
            selected_indices = np.random.choice(
                img_indices, size=min(n_images_per_wsi, len(img_indices)), replace=False
            )
            indices.extend(selected_indices)
    else:
        indices = list(range(0, n_images * step, step))
    logger.info(f"Using {len(indices)} images for feature extraction.")

    patch_quant_filter = patch_quantization_model(patch_size)

    xs = []
    ys = []
    image_index = []

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            for i in tqdm(indices, desc="Extracting features"):
                # Loading the data
                image_i, mask_i = dataset[i]  # image_i (C, H, W), mask_i (1, H, W)
                mask_i_quantized = patch_quant_filter(mask_i)  # (1, 1, H/PATCH_SIZE, W/PATCH_SIZE)
                mask_i_quantized = mask_i_quantized.squeeze(0).view(-1).detach()
                ys.append(mask_i_quantized)
                image_i = image_i.unsqueeze(0)  # Add batch dimension
                feats = model(image_i)  # "to device" is handled inside the model wrapper
                dim = feats.shape[1]  # (D,)
                xs.append(feats.squeeze().view(dim, -1).permute(1, 0).detach().cpu())
                image_index.append(i * torch.ones(ys[-1].shape))

    # Concatenate all lists into torch tensors
    xs_ = torch.cat(xs)
    ys_ = torch.cat(ys)
    image_index_ = torch.cat(image_index)

    return xs_, ys_, image_index_

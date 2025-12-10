"""Data loader for IBDColEPI dataset."""

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IBDColEPIDataLoader(Dataset):
    """Dataset loader for IBDColEPI images and labels."""

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path | None,
        image_files: list[str] | None = None,
        image_size: int = 1022,
        image_size_usage: str = "resize",
    ) -> None:
        """Initialize the dataset loader.

        Args:
            image_dir: Directory containing input images
            label_dir: Directory containing label masks
            image_files: Optional list of image file names to use
            image_size: Target size for resizing images (default: 1022)
            transform_to_tensor: Whether to transform images to tensors (default: True)
        """
        assert image_size_usage in ["resize", "assert"], "image_size_usage must be 'resize' or 'assert'"
        self.image_dir = Path(image_dir)
        self.image_size_usage = image_size_usage
        if image_files is not None:
            self.image_files = image_files
        else:
            self.image_files = sorted(os.listdir(self.image_dir))

        self.image_size = image_size

        if label_dir is None:
            self.label_dir = None
            self.label_files = [None] * len(self.image_files)
        else:
            self.label_dir = Path(label_dir)
            self.label_files = sorted(os.listdir(self.label_dir))
            if len(self.image_files) != len(self.label_files):
                raise ValueError(
                    f"Mismatch between number of images ({len(self.image_files)}) "
                    f"and labels ({len(self.label_files)})"
                )
            for img_file, label_file in zip(self.image_files, self.label_files, strict=False):
                if img_file != label_file:
                    raise ValueError(f"Image file {img_file} does not match label file {label_file}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        """Get a single image and label pair.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image, label) tensors
        """
        img_path = self.image_dir / self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.image_size_usage == "resize":
            image = image.resize((self.image_size, self.image_size))
        else:
            assert (
                image.size[0] == self.image_size and image.size[1] == self.image_size
            ), f"Image size {image.size} does not match expected size {self.image_size}"

        if self.label_dir is None:
            label = None
        else:
            label_path = self.label_dir / self.label_files[idx]
            label = Image.open(label_path).convert("L")
            if self.image_size_usage == "resize":
                label = label.resize((self.image_size, self.image_size), resample=Image.NEAREST)
            else:
                assert (
                    label.size[0] == self.image_size and label.size[1] == self.image_size
                ), f"Label size {label.size} does not match expected size {self.image_size}"
            label = label.point(lambda p: 255 if p > 0 else 0)  # Binarize the label

        # If images and labels are not tensors, convert them
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        if (self.label_dir is not None) and (not isinstance(label, torch.Tensor)):
            label = transforms.ToTensor()(label)

        return image, label

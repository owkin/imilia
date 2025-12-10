"""Data module for IBDColEPI dataset."""

from imilia.data.loaders import IBDColEPIDataLoader
from imilia.data.paths import (
    BASE_DIR,
    BASE_SEG_DIR,
    FEATURES_DIR,
    IMAGE_DIR,
    LABEL_DIR,
    WSI_PATH,
)


__all__ = [
    "IBDColEPIDataLoader",
    "BASE_DIR",
    "WSI_PATH",
    "BASE_SEG_DIR",
    "IMAGE_DIR",
    "LABEL_DIR",
    "FEATURES_DIR",
]

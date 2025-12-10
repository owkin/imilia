"""Data module for IBDColEPI dataset."""

from imilia.data.loaders import IBDColEPIDataLoader
from imilia.data.paths import (
    BASE_DIR,
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    WSI_PATH,
)


__all__ = [
    "IBDColEPIDataLoader",
    "BASE_DIR",
    "WSI_PATH",
    "TRAIN_IMAGE_DIR",
    "TRAIN_LABEL_DIR",
    "TEST_IMAGE_DIR",
    "TEST_LABEL_DIR",
]

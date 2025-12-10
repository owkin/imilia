"""Model module for feature extraction."""

from imilia.models.h0mini import H0miniModelWrapper
from imilia.models.patch_quantization import patch_quantization_model


PATCH_SIZE = 14

__all__ = ["H0miniModelWrapper", "PATCH_SIZE", "patch_quantization_model"]

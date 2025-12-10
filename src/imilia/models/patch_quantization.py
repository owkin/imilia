"""Patch quantization."""

import torch.nn as nn


def patch_quantization_model(patch_size: int) -> nn.Module:
    # Quantization filter for the given patch size
    patch_quant_filter = nn.Conv2d(1, 1, kernel_size=patch_size, stride=patch_size, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))

    return patch_quant_filter

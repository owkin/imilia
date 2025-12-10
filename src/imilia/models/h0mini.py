"""Model loading utilities."""

import timm
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class H0miniModelWrapper(torch.nn.Module):
    """Wrapper for H0-mini model to extract features from intermediate layers."""

    def __init__(self, device: torch.device | None = None, dynamic_img_size: bool = True):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(dynamic_img_size=dynamic_img_size)
        self.mean, self.std = self._get_normalize_transform_params()

    def _load_model(
        self,
        dynamic_img_size: bool = True,
    ) -> torch.nn.Module:
        """Load the feature extraction model (H0mini).

        Args:
            dynamic_img_size: Whether to use dynamic image size. Default: True

        Returns:
            Loaded model in eval mode
        """
        model = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=dynamic_img_size,
        )
        model = model.to(self.device)
        model.eval()
        return model

    def _get_normalize_transform_params(self):
        """Load the normalization transformation function for H0-mini model.
        Returns:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        full_transform = create_transform(**config)
        normalize_transform = next((t for t in full_transform.transforms if isinstance(t, T.Normalize)), None)
        if normalize_transform is not None:
            mean = normalize_transform.mean
            std = normalize_transform.std
            return mean, std
        else:
            raise ValueError("Normalize transform not found in the model's preprocessing pipeline.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor after passing through the model
        """
        x_normalized = TF.normalize(x, mean=self.mean, std=self.std)
        x_normalized = x_normalized.to(self.device)
        feats = self.model.forward_features(x_normalized)[:, self.model.num_prefix_tokens :, :].permute(0, 2, 1)
        return feats

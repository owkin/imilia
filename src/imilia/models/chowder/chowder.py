import torch
import warnings
from typing import Optional, List
import numpy as np
from .submodules import TilesMLP, ExtremeLayer, MLP


class Chowder(torch.nn.Module):
    """
    Chowder module.

    Example:
        >>> module = Chowder(in_features=128, out_features=1, n_top=5, n_bottom=5)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
    out_features: int
        controls the number of scores and, by extension, the number of out_features
    n_top: int
    n_bottom: int
    tiles_mlp_hidden: Optional[List[int]] = None
    mlp_hidden: Optional[List[int]] = None
    mlp_dropout: Optional[List[float]] = None
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_extreme: Optional[int] = None,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super(Chowder, self).__init__()

        if n_extreme is not None:
            warnings.warn(
                DeprecationWarning(
                    f"Use `n_extreme=None, n_top={n_extreme if n_top is None else n_top}, "
                    f"n_bottom={n_extreme if n_bottom is None else n_bottom}` instead."
                )
            )

            if n_top is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_top={n_top}`" f"with `n_top=n_extreme={n_extreme}`."
                    )
                )
            if n_bottom is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_bottom={n_bottom}`"
                        f"with `n_bottom=n_extreme={n_extreme}`."
                    )
                )

            n_top = n_extreme
            n_bottom = n_extreme

        if n_top is None and n_bottom is None:
            raise ValueError("At least one of `n_top` or `n_bottom` must not be None.")

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.score_model = TilesMLP(
            in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=out_features
        )
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom)

        mlp_in_features = n_top + n_bottom
        self.mlp = MLP(
            mlp_in_features,
            1,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.mlp.apply(self.weight_initialization)

    @classmethod
    def classification(cls, in_features: int, out_features: Optional[int] = 1) -> "Chowder":
        """
        Chowder operator using Chowder paper parameters as default.

        Parameters
        ----------
        in_features: int
        out_features: Optional[int] = 1

        Returns
        -------
        chowder: Chowder
        """
        return cls(
            in_features=in_features,
            out_features=out_features,
            n_extreme=None,
            n_top=5,
            n_bottom=5,
            tiles_mlp_hidden=None,
            mlp_hidden=[200, 100],
            mlp_dropout=None,
            mlp_activation=torch.nn.Sigmoid(),
            bias=False,
        )

    @classmethod
    def galaxy_classification(
        cls, in_features: int, out_features: Optional[int] = 1
    ) -> "Chowder":
        """
        Chowder operator using Galaxy's ChowderClassification parameters as default.

        Parameters
        ----------
        in_features: int
        out_features: Optional[int] = 1

        Returns
        -------
        chowder: Chowder
        """
        return cls(
            in_features=in_features,
            out_features=out_features,
            n_extreme=None,
            n_top=10,
            n_bottom=10,
            tiles_mlp_hidden=None,
            mlp_hidden=[128, 64],
            mlp_dropout=[0.3, 0.3],
            mlp_activation=torch.nn.ReLU(),
            bias=True,
        )

    @classmethod
    def galaxy_survival(cls, in_features: int, out_features: Optional[int] = 1) -> "Chowder":
        """
        Chowder operator using Galaxy's ChowderSurvival parameters as default.

        Parameters
        ----------
        in_features: int
        out_features: Optional[int] = 1

        Returns
        -------
        chowder: Chowder
        """
        return cls(
            in_features=in_features,
            out_features=out_features,
            n_extreme=None,
            n_top=25,
            n_bottom=25,
            tiles_mlp_hidden=None,
            mlp_hidden=[128],
            mlp_dropout=None,
            mlp_activation=None,
            bias=True,
        )

    @classmethod
    def chowder_v2(cls, in_features: int, out_features: Optional[int] = 1) -> "Chowder":
        """
        Chowder operator using Galaxy's ChowderV2 parameters as default.

        Parameters
        ----------
        in_features: int
        out_features: Optional[int] = 1

        Returns
        -------
        chowder: Chowder
        """
        return cls(
            in_features=in_features,
            out_features=out_features,
            n_extreme=None,
            n_top=10,
            n_bottom=10,
            tiles_mlp_hidden=[128],
            mlp_hidden=[128, 64],
            mlp_dropout=None,
            mlp_activation=torch.nn.Sigmoid(),
            bias=True,
        )

    @staticmethod
    def weight_initialization(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def check_polarity(
        self,
        x: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        channel: int = 0,
        output: int = 0,
    ):
        """
        Check the polarity of the contribution of a given channel to the predictions.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.
        channel: int
            index of the channel
        output: int
            index of the output

        Returns
        -------
        polarity, pvalue: Tuple[int, float]
        """
        assert x.shape[0] > 1, "At least 2 samples are needed to assess polarity."

        with torch.no_grad():
            pred, scores = self.forward(x, mask)

        tensor = torch.stack(
            [pred.cpu()[:, output], torch.mean(scores.cpu()[:, :, channel], dim=1)]
        )
        r = torch.corrcoef(tensor)[0, 1].numpy()

        if x.shape[0] == 2:
            pvalue = 1
        else:
            ab = x.shape[0] / 2 - 1
            dist = torch.distributions.beta.Beta(ab, ab)
            x = torch.linspace(-1, np.abs(r), 100)
            cdf = torch.trapz(dist.log_prob((x + 1) / 2).exp(), x) / 2
            cdf = min(cdf.numpy(), 1)
            pvalue = 2 * (1 - cdf)

        if pvalue > 0.05:
            warnings.warn("Sample size is too small to assess polarity.")
        polarity = np.sign(r)

        return polarity, pvalue

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        """
        scores = self.score_model(x=x, mask=mask)
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        # Apply MLP to the N_TOP + N_BOTTOM scores
        y = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)

        return y.squeeze(2), extreme_scores


class MultiChannelChowder(Chowder):
    """
    Chowder with multiple channels.
    Example:
        >>> module = MultiChannelChowder(in_features=128, out_features=1, n_channels=5, n_top=5, n_bottom=5)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)
    Parameters
    ----------
    in_features: int
    out_features: int
        controls the number of out_features
    n_channels: int
        controls the number of scores, if 1 this module is equivalent to Chowder
    n_top: int
    n_bottom: int
    tiles_mlp_hidden: Optional[List[int]] = None
    mlp_hidden: Optional[List[int]] = None
    mlp_dropout: Optional[List[float]] = None
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_channels: int = 5,
        n_extreme: Optional[int] = None,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super(Chowder, self).__init__()

        if n_extreme is not None:
            warnings.warn(
                DeprecationWarning(
                    f"Use `n_extreme=None, n_top={n_extreme if n_top is None else n_top}, "
                    f"n_bottom={n_extreme if n_bottom is None else n_bottom}` instead."
                )
            )

            if n_top is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_top={n_top}`" f"with `n_top=n_extreme={n_extreme}`."
                    )
                )
            if n_bottom is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_bottom={n_bottom}`"
                        f"with `n_bottom=n_extreme={n_extreme}`."
                    )
                )

            n_top = n_extreme
            n_bottom = n_extreme

        if n_top is None and n_bottom is None:
            raise ValueError("At least one of `n_top` or `n_bottom` must not be None.")

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.score_model = TilesMLP(
            in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=n_channels
        )
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom)

        mlp_in_features = (n_top + n_bottom) * n_channels
        self.mlp = MLP(
            mlp_in_features,
            out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.mlp.apply(self.weight_initialization)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(x=x, mask=mask)
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, N_CHANNELS)

        # Apply MLP to the (N_TOP + N_BOTTOM) * N_CHANNELS scores
        y = self.mlp(
            extreme_scores.reshape(-1, extreme_scores.shape[1] * extreme_scores.shape[2])
        )  # (B, OUT_FEATURES)

        return y, extreme_scores
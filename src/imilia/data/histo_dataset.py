"""Module to load and process histology features for machine learning tasks."""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

# TODO: refacto
class HistoDataset(Dataset):
    """
    PyTorch Dataset for histology data, including feature loading and padding.

    Parameters
    ----------
    feat_paths : list of str
        Paths to feature files (.npy format) for each sample.
    labels : list of int or float
        List of labels corresponding to each sample.
    max_tiles : int, optional
        Maximum number of tiles to include per sample. If None, the max
        tile count across samples is used.

    Attributes
    ----------
    feat_paths : list of str
        Paths to feature files (.npy format) for each sample.
    labels : list of int or float
        List of labels corresponding to each sample.
    max_tiles : int
        Maximum number of tiles per sample, based on `max_tiles` or data.
    n_feats : int
        Number of features per tile, excluding spatial coordinates.
    """

    def __init__(self, feat_paths, labels, max_tiles=None, slide_level=False, include_coords_in_feats=False):
        # Initialize paths and labels, ensure both are of equal length
        self.feat_paths = [Path(path) for path in feat_paths]
        assert labels.ndim <= 2, "Ammount of dimensions larger than max expected (2)."
        self.labels = labels.reshape(-1, 1)
        # self.enc_labels = self._one_hot_encode_labels()
        self.slide_level = slide_level
        self.include_coords_in_feats = include_coords_in_feats

        # # If the labels are of type bool, convert them to bool_ to allow the collate_fn to work
        # if all(isinstance(label[0], bool) for label in self.labels):
        #     self.labels = np.bool_(self.labels)
        assert len(self.feat_paths) == len(self.labels), "Features and labels count mismatch."

        # Set max_tiles to the largest feature count across samples if not specified (to be avoided as it is slow)
        if max_tiles is None:
            max_tiles = max(self._load_from_path(path).shape[0] for path in self.feat_paths)
        self.max_tiles = max_tiles

        # Get feature count per tile (excluding coordinates), setting `n_feats`
        sample_feats = self._load_from_path(self.feat_paths[0])
        self.n_feats = sample_feats.shape[1]

    def _one_hot_encode_labels(self):
        """One-hot encode the labels."""
        enc = OneHotEncoder()
        enc.fit(self.labels)
        enc_labels = enc.transform(self.labels).toarray()
        return enc_labels

    def _load_from_path(self, path):
        """
        Load features from a given path, excluding spatial coordinates.

        Parameters
        ----------
        path : str or Path
            Path to the features file.

        Returns
        -------
        feats : np.ndarray
            Loaded features, excluding spatial coordinates.
        """
        # Features can be .npy or .csv
        if str(path).endswith(".npy"):
            feats = np.load(path)
            if not self.include_coords_in_feats:
                feats = np.load(path)[:, 3:]
        else:
            raise ValueError("Features are expected to be stored in numpy arrays (.npy).")
        return feats

    def __len__(self):
        return len(self.feat_paths)

    def __getitem__(self, idx):
        # Load features, excluding spatial coordinates
        feats = self._load_from_path(self.feat_paths[idx])
        assert feats.shape[1] == self.n_feats, "Feature shape mismatch."

        if self.slide_level:
            # Aggregate features at slide level by averaging across tiles
            mask = np.sum(feats**2, axis=-1) == 0  # Identify zero-padded tiles
            feats = feats[~mask]  # Remove zero-padded tiles
            feats = np.mean(feats, axis=0, keepdims=True)
        else:
            # At tile-level, we want to enforce the same number of tiles for all samples
            # If there are less tiles than `max_tiles`, apply zero padding
            if feats.shape[0] < self.max_tiles:
                feats_padded = np.zeros((self.max_tiles, self.n_feats))
                feats_padded[: feats.shape[0], :] = feats
                feats = feats_padded
            else:
                feats = feats[: self.max_tiles, :]

        # return feats, self.enc_labels[idx]
        return feats, self.labels[idx]

    def set_slide_level(self, slide_level: bool):
        """
        Set whether the dataset is slide-level or tile-level.
        This allows to change the behavior of the dataset dynamically (after it has already been initialized).

        Parameters
        ----------
        slide_level : bool
            If True, the dataset will treat return aggregated slide-level features for each sample.
        """
        self.slide_level = slide_level

    def get_n_tiles_per_slide(self):
        """
        Get the number of tiles for each slide in the dataset.

        Returns
        -------
        n_tiles_list : list of int
            List containing the number of tiles for each slide.
        """
        n_tiles_list = []
        for path in self.feat_paths:
            feats = self._load_from_path(path)
            n_tiles = feats.shape[0]
            n_tiles_list.append(n_tiles)
        return n_tiles_list

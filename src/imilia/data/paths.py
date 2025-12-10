"""Path configuration for IBDColEPI dataset."""

import os
from pathlib import Path

import git


git_repo = git.Repo(__file__, search_parent_directories=True)
git_root = os.path.abspath(git_repo.git.rev_parse("--show-toplevel"))

####################################################################
# Base directory for the IBDColEPI dataset
# Edit this path according to your local setup
BASE_DIR = Path("/home/sagemaker-user/custom-file-systems/efs/fs-09913c1f7db79b6fd/PROJECT_IBDCOLEPI")
####################################################################

# WSI (Whole Slide Image) directory
WSI_PATH = BASE_DIR / "WSI"

# Segmentation dataset directory
BASE_SEG_DIR = BASE_DIR / "patch-dataset-HE" / "Trainset"

# Image and label directories
IMAGE_DIR = BASE_SEG_DIR / "Images_tif"
LABEL_DIR = BASE_SEG_DIR / "Labels_tif"

# Features directory (optional, for cached features)
FEATURES_DIR = BASE_DIR / "features_H0_mini"

# Dataframes with min/max tiles coordinates and scores
# TODO: replace "slide_path" in dfs by "slide_name"
MIN_MAX_TILES_DIR = Path(git_root) / "data" / "IBDColEpi" / "min_max_tiles"

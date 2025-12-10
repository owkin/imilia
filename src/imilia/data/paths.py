"""Path configuration for IBDColEPI dataset."""

from pathlib import Path
import git
import os

git_repo = git.Repo(__file__, search_parent_directories=True)
git_root = os.path.abspath(git_repo.git.rev_parse("--show-toplevel"))

####################################################################
# Base directory for the IBDColEPI dataset
# Edit this path according to your local setup
BASE_DIR = Path("/home/sagemaker-user/custom-file-systems/efs/fs-09913c1f7db79b6fd/PROJECT_IBDCOLEPI")
####################################################################

# WSI (Whole Slide Image) directory
WSI_PATH = BASE_DIR / "WSI"

# IBDColEpi: image and label directories for train and test sets
PATCH_DATASET_BASE_DIR = BASE_DIR / "patch-dataset-HE"
TRAIN_IMAGE_DIR = PATCH_DATASET_BASE_DIR / "Trainset" / "Images_tif"
TRAIN_LABEL_DIR = PATCH_DATASET_BASE_DIR / "Trainset" / "Labels_tif"
TEST_IMAGE_DIR = PATCH_DATASET_BASE_DIR / "Testset" / "Images_tif"
TEST_LABEL_DIR = PATCH_DATASET_BASE_DIR / "Testset" / "Labels_tif"

# Tiling coordinates
TILING_COORDS_DIR = Path(git_root) / "ibdcolepi_tiling_coords"

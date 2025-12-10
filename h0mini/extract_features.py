"""Script to extract features from IBDColEpi slides using H0-Mini model."""

import openslide
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import Dataset, DataLoader
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
import numpy as np
from pathlib import Path
import argparse
import logging
import os
import shutil
from tqdm import tqdm

from imilia.data.paths import TILING_COORDS_DIR
from imilia.data.loaders import IBDColEpiHistoLoader


H0MINI_PATCH_SIZE = 14  # H0-Mini patch size is 14x14
logging.basicConfig(level=logging.INFO, format='[%(processName)s] %(message)s')


class TileDataset(Dataset):
    """Loader for tiles from a whole slide image."""

    def __init__(self, slide: openslide.OpenSlide, tiles_coords: np.ndarray, tile_size: int = 224, transform=None):
        """
        Args:
            slide (openslide.OpenSlide): Whole slide image.
            tiles_coords (np.ndarray): Array of shape (n_tiles, 3) with (level, x, y) coordinates of tiles.
        """
        self.slide = slide
        self.tiles_coords = tiles_coords
        self.deepzoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.tiles_coords)

    def __getitem__(self, idx: int) -> np.ndarray:
        tile_level, tile_x, tile_y = self.tiles_coords[idx]
        coords = self.deepzoom.get_tile_coordinates(
            level=int(tile_level), address=(tile_x, tile_y)
        )
        tile = self.slide.read_region(location=coords[0], level=coords[1], size=coords[2]).convert("RGB")
        # Resize tile to (tile_size, tile_size)
        tile = tile.resize((self.tile_size, self.tile_size))
        if self.transform is None:
            tile_tensor = transforms.ToTensor()(tile)
            logging.warning("No transform provided, using ToTensor only.")
        else:
            tile_tensor = self.transform(tile)
        return tile_tensor, np.array([tile_level, tile_x, tile_y])

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features using H0-Mini model.")
    parser.add_argument("--save_dir", type=str, default="./h0mini_feats", help="Directory to save H0-mini features.")
    parser.add_argument("--tile_size", type=int, default=224, help="Tile size.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument("--overwrite", type=bool, default=False, help="Whether to overwrite existing features.")
    args = parser.parse_args()
    return args

def process_slide(slide_path, coords_path, h0mini_model, transform, args):
    try:
        slide_name = coords_path.parent.name
        save_path = Path(args.save_dir) / slide_name / "features.npy"
        if save_path.exists() and not args.overwrite:
            logging.info(f"Features for slide {slide_name} already exist at {save_path}, skipping. If you want to overwrite, set overwrite=True.")
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            slide = openslide.OpenSlide(slide_path)

            tiles_coords = np.load(coords_path)  # shape: (n_tiles, 3) with (level, x, y)
            tile_dataset = TileDataset(
                slide, 
                tiles_coords, 
                tile_size=args.tile_size, 
                transform=transform,
            )
            tile_loader = DataLoader(
                tile_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )

            h0mini_feats = []
            for batch_tiles, batch_coords in tile_loader:
                # batch_tiles: (batch_size, n_channels, dim_x, dim_y)
                # We recommend using mixed precision for faster inference
                with torch.autocast(device_type=args.device, dtype=torch.float16):
                    with torch.inference_mode():
                        output = h0mini_model((batch_tiles).to(args.device))  # (batch_tiles.shape[0], 261, 768)
                        # CLS token features (batch_tiles.shape[0], 768)
                        cls_features = output[:, 0]

                assert cls_features.shape == (batch_tiles.shape[0], 768)
                # Concatenate cls_features to batch_coords (batch_tiles.shape[0], 3+768)
                tiles_feats = np.concatenate(
                    [batch_coords.numpy(), cls_features.cpu().numpy()], axis=1
                )
                h0mini_feats.append(tiles_feats)
            h0mini_feats = np.vstack(h0mini_feats)  # shape: (n_tiles, 3+768)
            assert h0mini_feats.shape == (tiles_coords.shape[0], 3+768)
            np.save(save_path, h0mini_feats)
            logging.info(f"Features for slide {slide_name} saved at {save_path}")
            slide.close()
    except Exception as e:
        logging.error(f"Error processing slide {slide_path}: {e}")
        raise e

def main(args):
    if torch.cuda.is_available():
        args.device = "cuda"
        args.pin_memory = True
    else:
        args.device = "cpu"
        args.pin_memory = False

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    coords_paths = list(TILING_COORDS_DIR.glob("*_HE_*/tiles_coords.npy"))
    slides_paths_ = IBDColEpiHistoLoader().get_slides_paths()
    slides_paths = [slides_paths_[path.parent.name.split(".")[0]] for path in coords_paths]
    logging.info(f"Found {len(slides_paths)} slides and corresponding tiles coords")

    # TODO: remove this line to run on full dataset
    coords_paths = coords_paths[:5]
    slides_paths = slides_paths[:5]
    ################################################

    h0mini_model = timm.create_model(
        "hf-hub:bioptimus/H0-mini",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    h0mini_model.to(args.device)
    h0mini_model.eval()
    transform = create_transform(**resolve_data_config(h0mini_model.pretrained_cfg, model=h0mini_model))

    for slide_path, coords_path in tqdm(zip(slides_paths, coords_paths)):
        process_slide(slide_path, coords_path, h0mini_model, transform, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)

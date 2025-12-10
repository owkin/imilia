"""Chowder Inference Script"""

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import os

from imilia.models.chowder import get_cv_models

from imilia.data.histo_dataset import HistoDataset
from imilia.data.loaders import load_data
from imilia.data.constants import MAX_TILES_IBDCOLEPI as MAX_TILES


def parse_args():
    parser = argparse.ArgumentParser(description="Chowder Inference Script")
    parser.add_argument(
        "--feats_dir",
        type=str,
        required=True,
        help="Path to H0-mini features directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./chowder_preds",
        help="Directory to save predictions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader. 0 means no multiprocessing.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ibdcolepi",
        help="Name of the dataset.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pin_memory = True if torch.cuda.is_available() else False

    print(f"Loading models...")
    models = get_cv_models()

    # Load data
    print("Loading data...")
    x_paths, y_labels, patient_ids = load_data(feats_dir=Path(args.feats_dir))

    histo_ds = HistoDataset(
        feat_paths = x_paths,
        labels = y_labels,
        max_tiles = MAX_TILES,
        slide_level=False,
        include_coords_in_feats=False,
    )
    loader = DataLoader(
        histo_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    # Inference
    print("Running inference...")
    preds = []
    for x, _ in tqdm(loader):
        batch_models_preds = []
        for model in models:
            model.eval()
            x = x.to(model.device)
            x = x.to(torch.float32)
            with torch.no_grad():
                logits, _ = model(x)
            batch_models_preds.append(logits)
        # Aggregate predictions from different models (e.g., average)
        batch_agg_preds = torch.mean(torch.stack(batch_models_preds), dim=0)
        preds.append(batch_agg_preds)
    preds = torch.cat(preds, dim=0).squeeze()

    df_preds = pd.DataFrame({
        "feats_path": x_paths,
        "patient_id": patient_ids,
        "label": y_labels,
        "logits": preds.cpu().numpy(),
        "probs": torch.sigmoid(preds).cpu().numpy(),
    })

    # Save predictions
    df_preds.to_csv(save_dir / f"{args.dataset_name.lower()}_preds.csv")
    print(f"Predictions saved to {save_dir / f'{args.dataset_name.lower()}_preds.csv'}")

if __name__ == "__main__":
    main()
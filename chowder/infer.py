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


save_dir = Path("./chowder_preds")
batch_size = 10  # adjust based on memory
num_workers = 0  # >0 enables multiprocessing
dataset_name = "ibdcolepi"


def main():
    save_dir.mkdir(parents=True, exist_ok=True)

    pin_memory = True if torch.cuda.is_available() else False

    print(f"Loading models...")
    models = get_cv_models()

    # Load data
    print("Loading data...")
    x_paths, y_labels, patient_ids = load_data()
    
    # TODO: remove this line to run on full dataset
    x_paths = x_paths[:5]
    y_labels = y_labels[:5]
    patient_ids = patient_ids[:5]
    ##############################################

    histo_ds = HistoDataset(
        feat_paths = x_paths,
        labels = y_labels,
        max_tiles = MAX_TILES,
        slide_level=False,
        include_coords_in_feats=False,
    )
    loader = DataLoader(
        histo_ds,
        batch_size=batch_size,
        num_workers=num_workers,
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
    df_preds.to_csv(save_dir / f"{dataset_name.lower()}_preds.csv")
    print(f"Predictions saved to {save_dir / f'{dataset_name.lower()}_preds.csv'}")

if __name__ == "__main__":
    main()
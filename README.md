<h1 align="center">IMILIA</h1>

<div align="center">
    <img src="https://img.shields.io/badge/version-0.0.1-orange.svg" />
    <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python" />
    </a>
    <a href="https://github.com/owkin/imilia/actions?workflow=ci-cd" target="_blank">
        <img src="https://github.com/owkin/imilia/workflows/ci-cd/badge.svg" />
    </a>
    <img src="assets/cov_badge.svg"/>
    <a href="https://docs.astral.sh/uv/" target="_blank">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/refs/heads/main/assets/badge/v0.json" />
    </a>
</div>

<p align="center"><em>Interpretable multiple instance learning for inflammation prediction in IBD from H&E whole slide images</em></p>

---

This repository allows to reproducuce part of the results reporterd in the IMILIA paper: 
    
- H0-mini: inference for tile-level feature extraction
- Chowder (MIL) model: inference on the IBDColEpi public dataset
- EpiSeg: training and inference on the IBDColEpi public dataset

The code provided here does not currently include the prediction with HistoPLUS on min/max tiles. We are currently working to integrate it to this repo, but in the meantime we refer the reader to the instructions available in Hugging Face to run HistoPLUS: https://huggingface.co/Owkin-Bioptimus/histoplus. 

## Setup

1. Run the following command to install `uv` and setup the `uv` environment:

    ````
    make install-all
    ````

2. Download the IBDColEpi data from: https://www.kaggle.com/datasets/henrikpe/251-he-cd3-wsis-annotated-epithelium-ibdcolepi

3. Change `BASE_DIR` in `./src/imilia/data/paths.py` to the path corresponding to where you saved (and extracted) the IBDColEpi dataset. The `WSI_PATH` assumes all WSIs have been extracted into a folder named `WSI`. 

## H0-mini feature extraction

First step is to request access to the model on Hugging Face: https://huggingface.co/bioptimus/H0-mini

We have performed the tiling of the IBDColEpi WSIs and provide the coordinates to the different tiles in the folder `ibdcolepi_tiling_coords`. It contains one numpy array per slide of shape `(n_tiles, 3)` containing the coordinates `level, x, y` for all tiles.

Once request to the model has been granted, you can then run the feature extraction with H0-mini:

```
python ./h0mini/extract_features.py
```

Feel free to adjust the script parameters to suit your computing capabilities.

The features are saved in `./outputs/h0mini_feats` by default (unless you pass a different `--save_dir`).

## Chowder inference on IBDColEpi 

Once the features are extracted, you can run the inference with the pre-trained Chowder (model weights provided in the folder `./assets/chowder_weights`):

```
python ./chowder/infer.py --feats_dir ./outputs/h0mini_feats
```

Feel free to adjust the script parameters to suit your computing capabilities.

The predictions are saved in `./outputs/chowder_preds` by default (unless you pass a different `--save_dir`).

Then you can run the following script to assess the performance of the model from the predictions:

```
python ./chowder/compute_perfs.py 
```

_Retrieve tile-level scores: (TODO: add instructions/script)_


## EpiSeg training and inference

You can run model training with:

```
python ./episeg/train.py
```

The model is saved in `./outputs/episeg/<timestamp>` by default (unless you pass a different `--output-dir`). Note that a "timestamp" is attributed to the saved model and used as a parent folder.

Run the inference with the trained model on IBDColEpi pre-extracted test patches:

```
python ./episeg/infer.py --model-path ./outputs/episeg/<timestamp>/model.joblib --output-dir ./outputs/episeg/<timestamp>/preds/
```

Replace `<timestamp>` by the timestamp generated during the model training. Feel free to change the `--output-dir` as desired.

This script will by default use the pre-extracted test patches from the IBDColEpi dataset. It will save the binary epithelium masks (`mask_<img_idx>.png`), as well as images showing the ground truth and predicted masks overlaid on the original patch (`overlay_gt_<img_idx>.png` and `overlay_<img_idx>.png`, respectively).

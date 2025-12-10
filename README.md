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

## 📦 Setup

Run the following command to install uv and setup uv environment:

````
make install-all
````

Download the IBDColEpi data from: https://www.kaggle.com/datasets/henrikpe/251-he-cd3-wsis-annotated-epithelium-ibdcolepi

Change `BASE_DIR` in `/home/sagemaker-user/imilia/src/imilia/data/paths.py` to the path corresponding to where you saved IBDColEpi dataset.

## EpiSeg

Train the model:

```
python ./episeg/train.py. --output-dir <path/to/episeg_model_dir>
```

Infer the model on IBDColEpi pre-extracted test patches:

```
python ./episeg/infer.py --model-path <path/to/episeg_model_dir/timestamp/model.joblib> --image-dir --image-size --output-dir <path/to/episeg_preds_dir>
```

This will by default use the pre-extracted test patches from the IBDColEpi dataset. It'll save the binary epithelium masks (`mask_<img_idx>.png`), as well as images showing the ground truth and predicted masks overlaid on the original patch (`overlay_gt_<img_idx>.png` and `overlay_<img_idx>.png`, respectively).

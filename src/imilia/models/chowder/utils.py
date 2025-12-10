"""Utility functions for loading Chowder models and their parameters from logs."""

import ast
import os
import re
from pathlib import Path

import git
import torch
from torch.nn import Sigmoid

from .chowder import MultiChannelChowder


git_repo = git.Repo(__file__, search_parent_directories=True)
git_root = os.path.abspath(git_repo.git.rev_parse("--show-toplevel"))
CHOWDER_WEIGHTS_DIR = Path(git_root) / "assets" / "chowder_weights"


def _get_model_params_from_logs(models_folder: Path):
    ACTIVATIONS = {
        "Sigmoid()": Sigmoid(),
    }
    logs_path = models_folder / "exp.log"
    model_params = None
    with open(logs_path, "r") as f:
        for line in f:
            line = line.strip()
            if "model params" in line.lower():
                match = re.search(r"\{.*\}", line)
                if match:
                    dict_str = match.group(0)
                    model_params = ast.literal_eval(
                        re.sub(r"Sigmoid\(\)", "'Sigmoid()'", dict_str)
                    )  # safely convert to Python dict
                    if "mlp_activation" in model_params:
                        act_str = model_params["mlp_activation"]
                        model_params["mlp_activation"] = ACTIVATIONS.get(act_str, None)
                break
    if not model_params:
        raise ValueError("Model parameters not found in logs.")
    return model_params


def _load_model_from_model_path(model_path: Path):
    model_params = _get_model_params_from_logs(model_path.parents[1])
    model = MultiChannelChowder(**model_params)
    if torch.cuda.is_available():
        model.device = torch.device("cuda")
    else:
        model.device = torch.device("cpu")
    model.to(model.device)
    _ = model.load_state_dict(torch.load(model_path, map_location=model.device))
    model.eval()
    return model


def get_cv_models(models_folder: Path = CHOWDER_WEIGHTS_DIR):
    model_paths = list((models_folder).rglob("*.pt"))
    models = [_load_model_from_model_path(m_path) for m_path in model_paths]
    print(f"Loaded {len(models)} models.")
    return models

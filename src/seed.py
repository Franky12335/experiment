"""Seed Python, numerical libraries, and CUDA from one configuration."""
import os
from pathlib import Path

def configured_seed():
    from omegaconf import OmegaConf
    return int(OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/main.yaml").seed)

def seed_everything():
    import random
    import numpy as np
    import torch
    seed = configured_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed

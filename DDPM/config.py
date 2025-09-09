# DDPM/config.py

import torch
from dataclasses import dataclass

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    # General
    seed: int = 42
    device: torch.device = get_device()

    # Data
    dataset_name: str = "cifar10"   # any HF dataset name or local path
    split: str = "train"
    image_key: str | None = None    # auto-detect if None
    image_size: int = 128
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True

    # Model
    in_channels: int = 3
    out_channels: int = 3
    time_dim: int = 256

    # Training
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    epochs: int = 50
    grad_clip: float | None = None  # e.g., 1.0 to enable, or None to disable
    mixed_precision: bool = False   # set True if you want autocast on CUDA

    # Logging / Checkpoints
    log_every: int = 100
    save_every_epochs: int = 5
    save_dir: str = "checkpoints"

# Create a default config instance you can import directly
CONFIG = Config()
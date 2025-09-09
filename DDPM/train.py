# DDPM/train.py

import torch
from model import UNET
from ddpm_utils import Diffusion, train_loop
from config import CONFIG
from data import get_data  # from the helper we created earlier
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = CONFIG

    
    device = cfg.device
    set_seed(cfg.seed)

    # Data
    loader = get_data(
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Model
    model = UNET(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        time_dim=cfg.time_dim,
        image_size=cfg.image_size,
        device=device,
    )

    # Diffusion utils
    diffusion = Diffusion(
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        image_size=cfg.image_size,
        channels=cfg.in_channels,
        device=device,
    )

    # Train
    train_loop(
        model=model,
        diffusion=diffusion,
        dataloader=loader,
        epochs=cfg.epochs,
        device=device,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        use_amp=cfg.mixed_precision,
        log_every=cfg.log_every,
        run_name="ddpm_unet_hf",
        sample_every_epochs=cfg.save_every_epochs,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    main()
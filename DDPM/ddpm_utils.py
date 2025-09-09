# DDPM/ddpm_utils.py

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from model import UNET


class Diffusion:
    """
    Minimal DDPM utilities:
    - linear beta schedule
    - forward noise (q(x_t|x_0))
    - sampling (p(x_{t-1}|x_t))
    """
    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        image_size: int = 128,
        channels: int = 3,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.channels = channels
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_over_alphas = torch.sqrt(1.0 / self.alphas)

    def sample_timesteps(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        # t in [0, noise_steps-1]
        return torch.randint(0, self.noise_steps, (n,), device=device, dtype=torch.long)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """
        q(x_t | x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*eps
        """
        eps = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps
        return x_t, eps

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int) -> torch.Tensor:
        """
        Sample images from pure noise using the reverse process.
        Returns [-1, 1] normalized tensors of shape (n, C, H, W).
        """
        model.eval()
        x = torch.randn(n, self.channels, self.image_size, self.image_size, device=self.device)

        for i in tqdm(reversed(range(self.noise_steps)), total=self.noise_steps, desc="Sampling", leave=False):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps_theta = model(x, t)  # predict noise

            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)

            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta) + sigma_t * z
            x = self.sqrt_one_over_alphas[t].view(-1, 1, 1, 1) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta
            )
            if i > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = x + sigma_t * z

        model.train()
        return x  # in [-1, 1]


class Losses:
    def __init__(self, device: torch.device):
        self.device = device
        self.mse = nn.MSELoss()

    def mse_loss(self, pred_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
        return self.mse(pred_noise, true_noise)


@dataclass
class TrainState:
    global_step: int = 0


def train_loop(
    model: UNET,
    diffusion: Diffusion,
    dataloader,
    epochs: int,
    device: torch.device,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = None,
    use_amp: bool = False,
    log_every: int = 100,
    run_name: str = "ddpm-run",
    sample_every_epochs: int = 1,
    save_dir: str = "checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project="ddpm-training", name=run_name, config={
        "lr": lr,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "use_amp": use_amp,
        "image_size": diffusion.image_size,
        "noise_steps": diffusion.noise_steps,
    })

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=use_amp and (device.type == "cuda"))
    losses = Losses(device)
    state = TrainState()

    model.to(device)
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for i, x0 in enumerate(pbar):
            x0 = x0.to(device) 
            B = x0.shape[0]

            # Sample t and noisy image
            t = diffusion.sample_timesteps(B, device)
            x_t, noise = diffusion.add_noise(x0, t)

            with autocast(device_type="cuda", enabled=use_amp and (device.type == "cuda")):
                pred_noise = model(x_t, t)
                loss = losses.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            state.global_step += 1
            pbar.set_postfix(loss=loss.item())
            if state.global_step % log_every == 0:
                wandb.log({"train/loss": loss.item()}, step=state.global_step)

        # Sampling and checkpoint each epoch
        if (epoch + 1) % sample_every_epochs == 0:
            with torch.no_grad():
                samples = diffusion.sample(model, n=min(8, dataloader.batch_size if hasattr(dataloader, 'batch_size') else 8))
            # Map from [-1, 1] to [0, 1] for logging
            grid = (samples.clamp(-1, 1) + 1) / 2.0
            wandb.log({"samples": [wandb.Image(grid[i].cpu()) for i in range(grid.size(0))]}, step=state.global_step)

            ckpt_path = os.path.join(save_dir, f"{run_name}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            wandb.log({"ckpt_path": ckpt_path}, step=state.global_step)

    wandb.finish()
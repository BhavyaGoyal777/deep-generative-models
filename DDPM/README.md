# Denoising Diffusion Probabilistic Models (DDPM) Implementation

This repository contains an implementation of the Denoising Diffusion Probabilistic Models (DDPM) from the paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) by Ho et al. (2020).

## Implementation Details

- Implemented the core DDPM framework for image generation
- Simplified the loss function to use only MSE (removed the second term from the original paper's loss function as it was found to be less relevant for this implementation)
- Training on CIFAR-10 dataset

## Sample Generations

Here are some generated samples from the model:

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
  <img src="DDPM/generated images/media_images_samples_74500_0fa35389577f64b48f23.png" alt="Sample 1" width="100%">
  <img src="DDPM/generated images/media_images_samples_79900_0a0dfc54686cafa07b31.png" alt="Sample 2" width="100%">
  <img src="DDPM/generated images/media_images_samples_83600_3df0e81a7fc858d2a196.png" alt="Sample 3" width="100%">
  <img src="DDPM/generated images/media_images_samples_87200_1f08fd30b3c10ef87c2b.png" alt="Sample 4" width="100%">
  <img src="DDPM/generated images/media_images_samples_87200_89335ee761b404f3d6c8 (1).png" alt="Sample 5" width="100%">
  <img src="DDPM/generated images/media_images_samples_87200_8b076fd951dc9853b824.png" alt="Sample 6" width="100%">
</div>

## Future Work

- Implement DDIM (Denoising Diffusion Implicit Models) sampling for faster generation
- Add exponential moving average (EMA) for model weights
- Implement FID (Fréchet Inception Distance) score calculation for quantitative evaluation
- Add LPIPS (Learned Perceptual Image Patch Similarity) loss for better perceptual quality

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Other standard ML libraries (numpy, matplotlib, etc.)

## Usage

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```



# Deep Generative Models

This repository will contain implementations of various **deep generative models**, each designed to learn and generate complex data distributions, with a focus on image generation.

The goal is to build and study each model from scratch or with minimal dependencies for a deeper understanding of their working principles and differences.

---

##  Models To Be Implemented

| Model                     | Type                | Description |
|--------------------------|---------------------|-------------|
| **DCGAN**                | GAN                 | Deep Convolutional GAN using CNNs for stable image generation. |
| **WGAN**                 | GAN                 | Wasserstein GAN using either weight clipping or gradient penalty for training stability. |
| **BiGAN**                | GAN + Inference     | Bidirectional GAN that includes both a generator and an encoder. |
| **Conditional GAN**      | GAN                 | GAN that generates outputs conditioned on class labels or other data. |
| **CycleGAN**             | GAN                 | Unpaired image-to-image translation using cycle consistency loss. |
| **StyleGAN**             | GAN                 | Style-based generator architecture capable of high-quality image synthesis. |
| **VAE**                  | Latent Variable     | Variational Autoencoder with continuous latent space. |
| **β-VAE**                | Latent Variable     | VAE with disentanglement control using a β term in the KL loss. |
| **VQ-VAE**               | Latent Variable     | Vector Quantized VAE with discrete latent representations. |
| **Masked Autoencoder**   | Self-Supervised AE  | Learns by reconstructing masked image patches (like MAE). |
| **DDPM**                 | Diffusion           | Denoising Diffusion Probabilistic Model for step-by-step image generation. |
| **Score-Based Diffusion**| Diffusion           | Uses score matching to guide generation through reverse-time SDEs. |
| **Conditional Diffusion**| Diffusion           | Diffusion model conditioned on class or text labels. |
| **Latent Diffusion**     | Diffusion + Latent  | Applies diffusion in the latent space for efficiency. |
| **Flow-Based Models**    | Normalizing Flow    | Models like RealNVP or Glow that learn exact data likelihood via invertible mappings. |

---

## Planned Repository Structure

deep-generative-models/
│
├── DCGAN/
├── WGAN/
├── BiGAN/
├── ConditionalGAN/
├── CycleGAN/
├── StyleGAN/
├── VAE/
├── BetaVAE/
├── VQ-VAE/
├── MaskedAE/
├── DDPM/
├── ScoreBased/
├── ConditionalDiffusion/
├── LatentDiffusion/
├── FlowModels/
└── README.md



---

## Planned Features

- Implement each model from scratch using **PyTorch**
- Train on diverse datasets (e.g., Pokémon, landscapes, CIFAR-10)
- Visualize generations across epochs
- Track training metrics and logs using **Weights & Biases**
- Compare models qualitatively and quantitatively (FID, IS, etc.)

---

##  References

- Goodfellow et al., “Generative Adversarial Nets”
- Kingma & Welling, “Auto-Encoding Variational Bayes”
- Ho et al., “Denoising Diffusion Probabilistic Models”
- Karras et al., “A Style-Based Generator Architecture for GANs”
- Rezende & Mohamed, “Variational Inference with Normalizing Flows”

---

Stay tuned as models are implemented one by one with full documentation and results.
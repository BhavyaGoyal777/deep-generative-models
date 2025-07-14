# DCGAN on Pokémon Images

This project showcases a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the [huggan/pokemon](https://huggingface.co/datasets/huggan/pokemon) dataset to generate Pokémon-like creatures.

##  Dataset
- Source: [huggan/pokemon](https://huggingface.co/datasets/huggan/pokemon)
- Number of Images: 833
- Resolution: Resized to **256x256**

##  Model Architecture
The DCGAN model uses:
- Convolutional layers with BatchNorm and ReLU/LeakyReLU activations.
- A Generator that upsamples latent vectors (random noise) into 256x256 Pokémon images.
- A Discriminator that classifies images as real or fake.

## Training Details
- Image size: `256x256`
- Number of epochs: 100
- Optimizer: Adam
- Loss: Binary Cross Entropy
- Framework: PyTorch

## Training Loss Curve

Below is the loss curve showing how Generator and Discriminator loss evolved during training:

![Loss Curve](DCGAN/loss.png)

## Generated Samples

### After 71 Epochs  
![Generated Epoch 71](DCGAN/generated_images/epoch_071.png)

### After 95 Epochs  
![Generated Epoch 95](DCGAN/generated_images/epoch_095.png)



## Future Work
- Train on higher-resolution datasets
- Experiment with conditional GANs using labels (e.g., Pokémon types)
- Compare performance with other GAN variants (e.g., WGAN-GP, StyleGAN)

---


import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of image paths
image_dir = "generated images"
image_files = [
    "media_images_samples_74500_0fa35389577f64b48f23.png",
    "media_images_samples_79900_0a0dfc54686cafa07b31.png",
    "media_images_samples_83600_3df0e81a7fc858d2a196.png",
    "media_images_samples_87200_1f08fd30b3c10ef87c2b.png",
    "media_images_samples_87200_89335ee761b404f3d6c8 (1).png",
    "media_images_samples_87200_8b076fd951dc9853b824.png"
]

# Create figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('DDPM Generated Samples', fontsize=16)

# Load and plot each image
for idx, img_file in enumerate(image_files):
    img_path = os.path.join(image_dir, img_file)
    try:
        img = mpimg.imread(img_path)
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Sample {idx + 1}', fontsize=10)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

# Adjust layout and save
plt.tight_layout()
plt.savefig('ddpm_samples_grid.png', bbox_inches='tight', dpi=150)
print("Grid image saved as 'ddpm_samples_grid.png'")

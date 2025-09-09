
from model import UNET

model = UNET()


total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
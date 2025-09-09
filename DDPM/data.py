import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from datasets import load_dataset

#
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]  
        if self.transform:
            image = self.transform(image)
        return image



def get_data(
    dataset_name="N-o-1/pokemon-images",
    split="train",
    image_size=128,
    batch_size=64,
    num_workers=4,
):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),              
        transforms.ToTensor(),                          
        transforms.Normalize(mean=[0.5, 0.5, 0.5],      
                             std=[0.5, 0.5, 0.5]),
    ])

    hf_data = load_dataset(dataset_name, split=split)
    dataset = HFDataset(hf_data, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,               
        persistent_workers=num_workers > 0,
        drop_last=True,                  
    )

    return loader



if __name__ == "__main__":
    loader = get_data()
    for i, batch in enumerate(loader):
        print(batch.shape)
        if i == 5:
            break    
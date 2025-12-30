
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CSVDataset(Dataset):
    """
    Dataset for single-pathology X-ray regression.
    Handles image loading, optional caching, and label scaling.
    """
    def __init__(self, dataframe, image_root_dir, target_columns,
                 transform=None, save_dir=None, use_saved_images=False):
        self.data = dataframe
        self.image_root_dir = image_root_dir
        self.target_columns = target_columns
        self.transform = transform
        self.save_dir = save_dir
        self.use_saved_images = use_saved_images

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_index = row["Unnamed: 0"]
        saved_path = os.path.join(self.save_dir, f"{image_index}.pt")

        if self.use_saved_images:
            image_tensor = torch.load(saved_path)
        else:
            img_path = os.path.join(self.image_root_dir, row["Path"])
            image = Image.open(img_path).convert("L")
            image_tensor = transforms.ToTensor()(image)
            if self.save_dir:
                torch.save(image_tensor, saved_path)

        label = torch.tensor(
            [(row[self.target_columns[0]] + 1) / 2],
            dtype=torch.float32,
        )
        return image_tensor, label

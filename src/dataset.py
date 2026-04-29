import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import torchvision.transforms as transforms

class ParquetMRIDataset(Dataset):
    """
    Custom Dataset Loader for Parquet files containing MRI images and labels.
    """
    def __init__(self, parquet_path, transform=None):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        
        # Mapping string labels to integers
        self.label_map = {
            'NonDemented': 0, 
            'VeryMildDemented': 1, 
            'MildDemented': 2, 
            'ModerateDemented': 3
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Decode image bytes
        try:
            img_bytes = row['image']['bytes']
        except TypeError:
            img_bytes = row['image']
        except KeyError:
            img_bytes = row['image']
            
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Get Label
        label_val = row.get('label', 0)
        if isinstance(label_val, (int, np.integer)):
            label = int(label_val)
        else:
            label = self.label_map.get(str(label_val), 0)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(is_train=True):
    """
    Transforms optimized for MRI handling
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Image_Dataset_(Dataset):
    def __init__(self, csv_path, train = True, transform=None):
        super(Image_Dataset_, self).__init__()
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.train = train
        
    def __getitem__(self, idx):
        image = Image.open(self.df["path"][idx])
        if self.train:
             label = self.df["label"][idx]

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        if self.train:
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.df["path"])

import multiprocessing
import os
import zipfile

import cv2
import kaggle
import numpy as np
import pandas as pd
import psutil
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split
import albumentations as A


class PogDataset(Dataset):
    def __init__(self, competition_id: str = "kaggle-pog-series-s01e01", image_size: int = 256, scale: int = 1):
        kaggle.api.competition_download_files(competition_id, quiet=False)

        if not os.path.exists(competition_id):
            with zipfile.ZipFile(competition_id + ".zip", 'r') as zip_ref:
                zip_ref.extractall(competition_id)

        df = pd.read_parquet(os.path.join(competition_id, "dataset.parquet")
                             if os.path.exists(os.path.join(competition_id, "dataset.parquet"))
                             else os.path.join(competition_id, "train.parquet"))

        df = df.loc[df["has_thumbnail"] == True]

        self.image_id = df["video_id"]
        self.targets = df["view_count"]
        self.competition_id = competition_id

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=int(image_size * scale)),
            A.PadIfNeeded(min_height=int(image_size * scale), min_width=int(image_size * scale),
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ])

    def __len__(self):
        return self.targets.size

    def __getitem__(self, i):
        image_id = self.image_id.iloc[i]
        targets = self.targets.iloc[i]
        image = cv2.imread(os.path.join("../", self.competition_id, "thumbnails", image_id + ".jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.transform(image=image)
        return result["image"], np.array([targets], dtype=np.float32)


def get_loader(batch_size, competition_name: str = "kaggle-pog-series-s01e01", image_size: int = 256, scale: int = 1):
    dataset = PogDataset(competition_name, image_size, scale)
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=psutil.cpu_count() // 2)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=psutil.cpu_count() // 2)
    return train_loader, val_loader

import os
import zipfile
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# --- CONFIG ---
IMG_SIZE = 200
BATCH_SIZE = 64
VAL_SPLIT = 0.2
SEED = 42

def set_seed(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BeeDataset(Dataset):
    def __init__(self, df, image_dir, img_size=200):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_id):
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.img_size, self.img_size))
        gray = gray.astype(np.float32) / 255.0
        return gray

    def _median_sobel(self, gray):
        med = cv2.medianBlur((gray * 255).astype(np.uint8), 3)
        med = med.astype(np.float32) / 255.0
        gx = cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        max_val = mag.max() + 1e-8
        mag = mag / max_val
        return mag

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        label = int(row["genus"])  # 0 = Bombus, 1 = Apis

        gray = self._load_image(img_id)
        edge = self._median_sobel(gray)
        feat = edge.reshape(-1)

        feat_tensor = torch.from_numpy(feat).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return feat_tensor, label_tensor

def create_dataloaders(csv_path, image_dir,
                       img_size=IMG_SIZE,
                       batch_size=BATCH_SIZE,
                       val_split=VAL_SPLIT,
                       seed=SEED):
    import torch
    set_seed(seed)

    df = pd.read_csv(csv_path)
    assert "id" in df.columns and "genus" in df.columns

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=seed,
        stratify=df["genus"]
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_split,
        random_state=seed,
        stratify=train_df["genus"]
    )

    input_dim = img_size * img_size

    train_dataset = BeeDataset(train_df, image_dir, img_size)
    val_dataset   = BeeDataset(val_df,   image_dir, img_size)
    test_dataset  = BeeDataset(test_df,  image_dir, img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim

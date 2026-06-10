import torch
from torch.utils.data import Dataset, Subset
import rasterio
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class EuroSatDataset(Dataset):
    def __init__(self, root_dir, transform=None, pca=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.pca = pca

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            for tif_path in sorted(class_dir.glob('*.tif')):
                self.images.append(str(tif_path))
                self.labels.append(self.class_to_idx[cls_name])

        print(f"Loaded {len(self.images)} images across {len(self.classes)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        with rasterio.open(img_path) as src:
            img = src.read()  # (13, 64, 64)

        img = torch.from_numpy(img.astype(np.float32))

        # Basic normalization (z-score per band)
        mean = img.mean(dim=(1, 2), keepdim=True)
        std = img.std(dim=(1, 2), keepdim=True) + 1e-8
        img = (img - mean) / std

        if self.pca is not None:
            img = self.apply_pca(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def apply_pca(self, img):
        C, H, W = img.shape
        flat = img.reshape(C, -1).T  # (H*W, C)
        reduced = self.pca.transform(flat.cpu().numpy())
        reduced = torch.from_numpy(reduced.T).float()  # (n_components, H*W)
        return reduced.reshape(-1, H, W)
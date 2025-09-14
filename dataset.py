import os

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

class EuroSATMSDataset(Dataset):
    def __init__(self, root, indices, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith('.tif'):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = tifffile.imread(path)  # shape: (13, H, W)

        if image.ndim == 3 and image.shape[0] != 13:
            image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)

        image = (image - image.min()) / (image.max() - image.min())

        return image, label

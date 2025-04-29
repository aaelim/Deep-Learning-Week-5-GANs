"""
dataset.py
──────────
Utility code for loading the Monet ↔ Photo training images into PyTorch
DataLoader objects.

Key idea
--------
• Treats the two domains as unpaired; each batch contains a random
  pairing of Monet + Photo images.

• Because there are only 300 Monet paintings, but 7k photos, the Dataset
  is written so that __len__ equals the larger domain size, and cycle
  through Monets as needed.
"""

import random
import itertools
import pathlib
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MonetPhotoDataset(Dataset):
    """
    Returns a dict with two keys:
      { "monet": Tensor[C×H×W], "photo": Tensor[C×H×W] }

    Every iteration receive:
      • a Monet image, cycling modulo len(monet)
      • a randomly-sampled Photo image (uniform)

    Normalisation range = [-1, 1]
    """

    def __init__(self, monet_dir: str, photo_dir: str, size: int = 256):
        self.monet  = sorted(pathlib.Path(monet_dir).glob("*.jpg"))
        self.photo  = sorted(pathlib.Path(photo_dir).glob("*.jpg"))
        self.len_m, self.len_p = len(self.monet), len(self.photo)

        # Shared augment / preprocessing pipeline
        self.t = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),                          # [0,1]
            T.Normalize(mean=[0.5]*3, std=[0.5]*3) # → [-1,1]
        ])

    def __len__(self) -> int:
        """
        DataLoader length = the larger domain so that
        one full epoch sees each Photo once.
        """
        return max(self.len_m, self.len_p)

    def __getitem__(self, idx: int):
        # Monet cycles when we run out
        m_path = self.monet[idx % self.len_m]
        # Photo chosen uniformly each call
        p_path = self.photo[random.randint(0, self.len_p - 1)]

        return {
            "monet": self.t(Image.open(m_path).convert("RGB")),
            "photo": self.t(Image.open(p_path).convert("RGB")),
        }


def make_dataloaders(
    monet_dir: str,
    photo_dir: str,
    bs: int,
    workers: int = 4,
) -> DataLoader:
    """Helper that returns a single shuffled training loader."""
    ds = MonetPhotoDataset(monet_dir, photo_dir)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
    )

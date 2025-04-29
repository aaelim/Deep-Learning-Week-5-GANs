"""
utils.py
────────
Common helper functions:
  • seed_all   – reproducible RNG for Python, NumPy, PyTorch, CUDA
  • init_weights – normal(0,0.02) weight init used by most GAN papers
"""

import random
import numpy as np
import torch


def seed_all(seed: int = 42) -> None:
    """Seed Python, NumPy, and both CPU / GPU PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN reproducibility settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(m: torch.nn.Module) -> None:
    """
    Initialise Conv / Deconv / BatchNorm weights to
    N(0, 0.02) as in the original DCGAN paper.
    """
    if isinstance(
        m,
        (
            torch.nn.Conv2d,
            torch.nn.ConvTranspose2d,
            torch.nn.BatchNorm2d,
        ),
    ):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

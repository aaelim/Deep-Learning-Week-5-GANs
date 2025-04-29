"""
models.py
─────────
CycleGAN architecture:

  • Generator  – ResNet-based encoder-decoder with reflection padding
  • Discriminator – 70×70 PatchGAN

Both networks use InstanceNorm (style-transfer best practice).
"""

import torch
import torch.nn as nn
from utils import init_weights


#  Building blocks
class ResnetBlock(nn.Module):
    """Two 3×3 convs with residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


#  Generator
class Generator(nn.Module):
    """
    ResNet-based generator as in CycleGAN paper.

    Args
    ----
    in_c      : # input channels (3 for RGB)
    ngf       : # filters in first conv layer
    n_blocks  : # ResNet blocks (6 for 256×256, 9 for 512×512 images)
    """

    def __init__(self, in_c: int = 3, ngf: int = 64, n_blocks: int = 6):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # Down-sampling (×2)
        curr = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(curr, curr * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(curr * 2),
                nn.ReLU(True),
            ]
            curr *= 2

        # N× ResNet blocks
        layers += [ResnetBlock(curr) for _ in range(n_blocks)]

        # Up-sampling (×2)
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    curr, curr // 2, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(curr // 2),
                nn.ReLU(True),
            ]
            curr //= 2

        # Final RGB conv + tanh to map into [-1,1]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr, 3, 7),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)


#  PatchGAN Discriminator
class Discriminator(nn.Module):
    """
    70×70 PatchGAN: classifies each 70×70 patch as real/fake → 1×H'×W' map.

    Args
    ----
    in_c  : # channels in input image
    ndf   : # filters in first layer
    """

    def __init__(self, in_c: int = 3, ndf: int = 64):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        ]

        curr = ndf
        for n in [curr * 2, curr * 4, curr * 8]:
            layers += [
                nn.Conv2d(
                    curr,
                    n,
                    4,
                    stride=2 if n != curr * 8 else 1,
                    padding=1,
                ),
                nn.InstanceNorm2d(n),
                nn.LeakyReLU(0.2, True),
            ]
            curr = n

        layers += [nn.Conv2d(curr, 1, 4, 1, 1)]  # output logits

        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

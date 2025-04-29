"""
infer.py
────────
Offline inference script.

Loads a trained Generator checkpoint and converts every .jpg in
--photo_dir into Monet-style images, saving them into --out_dir.
"""

import argparse
import pathlib
from PIL import Image

import torch
import torchvision.transforms as T

from models import Generator


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load generator weights
    G = Generator().to(device).eval()
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Output directory
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Pre-processing transform
    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    # Loop through photos, save stylised images
    for p in pathlib.Path(args.photo_dir).glob("*.jpg"):
        img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        fake = G(img).clamp(-1, 1)  # [-1,1]
        fake = (fake * 0.5 + 0.5) * 255  # → [0,255]
        Image.fromarray(
            fake.squeeze().permute(1, 2, 0).cpu().numpy().astype("uint8")
        ).save(out_dir / f"{p.stem}.jpg", quality=95)

    print("Generated", len(list(out_dir.glob("*.jpg"))), "images at", out_dir)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", required=True, help="*.pt file from training")
    a.add_argument("--photo_dir", required=True, help="Directory with input photos")
    a.add_argument("--out_dir", default="gen", help="Where to write *.jpg outputs")
    main(a.parse_args())

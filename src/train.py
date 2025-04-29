"""
train.py
────────
CycleGAN training loop for Monet-style transfer.


Mixed-precision (AMP) support via --fp16
Resume from checkpoint with --resume
Per-epoch checkpoints + CSV loss log
Prints a concise summary line each epoch
"""

# — imports —
import argparse, csv, itertools, sys
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from dataset import make_dataloaders
from models  import Generator, Discriminator
from utils   import seed_all

# helper utils
def cycle(it):
    """Re-creates itertools.cycle but avoids importing whole module elsewhere."""
    while True:
        for x in it:
            yield x


def save_ckpt(gen: nn.Module, out_dir: Path, epoch: int) -> Path:
    """
    Save generator weights every N epochs and also duplicate to 'latest_G_A2B.pt'
    so inference scripts can always grab the freshest checkpoint.
    """
    ckpt = out_dir / f"E{epoch:03d}_G_A2B.pt"
    torch.save(gen.state_dict(), ckpt)
    torch.save(gen.state_dict(), out_dir / "latest_G_A2B.pt")
    return ckpt


# main routine
def main(cfg):
    seed_all()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create networks 
    G_A2B, G_B2A = Generator().to(device), Generator().to(device)
    D_A, D_B     = Discriminator().to(device), Discriminator().to(device)

    # optimisers + schedulers 
    opt_G = optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )
    total_iters = cfg.epochs * cfg.steps_per_epoch
    sched_G = optim.lr_scheduler.LinearLR(opt_G, 1.0, 0.0, total_iters)
    sched_D = optim.lr_scheduler.LinearLR(opt_D, 1.0, 0.0, total_iters)

    #  loss functions 
    L1, MSE = nn.L1Loss(), nn.MSELoss()
    scaler  = GradScaler(enabled=cfg.fp16)

    #  dataloader 
    dl      = make_dataloaders(cfg.monet_dir, cfg.photo_dir,
                               bs=cfg.batch_size, workers=4)
    dl_iter = cycle(dl)

    #  bookkeeping paths 
    out_dir = Path(cfg.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = out_dir / "losses.csv"
    if not loss_csv.exists():
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "loss_G", "loss_D_A", "loss_D_B"])

    #  resume logic 
    start_epoch = 0
    global_step = 0
    if cfg.resume and Path(cfg.resume).is_file():
        G_A2B.load_state_dict(torch.load(cfg.resume, map_location=device))
        try:
            start_epoch = int(Path(cfg.resume).stem.split("_")[0][1:])
            print(f"[INFO] Resumed from {cfg.resume} (epoch {start_epoch})")
        except Exception:
            print(f"[INFO] Resumed weights from {cfg.resume}")

    #  main epoch loop 
    for epoch in range(start_epoch, cfg.epochs):
        epoch_G = epoch_DA = epoch_DB = 0.0

        for step in range(cfg.steps_per_epoch):
            data = next(dl_iter)
            real_A = data["photo"].to(device)  # domain A
            real_B = data["monet"].to(device)  # domain B

            # GENERATORS 
            opt_G.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.fp16):
                fake_B = G_A2B(real_A)   # A to B
                fake_A = G_B2A(real_B)   # B to A

                rec_A  = G_B2A(fake_B)   # A' to A
                rec_B  = G_A2B(fake_A)   # B' to B

                idt_A  = G_B2A(real_A)   # identity loss
                idt_B  = G_A2B(real_B)

                loss_idt = (L1(idt_A, real_A) + L1(idt_B, real_B)) * 0.5 * 5.0
                loss_cyc = (L1(rec_A, real_A) + L1(rec_B, real_B)) * 10.0
                loss_adv = (
                    MSE(D_B(fake_B), torch.ones_like(D_B(fake_B))) +
                    MSE(D_A(fake_A), torch.ones_like(D_A(fake_A)))
                )

                loss_G = loss_adv + loss_cyc + loss_idt

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()
            sched_G.step()

            # DISCRIMINATORS 
            for netD, real, fake in [
                (D_A, real_A, fake_A.detach()),
                (D_B, real_B, fake_B.detach())
            ]:
                opt_D.zero_grad(set_to_none=True)
                with autocast(enabled=cfg.fp16):
                    loss_real = MSE(netD(real),  torch.ones_like(netD(real)))
                    loss_fake = MSE(netD(fake), torch.zeros_like(netD(fake)))
                    loss_D = (loss_real + loss_fake) * 0.5

                scaler.scale(loss_D).backward()
                scaler.step(opt_D)
                scaler.update()
                sched_D.step()

                if netD is D_A:
                    loss_DA_val = loss_D.item()
                else:
                    loss_DB_val = loss_D.item()

            # accumulate stats
            epoch_G  += loss_G.item()
            epoch_DA += loss_DA_val
            epoch_DB += loss_DB_val

            # log every step for later curve-plot
            with open(loss_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [global_step, loss_G.item(), loss_DA_val, loss_DB_val]
                )
            global_step += 1

        #  end-of-epoch: summary and checkpoint 
        avg_G  = epoch_G  / cfg.steps_per_epoch
        avg_DA = epoch_DA / cfg.steps_per_epoch
        avg_DB = epoch_DB / cfg.steps_per_epoch

        if (epoch + 1) % cfg.save_freq == 0:
            ckpt = save_ckpt(G_A2B, out_dir, epoch + 1)
        else:
            ckpt = out_dir / "latest_G_A2B.pt"

        print(
            f"Epoch {epoch+1:>3}/{cfg.epochs} | "
            f"G={avg_G:6.3f} | DA={avg_DA:5.3f} | DB={avg_DB:5.3f} | "
            f"chkpt={ckpt.name}",
            flush=True,
        )

    # final save
    ckpt = save_ckpt(G_A2B, out_dir, cfg.epochs)
    print(f"[INFO] Training complete – final checkpoint: {ckpt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--monet_dir",  required=True)
    p.add_argument("--photo_dir",  required=True)
    p.add_argument("--save_dir",   default="outputs")
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps_per_epoch", type=int, default=400)
    p.add_argument("--save_freq",  type=int, default=1,
                   help="save every N epochs (default 1)")
    p.add_argument("--resume",     default="",
                   help="path to checkpoint to resume from")
    p.add_argument("--fp16",       action="store_true")
    cfg = p.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)
    main(cfg)


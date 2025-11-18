# train.py
import argparse, time, random, copy
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

from src.data import Div2kDataSet
from src.model import (
    UNetDenoise,
    apply_channel_pruning,
    apply_unstructured_pruning,
    bake_unstructured_pruning,
)


# ------------------------- utils -------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def psnr(batch_pred, batch_gt, eps=1e-8):
    mse = torch.mean((batch_pred - batch_gt) ** 2, dim=(1, 2, 3))
    return torch.mean(20.0 * torch.log10(1.0 / torch.sqrt(mse + eps)))


def save_ckpt(state, save_dir, tag):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{tag}.pth"
    torch.save(state, path)
    return str(path)


# ------------------------- train/eval steps -------------------------
def run_epoch(model, loader, optimizer, device, train=True, grad_clip=0.0, criterion=nn.MSELoss()):
    model.train(train)
    loss_meter, psnr_meter, n = 0.0, 0.0, 0
    mode = "train" if train else "val"

    pbar = tqdm(loader, desc=f"{mode} epoch", leave=False, dynamic_ncols=True)

    for noisy, clean in pbar:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            out = model(noisy)
            loss = criterion(out, clean)
            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(noisy)
                loss = criterion(out, clean)

        with torch.no_grad():
            batch_psnr = psnr(out, clean)

        bs = noisy.size(0)
        loss_meter += loss.item() * bs
        psnr_meter += batch_psnr.item() * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{batch_psnr.item():.2f}")

    if n == 0:
        raise ValueError("No samples processed — check DataLoader or loop indentation.")
    return loss_meter / n, psnr_meter / n


# ------------------------- main -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="DIV2K root with train/ and (optional) val/ pngs")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--model_tag", type=str, default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--sigma", type=float, default=25.0,
                   help="noise std in [0,255] terms; dataset scales to [0,1]")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--use_channel_prune", action="store_true")
    p.add_argument("--channel_prune_sparsity", type=float, default=0.1)
    p.add_argument("--channel_prune_steps", type=int, default=1)
    p.add_argument("--use_unstructured_prune", action="store_true")
    p.add_argument("--unstructured_amount", type=float, default=0.5)
    p.add_argument("--use_qat", action="store_true")


    args = p.parse_args()

    print(f"===== training {args.model_tag} =====\n")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ------------------ datasets ------------------
    train_set = Div2kDataSet(
        root=args.data_root,
        split="train",
        patch_size=args.patch_size,
        sigma=args.sigma,
        augment=True,
    )
    val_set = Div2kDataSet(
        root=args.data_root,
        split="valid",
        patch_size=args.patch_size,
        sigma=args.sigma,
        augment=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------ model (base UNet) ------------------
    model = UNetDenoise(
        in_ch=3,
        out_ch=3,
        base_ch=args.base_ch,
    ).to(device)

    # ------------------ channel pruning  ------------------
    if args.use_channel_prune:
        example_input = torch.randn(1, 3, args.patch_size, args.patch_size, device=device)
        model = apply_channel_pruning(
            model,
            example_input=example_input,
            ch_sparsity=args.channel_prune_sparsity,
            iterative_steps=args.channel_prune_steps,
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------------ QAT setup ------------------
    if args.use_qat:
        # QAT is applied on the (possibly channel-pruned) float model
        example_input = torch.randn(1, 3, args.patch_size, args.patch_size, device=device)

        qconfig = tq.get_default_qat_qconfig("fbgemm")
        model = prepare_qat_fx(model, {"": qconfig}, example_inputs=(example_input,))

        # Reduce LR for QAT fine-tuning
        opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)

    best_psnr = -1.0

    print("starting loop")
    # ------------------ training loop ------------------
    for epoch in range(args.epochs):
        t0 = time.time()

        tr_loss, tr_psnr = run_epoch(
            model, train_loader, opt, device,
            train=True, grad_clip=args.grad_clip
        )
        va_loss, va_psnr = run_epoch(
            model, val_loader, opt, device,
            train=False
        )

        elapsed = time.time() - t0

        print(
            f"[{epoch+1:03d}/{args.epochs}] "
            f"train: loss={tr_loss:.6f}, psnr={tr_psnr:.2f} | "
            f"val: loss={va_loss:.6f}, psnr={va_psnr:.2f} | "
            f"time={elapsed:.1f}s"
        )

        # track and save best float checkpoint (optionally unstructured-pruned)
        if va_psnr > best_psnr:
            best_psnr = va_psnr

            # clone model for saving so we don't disturb the training model
            model_for_save = copy.deepcopy(model).cpu()

            if args.use_unstructured_prune:
                model_for_save = apply_unstructured_pruning(
                    model_for_save,
                    amount=args.unstructured_amount,
                )
                model_for_save = bake_unstructured_pruning(model_for_save)

            path = save_ckpt(
                {
                    "epoch": epoch,
                    "model": model_for_save.state_dict(),
                    "best_psnr": best_psnr,
                    "args": vars(args),
                },
                args.save_dir,
                f"{args.model_tag}_best",
            )
            print(f"  ↳ new best ({best_psnr:.2f} dB) saved to {path}")

    # after the training loop, export quantized model if QAT was used
    if args.use_qat:
        # for inference, run quantized model on CPU
        model_cpu = model.to("cpu").eval()

        # convert_fx uses the observer stats collected during QAT
        quant_model = convert_fx(model_cpu)

        quant_path = save_ckpt(
            {
                "epoch": args.epochs - 1,
                "model": quant_model.state_dict(),
                "best_psnr": best_psnr,
                "args": vars(args),
            },
            args.save_dir,
            f"{args.model_tag}_int8_final",
        )
        print(f"[info] quantized INT8 model saved to {quant_path}")


if __name__ == "__main__":
    main()

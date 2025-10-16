# src/eval.py
import torch, argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import UNetDenoise
from src.data import Div2kDataSet


@torch.no_grad()
def psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=(1,2,3))
    return torch.mean(20.0 * torch.log10(1.0 / torch.sqrt(mse + eps)))


@torch.no_grad()
def ssim(pred, target, C1=0.01 ** 2, C2=0.03 ** 2):
    """Simple differentiable SSIM for [0,1] tensors"""
    mu_x = torch.mean(pred, dim=(2,3), keepdim=True)
    mu_y = torch.mean(target, dim=(2,3), keepdim=True)
    sigma_x = torch.var(pred, dim=(2,3), keepdim=True)
    sigma_y = torch.var(target, dim=(2,3), keepdim=True)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=(2,3), keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return torch.mean(ssim_map)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    loss_meter, psnr_meter, ssim_meter, n = 0.0, 0.0, 0.0, 0

    for noisy, clean in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy)
        loss = loss_fn(out, clean)
        batch_psnr = psnr(out, clean)
        batch_ssim = ssim(out, clean)
        bs = noisy.size(0)

        loss_meter += loss.item() * bs
        psnr_meter += batch_psnr.item() * bs
        ssim_meter += batch_ssim.item() * bs
        n += bs

    return {
        "Loss": loss_meter / n,
        "PSNR": psnr_meter / n,
        "SSIM": ssim_meter / n
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="DIV2K dataset root")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint .pth")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    val_set = Div2kDataSet(root=args.data_root, split="valid",
                           patch_size=args.patch_size, sigma=25.0, augment=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = UNetDenoise(in_ch=3, out_ch=3, base_ch=64).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.ckpt_path}")

    results = evaluate(model, val_loader, device)
    print(f"\n Evaluation Results:")
    print(f"Loss : {results['Loss']:.6f}")
    print(f"PSNR : {results['PSNR']:.2f} dB")
    print(f"SSIM : {results['SSIM']:.4f}")


if __name__ == "__main__":
    main()

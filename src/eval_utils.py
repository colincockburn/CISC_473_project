import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
import yaml

from src.model import UNetDenoise, apply_channel_pruning
from src.data import Div2kDataSet

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch PSNR for [0,1] images: (N,C,H,W) -> scalar."""
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return torch.mean(20.0 * torch.log10(1.0 / torch.sqrt(mse + eps)))


@torch.no_grad()
def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Very simple SSIM over whole image, per-batch, for [0,1] tensors."""
    mu_x = torch.mean(pred, dim=(2, 3), keepdim=True)
    mu_y = torch.mean(target, dim=(2, 3), keepdim=True)
    sigma_x = torch.var(pred, dim=(2, 3), keepdim=True)
    sigma_y = torch.var(target, dim=(2, 3), keepdim=True)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=(2, 3), keepdim=True)

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / den
    return torch.mean(ssim_map)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute Loss / PSNR / SSIM over our dataloader."""
    model.eval()
    loss_fn = nn.MSELoss()
    loss_meter, psnr_meter, ssim_meter, n = 0.0, 0.0, 0.0, 0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

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
        "SSIM": ssim_meter / n,
    }


def count_params(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def model_sparsity(model: nn.Module) -> float:
    """Fraction of weights that are exactly zero."""
    zero = 0
    total = 0
    for p in model.parameters():
        t = p.detach()
        zero += (t == 0).sum().item()
        total += t.numel()
    return zero / total if total > 0 else 0.0


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    device: torch.device,
    input_shape=(1, 3, 128, 128),
    warmup: int = 10,
    runs: int = 50,
) -> Tuple[float, float]:
    """
    Returns (mean_ms, std_ms) over `runs` iterations.
    """
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    # warmup
    for _ in range(warmup):
        _ = model(x)

    times_ms = []
    for _ in range(runs):
        t0 = time.time()
        _ = model(x)
        times_ms.append((time.time() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def build_base_or_pruned_unet_from_args(args: dict, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Rebuild a float UNet (base or channel-pruned) using the args stored in the checkpoint.
    Returns (model, patch_size).
    """
    base_ch = int(args.get("base_ch", 64))
    patch_size = int(args.get("patch_size", 128))

    model = UNetDenoise(
        in_ch=3,
        out_ch=3,
        base_ch=base_ch,
    ).to(device)

    use_channel_prune = bool(args.get("use_channel_prune", False))
    if use_channel_prune:
        ch_sparsity = float(args.get("channel_prune_sparsity", 0.5))
        steps = int(args.get("channel_prune_steps", 1))
        example_input = torch.randn(1, 3, patch_size, patch_size, device=device)
        print(
            f"[Eval] Re-applying channel pruning in eval: "
            f"ch_sparsity={ch_sparsity}, steps={steps}"
        )
        model = apply_channel_pruning(
            model,
            example_input=example_input,
            ch_sparsity=ch_sparsity,
            iterative_steps=steps,
        ).to(device)

    return model, patch_size


def load_float_model(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    """
    Load a float model (base or channel-pruned) from a _best.pth checkpoint.

    The checkpoint structure is assumed to be:
        {
          "epoch": ...,
          "model": state_dict,
          "best_psnr": ...,
          "args": vars(args),
        }
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})

    model, patch_size = build_base_or_pruned_unet_from_args(args, device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, patch_size


def load_quant_model(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    """
    Load a QAT / QAT+channel-pruned checkpoint as a quantized INT8 FX model.

    We:
      - Rebuild the base/pruned UNet from args,
      - Rebuild the QAT FX graph using the same qconfig + example_input logic,
      - Convert to a quantized FX model via convert_fx,
      - Load the saved INT8 state_dict into that quantized model.

    This allows us to benchmark *true* INT8 latency and quality.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})

    # Build float base/pruned architecture on CPU
    cpu_device = torch.device("cpu")
    float_model, patch_size = build_base_or_pruned_unet_from_args(args, cpu_device)

    # Rebuild QAT FX graph + quantized FX graph template
    example_input = torch.randn(1, 3, patch_size, patch_size, device=cpu_device)
    qconfig = tq.get_default_qat_qconfig("fbgemm")
    qat_model = prepare_qat_fx(float_model, {"": qconfig}, example_inputs=(example_input,))
    quant_model = convert_fx(qat_model.eval())

    # Load saved INT8 state_dict (weights + scales) into this quant graph
    quant_sd = ckpt["model"]
    missing, unexpected = quant_model.load_state_dict(quant_sd, strict=False)
    if unexpected:
        print(f"[Eval QAT] Ignoring {len(unexpected)} unexpected keys (aux buffers).")
    if missing:
        print(f"[Eval QAT] {len(missing)} missing keys kept at init values.")

    quant_model.to(device).eval()
    return quant_model, patch_size


def build_val_loader(data_root: Path, cfg: dict) -> DataLoader:
    """Build DIV2K validation loader using cfg[data]."""
    patch_size = int(cfg["data"]["patch_size"])
    sigma = float(cfg["data"].get("sigma", 25.0))

    val_set = Div2kDataSet(
        root=str(data_root),
        split="valid",
        patch_size=patch_size,
        sigma=sigma,
        augment=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["data"].get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=True,
    )
    return val_loader


def evaluate_all_models(
    cfg: dict,
    data_root: Path,
    save_dir: Path,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Run full evaluation for:
      - base        (float)
      - prune       (float, channel-pruned)
      - qat         (INT8)
      - qat_prune   (INT8, channel-pruned)

    Returns:
      results[name] = {
          Loss, PSNR, SSIM,
          Params, Sparsity,
          Size_MB, Latency_ms_mean, Latency_ms_std,
      }
    """
    val_loader = build_val_loader(data_root, cfg)

    variants = {
        "base": {
            "tag": cfg["saves"]["base_model_name"],
            "ckpt_suffix": "_best.pth",
            "quant": False,
        },
        "prune": {
            "tag": cfg["saves"]["prune_model_name"],
            "ckpt_suffix": "_best.pth",
            "quant": False,
        },
        "qat": {
            "tag": cfg["saves"]["qat_model_name"],
            "ckpt_suffix": "_int8_final.pth",
            "quant": True,
        },
        "qat_prune": {
            "tag": cfg["saves"]["qat_prune_model_name"],
            "ckpt_suffix": "_int8_final.pth",
            "quant": True,
        },
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, conf in variants.items():
        tag = conf["tag"]
        ckpt_path = save_dir / f"{tag}{conf['ckpt_suffix']}"
        assert ckpt_path.exists(), f"Missing checkpoint for {name}: {ckpt_path}"

        print(f"\n===== {name.upper()} =====")
        print(f"CKPT: {ckpt_path}")

        # Load model (float or quantized)
        if conf["quant"]:
            model, patch_size = load_quant_model(ckpt_path, device=device)
        else:
            model, patch_size = load_float_model(ckpt_path, device=device)

        # Quality metrics
        qa = evaluate(model, val_loader, device)

        # Params / sparsity
        model_cpu = model.to("cpu")
        params = count_params(model_cpu)
        sparsity = model_sparsity(model_cpu)

        # Checkpoint size
        size_mb = ckpt_path.stat().st_size / (1024 ** 2)

        # Latency on CPU (batch=1, patch_size from checkpoint)
        lat_mean, lat_std = benchmark_model(
            model_cpu, torch.device("cpu"),
            input_shape=(1, 3, patch_size, patch_size)
        )

        results[name] = {
            "Loss": qa["Loss"],
            "PSNR": qa["PSNR"],
            "SSIM": qa["SSIM"],
            "Params": params,
            "Sparsity": sparsity,
            "Size_MB": size_mb,
            "Latency_ms_mean": lat_mean,
            "Latency_ms_std": lat_std,
        }

    # 
    if "base" in results and "qat" in results:
        results["qat"]["Params"] = results["base"]["Params"]
        results["qat"]["Sparsity"] = results["base"]["Sparsity"]

    if "prune" in results and "qat_prune" in results:
        results["qat_prune"]["Params"] = results["prune"]["Params"]
        results["qat_prune"]["Sparsity"] = results["prune"]["Sparsity"]

    return results



def print_summary_table(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print a summary table with deltas vs base."""
    base = results["base"]
    base_psnr = base["PSNR"]
    base_ssim = base["SSIM"]
    base_lat = base["Latency_ms_mean"]

    header = (
        "Model        "
        "Params(M)  "
        "Size(MB)  "
        "Sparsity  "
        "Lat(ms)  "
        "Speedup  "
        "PSNR(dB)  "
        "ΔPSNR  "
        "SSIM   "
        "ΔSSIM"
    )
    print("\n===== SUMMARY TABLE =====")
    print(header)
    print("-" * len(header))

    for name in ["base", "prune", "qat", "qat_prune"]:
        r = results[name]
        params_m = r["Params"] / 1e6
        size_mb = r["Size_MB"]
        sparse = r["Sparsity"]
        lat = r["Latency_ms_mean"]
        speedup = base_lat / lat if lat > 0 else 0.0
        psnr_v = r["PSNR"]
        ssim_v = r["SSIM"]
        d_psnr = psnr_v - base_psnr
        d_ssim = ssim_v - base_ssim

        print(
            f"{name:<12}"
            f"{params_m:8.2f}  "
            f"{size_mb:7.2f}  "
            f"{sparse:7.2f}  "
            f"{lat:7.2f}  "
            f"{speedup:7.2f}  "
            f"{psnr_v:7.2f}  "
            f"{d_psnr:6.2f}  "
            f"{ssim_v:5.4f}  "
            f"{d_ssim:+6.4f}"
        )


def main():
    """CLI entrypoint: read env/cfg, evaluate all models, print summary."""
    REPO_DIR = os.getenv("REPO_DIR")
    DATA_ROOT = os.getenv("DATA_ROOT")
    SAVE_DIR = os.getenv("SAVE_DIR")

    assert REPO_DIR is not None, "REPO_DIR env var must be set"
    assert DATA_ROOT is not None, "DATA_ROOT env var must be set"
    assert SAVE_DIR is not None, "SAVE_DIR env var must be set"

    repo_path = Path(REPO_DIR)
    data_root = Path(DATA_ROOT)
    save_dir = Path(SAVE_DIR)

    with open(repo_path / "configs" / "default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")  # use CPU for fair quantized speed comparison
    print(f"Using device for PyTorch eval: {device}")

    results = evaluate_all_models(cfg, data_root, save_dir, device=device)
    print_summary_table(results)


if __name__ == "__main__":
    main()

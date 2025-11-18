#!/usr/bin/env python
# export_onnx_models.py

import torch
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from pathlib import Path
import os
from src.model import UNetDenoise
import yaml
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq


# ------------------------------------------------------------------
# Config: edit these paths/names if needed
# ------------------------------------------------------------------
CKPT_DIR = Path("checkpoints/run1")
ONNX_DIR = Path(os.getenv("EXPORT_DIR"))
SAVE_DIR =  os.getenv("SAVE_DIR")
REPO_DIR = os.getenv("REPO_DIR")

with open(f"{REPO_DIR}/configs/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ckpt_paths = {
    cfg["saves"]["base_model_name"]:              Path(SAVE_DIR) / f"{cfg['saves']['base_model_name']}_best.pth",
    cfg["saves"]["prune_model_name"]:             Path(SAVE_DIR) / f"{cfg['saves']['prune_model_name']}_best.pth",
    cfg["saves"]["qat_model_name"] + "_int8":     Path(SAVE_DIR) / f"{cfg['saves']['qat_model_name']}_int8_final.pth",
    cfg["saves"]["qat_prune_model_name"] + "_int8": Path(SAVE_DIR) / f"{cfg['saves']['qat_prune_model_name']}_int8_final.pth",
}

ONNX_PATHS = {
    "base":          ONNX_DIR / "base.onnx",
    "prune":         ONNX_DIR / "prune.onnx",
    "qat_int8":      ONNX_DIR / "qat_dequant.onnx",
    "qat_prune_int8": ONNX_DIR / "qat_prune_dequant.onnx",
}


# ------------------------------------------------------------------
# Shared ONNX export helper
# ------------------------------------------------------------------
def export_onnx(model: torch.nn.Module, onnx_path: Path, patch_size: int = 128):
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, patch_size, patch_size)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        input_names=["noisy"],
        output_names=["denoised"],
        dynamic_axes={"noisy": {0: "batch"}, "denoised": {0: "batch"}},
        opset_version=17,
    )
    print(f"[OK] Exported ONNX -> {onnx_path}")


# ------------------------------------------------------------------
# Float (base / prune) loaders
# ------------------------------------------------------------------
def load_base_or_prune_model(ckpt_path: Path, use_pruning: bool) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})

    base_ch = args.get("base_ch", 64)
    patch_size = args.get("patch_size", 128)

    model = UNetDenoise(
        in_ch=3,
        out_ch=3,
        base_ch=base_ch,
        use_pruning=use_pruning,

    )

    model.load_state_dict(ckpt["model"])

    # For pruned model, bake pruning masks into real weights
    if use_pruning:
        model.remove_pruning_reparam()

    return model, patch_size


# ------------------------------------------------------------------
# QAT INT8 → dequantized float model
# ------------------------------------------------------------------
def load_qat_dequant_model(ckpt_path: Path) -> tuple[torch.nn.Module, int]:
    """
    Load an INT8 QAT checkpoint and map its (possibly quantized) tensors
    onto a plain float UNetDenoise by dequantizing where needed.
    Result: float model suitable for ONNX export, but with QAT-trained weights.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})

    base_ch = args.get("base_ch", 64)
    patch_size = args.get("patch_size", 128)

    # Plain float backbone (no pruning needed for ONNX –
    # pruning just zeros channels, shapes don’t change)
    model = UNetDenoise(
        in_ch=3,
        out_ch=3,
        base_ch=base_ch,
        use_pruning=False,
    )

    float_sd = model.state_dict()
    q_sd = ckpt["model"]

    for k in float_sd.keys():
        if k in q_sd:
            v = q_sd[k]
            if isinstance(v, torch.Tensor) and v.is_quantized:
                float_sd[k] = v.dequantize()
            else:
                float_sd[k] = v

    # Ignore extra quantization-specific keys in q_sd
    model.load_state_dict(float_sd, strict=False)

    return model, patch_size



# ------------------------------------------------------------------
# Main: export all four variants
# ------------------------------------------------------------------
def main():
    for name, ckpt_path in ckpt_paths.items():
        print(f"\n===== Exporting {name} from {ckpt_path} =====")
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        onnx_path = ONNX_PATHS[name]

        if name == cfg["saves"]["base_model_name"]:
            model, patch = load_base_or_prune_model(ckpt_path, use_pruning=False)

        elif name == cfg["saves"]["prune_model_name"]:
            model, patch = load_base_or_prune_model(ckpt_path, use_pruning=True)

        elif name == cfg["saves"]["qat_model_name"] + "_int8":
            model, patch = load_qat_dequant_model(ckpt_path)

        elif name == cfg["saves"]["qat_prune_model_name"] + "_int8":
            model, patch = load_qat_dequant_model(ckpt_path)

        else:
            raise ValueError(f"Unknown model name: {name}")

        export_onnx(model, onnx_path, patch_size=patch)

    print("\nAll ONNX exports done.")


if __name__ == "__main__":
    main()

import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx

from src.eval_utils import (
    build_base_or_pruned_unet_from_args,
    load_float_model,
)


def export_onnx(model: nn.Module, onnx_path: Path, patch_size: int):
    """ export model as onnx file """
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, patch_size, patch_size)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["noisy"],
        output_names=["denoised"],
        opset_version=17,
        dynamic_axes={"noisy": {0: "batch"}, "denoised": {0: "batch"}},
    )
    print(f"[ONNX] Exported → {onnx_path}")


# QAT → Float Reconstruction
def load_float_from_qat_checkpoint(ckpt_path: Path):
    """
    convert our qat model to a float model so it can be output by onnx.
    """

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    qat_sd = ckpt["model"]

    # rebuild float architecture
    float_model, patch_size = build_base_or_pruned_unet_from_args(args, device="cpu")

    # rebuild QAT graph to align weights
    example = torch.randn(1, 3, patch_size, patch_size)
    qconfig = tq.get_default_qat_qconfig("fbgemm")
    qat_graph = prepare_qat_fx(float_model, {"": qconfig}, example_inputs=(example,))

    # Load checkpoint
    qat_graph.load_state_dict(qat_sd, strict=False)

    # copy all weights into a clean float model
    float_model_rebuilt, _ = build_base_or_pruned_unet_from_args(args, device="cpu")
    float_model_rebuilt.load_state_dict(qat_graph.state_dict(), strict=False)

    return float_model_rebuilt, patch_size


def main():
    REPO_DIR = Path(os.getenv("REPO_DIR"))
    SAVE_DIR = Path(os.getenv("SAVE_DIR"))
    EXPORT_DIR = Path(os.getenv("EXPORT_DIR"))

    assert REPO_DIR.exists(), "REPO_DIR does not exist"
    assert SAVE_DIR.exists(), "SAVE_DIR does not exist"

    with open(REPO_DIR / "configs" / "default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    names = {
        "base": cfg["saves"]["base_model_name"],
        "prune": cfg["saves"]["prune_model_name"],
        "qat": cfg["saves"]["qat_model_name"],
        "qat_prune": cfg["saves"]["qat_prune_model_name"],
    }

    for key, tag in names.items():
        ckpt_path = SAVE_DIR / f"{tag}_best.pth"
        onnx_path = EXPORT_DIR / f"{tag}.onnx"

        print(f"\n===== Exporting {key.upper()} =====")
        print(f"CKPT: {ckpt_path}")

        if not ckpt_path.exists():
            print("  → missing, skipping")
            continue

        # QAT models must be converted to FLOAT before ONNX export
        is_qat = key in ("qat", "qat_prune")

        if is_qat:
            model, patch_size = load_float_from_qat_checkpoint(ckpt_path, device="cpu")
        else:
            model, patch_size = load_float_model(ckpt_path, device="cpu")

        export_onnx(model, onnx_path, patch_size)

    print("\nAll ONNX exports completed successfully.")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune  
import torch_pruning as tp

class DoubleConv(nn.Module):
    """ reusable convolution layer to be stacked in model """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """ endconding down section of Unet"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ decoding up section of Unet"""
    def __init__(self, c_in, c_skip, c_out=None):
        super().__init__()
        if c_out is None:
            c_out = c_skip
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.proj = nn.Conv2d(c_in, c_skip, kernel_size=1, bias=False)
        self.conv = DoubleConv(2 * c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.proj(x)

        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        x = F.pad(x, (0, dw, 0, dh))

        return self.conv(torch.cat([skip, x], dim=1))


class UNetDenoise(nn.Module):
    """ our UNet denoiser model class."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 64):
        super().__init__()
        # encoder
        self.inc   = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        # bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)
        # decoder
        self.up3 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8,  base_ch * 4)
        self.up1 = Up(base_ch * 4,  base_ch * 2)
        self.up0 = Up(base_ch * 2,  base_ch)
        # head
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up3(x5, x4)
        x = self.up2(x,  x3)
        x = self.up1(x,  x2)
        x = self.up0(x,  x1)
        noise_pred = self.outc(x)
        return torch.clamp(x_in - noise_pred, 0, 1)


def apply_channel_pruning(
    model: nn.Module,
    example_input: torch.Tensor,
    ch_sparsity: float = 0.5,
    iterative_steps: int = 1,
):
    """ pruning entire channels lets us reduce model size. This function implements
    channel pruning
    """
    # pruning needs eval mode
    model.eval()

    imp = tp.importance.MagnitudeImportance(p=2)

    # Don't prune the final conv
    ignored_layers = [model.outc]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_input,
        importance=imp,
        ch_sparsity=ch_sparsity,
        iterative_steps=iterative_steps,
        ignored_layers=ignored_layers,
    )

    for _ in range(iterative_steps):
        pruner.step()

    return model


# Unstructured pruning (sparsity on top of structured pruning)
def apply_unstructured_pruning(model: nn.Module, amount: float = 0.5) -> nn.Module:
    """ applys unstructired pruning """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)
    return model


def bake_unstructured_pruning(model: nn.Module) -> nn.Module:
    """
    Remove pruning reparam (weight_orig + mask) and keep only dense `weight`
    tensors with zeros baked in. This just makes combining with QAT a bit easier
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")
    return model

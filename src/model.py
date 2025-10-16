import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
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
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out=None):
        super().__init__()
        if c_out is None:
            c_out = c_skip
        # learnable upsampling that doubles H and W exactly
        self.up = nn.ConvTranspose2d(c_in, c_skip, kernel_size=2, stride=2)
        self.conv = DoubleConv(2 * c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        # make sure the skip feature and upsampled feature align perfectly
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, dw, 0, dh))
        return self.conv(torch.cat([skip, x], dim=1))


class UNetDenoise(nn.Module):
    """Symmetric U-Net for RGB denoising (predicts residual noise)."""
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        # encoder
        self.inc   = DoubleConv(in_ch, base_ch)              
        self.down1 = Down(base_ch, base_ch * 2)             
        self.down2 = Down(base_ch * 2, base_ch * 4)         
        self.down3 = Down(base_ch * 4, base_ch * 8)     
        # bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)  # C=16b
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

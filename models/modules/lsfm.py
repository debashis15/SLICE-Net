from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSFM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_dil = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.act(self.conv1(f))
        H, W = z.shape[-2:]
        ph, pw = (H % 2, W % 2)
        if ph or pw:
            z = F.pad(z, (0, pw, 0, ph), mode='reflect')
        z = self.pool(z)
        z = self.conv_dil(z)
        g = self.sigmoid(z)
        g = F.interpolate(g, size=f.shape[-2:], mode='nearest')
        return (f * g, f * (1.0 - g))

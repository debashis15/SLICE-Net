from __future__ import annotations
import torch
import torch.nn as nn

class _HFBranch(nn.Module):
    def __init__(self, channels: int, kernel: int=3) -> None:
        super().__init__()
        pad = kernel // 2
        self.dw1 = nn.Conv2d(channels, channels, kernel, padding=pad, groups=channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.dw2 = nn.Conv2d(channels, channels, kernel, padding=pad, groups=channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hf: torch.Tensor) -> torch.Tensor:
        a = self.act1(self.dw1(hf))
        a = self.act2(self.dw2(a))
        a = self.sigmoid(self.proj(a))
        return hf * a

class _LFBranch(nn.Module):
    def __init__(self, channels: int, kernel: int=3, dilation: int=2) -> None:
        super().__init__()
        pad = dilation * (kernel // 2)
        self.dil1 = nn.Conv2d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dil2 = nn.Conv2d(channels, channels, 1)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lf: torch.Tensor) -> torch.Tensor:
        a = self.act1(self.dil1(lf))
        a = self.gap(a)
        a = self.act2(self.dil2(a))
        a = self.sigmoid(self.proj(a))
        return lf * a

class FAAR(nn.Module):
    def __init__(self, channels: int, kernel: int=3) -> None:
        super().__init__()
        self.hf_branch = _HFBranch(channels, kernel=kernel)
        self.lf_branch = _LFBranch(channels, kernel=kernel)

    def forward(self, lf: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        r_lf = self.lf_branch(lf)
        r_hf = self.hf_branch(hf)
        return torch.cat([r_lf, r_hf], dim=1)

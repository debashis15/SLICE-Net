from __future__ import annotations
import torch
import torch.nn as nn

class _MKBranch(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel, padding='same')
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dw = nn.Conv2d(channels, channels, kernel, padding='same', groups=channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.conv(x))
        a = self.sigmoid(self.pool(self.dw(z)))
        return z * a

class PMKSAB(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...]=(1, 2, 3, 4, 5)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([_MKBranch(channels, k) for k in kernels])
        self.fuse = nn.Conv2d(channels * len(kernels), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1))

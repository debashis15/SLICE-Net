from __future__ import annotations
import torch
import torch.nn as nn

class VEB(nn.Module):
    def __init__(self, in_channels: int, out_channels: int=3) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels, out_channels, 3, padding=1))

    def forward(self, fused: torch.Tensor, low_light: torch.Tensor) -> torch.Tensor:
        residual = self.body(fused)
        if residual.shape[-2:] != low_light.shape[-2:]:
            residual = nn.functional.interpolate(residual, size=low_light.shape[-2:], mode='bilinear', align_corners=False)
        return low_light - residual

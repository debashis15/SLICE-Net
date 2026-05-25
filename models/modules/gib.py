from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_dir_kernel(direction: str) -> torch.Tensor:
    if direction == 'x':
        k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
    elif direction == 'y':
        k = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 4.0
    elif direction == 'd1':
        k = torch.tensor([[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]) / 4.0
    elif direction == 'd2':
        k = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]]) / 4.0
    else:
        raise ValueError(f'Unknown direction {direction!r}')
    return k

class DirectionalGradient(nn.Module):
    def __init__(self, channels: int=3) -> None:
        super().__init__()
        kernels = torch.stack([_make_dir_kernel(d) for d in ('x', 'y', 'd1', 'd2')], dim=0)
        kernels = kernels.unsqueeze(1)
        self.register_buffer('kx', kernels[0:1].repeat(channels, 1, 1, 1))
        self.register_buffer('ky', kernels[1:2].repeat(channels, 1, 1, 1))
        self.register_buffer('kd1', kernels[2:3].repeat(channels, 1, 1, 1))
        self.register_buffer('kd2', kernels[3:4].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.ky, padding=1, groups=self.channels)
        gd1 = F.conv2d(x, self.kd1, padding=1, groups=self.channels)
        gd2 = F.conv2d(x, self.kd2, padding=1, groups=self.channels)
        return torch.sqrt(gx * gx + gy * gy + gd1 * gd1 + gd2 * gd2 + 1e-08)

class GradientInteractionBlock(nn.Module):
    def __init__(self, in_channels: int=3, feat_channels: int=16) -> None:
        super().__init__()
        self.grad = DirectionalGradient(in_channels)
        self.proj_img = nn.Sequential(nn.Conv2d(in_channels, feat_channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.proj_grad = nn.Sequential(nn.Conv2d(in_channels, feat_channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.grad(x)
        return torch.cat([self.proj_img(x), self.proj_grad(g)], dim=1)

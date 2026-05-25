from __future__ import annotations
import math
import torch
import torch.nn as nn

def _build_dct_matrix(N: int) -> torch.Tensor:
    n = torch.arange(N, dtype=torch.float64)
    k = n.view(-1, 1)
    basis = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
    alpha = torch.full((N,), math.sqrt(2.0 / N), dtype=torch.float64)
    alpha[0] = math.sqrt(1.0 / N)
    return (alpha.view(-1, 1) * basis).float()

class BlockDCT2d(nn.Module):
    def __init__(self, block_size: int=4) -> None:
        super().__init__()
        self.block_size = block_size
        D = _build_dct_matrix(block_size)
        self.register_buffer('D', D, persistent=False)
        self.register_buffer('Dt', D.t().contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.block_size
        B, C, H, W = x.shape
        assert H % s == 0 and W % s == 0, f'Input H={H}, W={W} must be divisible by block_size={s}.'
        x = x.view(B, C, H // s, s, W // s, s)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = torch.matmul(self.D, x)
        x = torch.matmul(x, self.Dt)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()
        x = x.view(B, C * s * s, H // s, W // s)
        return x

class BlockIDCT2d(nn.Module):
    def __init__(self, block_size: int=4, channels: int=3) -> None:
        super().__init__()
        self.block_size = block_size
        self.channels = channels
        D = _build_dct_matrix(block_size)
        self.register_buffer('D', D, persistent=False)
        self.register_buffer('Dt', D.t().contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.block_size
        C = self.channels
        B, CL, h, w = x.shape
        assert CL == C * s * s, f'Channel dim {CL} does not match channels={C} * block_size^2={s * s}.'
        x = x.view(B, C, s, s, h, w)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()
        x = torch.matmul(self.Dt, x)
        x = torch.matmul(x, self.D)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, h * s, w * s)
        return x

def pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    if pad_h or pad_w:
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return (x, (pad_h, pad_w))

def crop_pad(x: torch.Tensor, pads: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pads
    if pad_h:
        x = x[..., :-pad_h, :]
    if pad_w:
        x = x[..., :-pad_w]
    return x

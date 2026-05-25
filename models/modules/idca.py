from __future__ import annotations
import torch
import torch.nn as nn

class _ScaleEmbed(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale_conv = nn.Conv2d(channels, channels, kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale_conv(self.gap(self.proj(x)))

class IDCA(nn.Module):
    def __init__(self, p_channels: int, q_channels: int, scales: tuple[int, ...]=(1, 3, 5)) -> None:
        super().__init__()
        common = min(p_channels, q_channels)
        self.proj_p = nn.Conv2d(p_channels, common, 1) if p_channels != common else nn.Identity()
        self.proj_q = nn.Conv2d(q_channels, common, 1) if q_channels != common else nn.Identity()
        self.p_branches = nn.ModuleList([_ScaleEmbed(common, k) for k in scales])
        self.q_branches = nn.ModuleList([_ScaleEmbed(common, k) for k in scales])
        self.dw_refine = nn.ModuleList([nn.Conv2d(common, common, 3, padding=1, groups=common) for _ in scales])
        self.attn_proj = nn.Conv2d(common, common, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.out_channels = common * 2
        self.common_channels = common

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = self.proj_p(p)
        q = self.proj_q(q)
        if q.shape[-2:] != p.shape[-2:]:
            q = nn.functional.interpolate(q, size=p.shape[-2:], mode='bilinear', align_corners=False)
        c_sum = None
        for pb, qb, dw in zip(self.p_branches, self.q_branches, self.dw_refine):
            zp = pb(p)
            zq = qb(q)
            c_k = dw(zp * zq)
            c_sum = c_k if c_sum is None else c_sum + c_k
        attn = self.sigmoid(self.attn_proj(c_sum))
        fused = torch.cat([p, q], dim=1)
        attn_full = torch.cat([attn, attn], dim=1)
        return fused * attn_full

from __future__ import annotations
import math
import warnings
from typing import Optional
import torch
import torch.nn.functional as F

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float=1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return (10.0 * torch.log10(data_range ** 2 / mse)).mean()

def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= (window_size - 1) / 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(channels, 1, window_size, window_size)
    return window.contiguous()

def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int=11, sigma: float=1.5, data_range: float=1.0) -> torch.Tensor:
    C = pred.shape[1]
    window = _gaussian_kernel(window_size, sigma, C, pred.device)
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq, mu2_sq = (mu1 * mu1, mu2 * mu2)
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu12
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    smap = (2 * mu12 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return smap.mean()

class _NIQEHolder:
    def __init__(self) -> None:
        self._metric = None
        self._available: Optional[bool] = None

    def _ensure(self, device: torch.device) -> None:
        if self._available is False:
            return
        if self._metric is not None:
            return
        try:
            import pyiqa
            self._metric = pyiqa.create_metric('niqe', device=device)
            self._available = True
        except Exception as e:
            warnings.warn(f'pyiqa not available, NIQE will return NaN. ({e})', RuntimeWarning)
            self._available = False

    def __call__(self, x: torch.Tensor) -> float:
        self._ensure(x.device)
        if not self._available:
            return float('nan')
        with torch.no_grad():
            return float(self._metric(x).mean().item())
niqe = _NIQEHolder()

def evaluate_pair(pred: torch.Tensor, target: torch.Tensor, data_range: float=1.0) -> dict[str, float]:
    with torch.no_grad():
        return {'psnr': float(psnr(pred, target, data_range=data_range).item()), 'ssim': float(ssim(pred, target, data_range=data_range).item())}

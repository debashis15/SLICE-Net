from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float=0.001, reduction: str='mean') -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= (window_size - 1) / 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window.expand(channels, 1, window_size, window_size).contiguous()

def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int=11, sigma: float=1.5, data_range: float=1.0) -> torch.Tensor:
    C = pred.shape[1]
    window = _gaussian_window(window_size, sigma, C, pred.device)
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
    ssim_map = (2 * mu12 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class CharbonnierSSIMLoss(nn.Module):
    def __init__(self, alpha: float=0.85, eps: float=0.001) -> None:
        super().__init__()
        self.alpha = alpha
        self.charb = CharbonnierLoss(eps=eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.charb(pred, target) + (1 - self.alpha) * (1 - ssim(pred, target))

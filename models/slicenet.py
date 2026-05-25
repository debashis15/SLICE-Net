from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from .modules import BlockDCT2d, BlockIDCT2d, FAAR, GradientInteractionBlock, IDCA, LSFM, PMKSAB, VEB, crop_pad, pad_to_multiple

@dataclass
class SLICENetConfig:
    in_channels: int = 3
    out_channels: int = 3
    feat_channels: int = 40
    num_stages: int = 4
    block_size: int = 4
    pmksab_kernels: tuple[int, ...] = (1, 2, 3, 4, 5)
    idca_scales: tuple[int, ...] = (1, 3, 5)

class FrequencyStage(nn.Module):
    def __init__(self, channels: int, block_size: int) -> None:
        super().__init__()
        self.dct = BlockDCT2d(block_size=block_size)
        self.lsfm = LSFM(channels=channels * block_size * block_size)
        self.faar = FAAR(channels=channels * block_size * block_size)
        self.merge = nn.Conv2d(2 * channels * block_size * block_size, channels * block_size * block_size, 1)
        self.idct = BlockIDCT2d(block_size=block_size, channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, pads = pad_to_multiple(x, self.dct.block_size)
        f = self.dct(x_pad)
        lf, hf = self.lsfm(f)
        f_ref = self.faar(lf, hf)
        f_ref = self.merge(f_ref)
        x_back = self.idct(f_ref)
        x_back = crop_pad(x_back, pads)
        return x + x_back

class SLICENet(nn.Module):
    def __init__(self, cfg: SLICENetConfig | None=None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SLICENetConfig()
        self.cfg = cfg
        self.gib = GradientInteractionBlock(in_channels=cfg.in_channels, feat_channels=cfg.feat_channels)
        s_channels = 2 * cfg.feat_channels
        self.gesdp = nn.ModuleList([PMKSAB(channels=s_channels, kernels=cfg.pmksab_kernels) for _ in range(cfg.num_stages)])
        self.lfrp = nn.ModuleList([FrequencyStage(channels=cfg.in_channels, block_size=cfg.block_size) for _ in range(cfg.num_stages)])
        self.freq_to_feat = nn.Conv2d(cfg.in_channels, s_channels, 1)
        self.idca = IDCA(p_channels=s_channels, q_channels=s_channels, scales=cfg.idca_scales)
        self.veb = VEB(in_channels=self.idca.out_channels, out_channels=cfg.out_channels)

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.gib(x)
        for blk in self.gesdp:
            s = blk(s) + s
        f_spatial = x
        for stage in self.lfrp:
            f_spatial = stage(f_spatial)
        f_feat = self.freq_to_feat(f_spatial)
        u = self.idca(s, f_feat)
        return self.veb(u, x)

def build_model(cfg: SLICENetConfig | None=None) -> SLICENet:
    return SLICENet(cfg)

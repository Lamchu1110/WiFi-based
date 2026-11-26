"""Lightweight temporal convolutions preserving the input time axis."""
import torch
from torch import nn


class MultiScaleDilatedTCN(nn.Module):
    def __init__(self, channels, mid_channels=None):
        super().__init__()
        # 320 channels for MM-Fi / 160 for WiPose: about 0.25M extra
        # parameters including the XYZ decoder, for either backbone.
        mid_channels = mid_channels if mid_channels is not None else max(1, 40960 // channels)
        if channels < 1 or mid_channels < 1:
            raise ValueError("Channel counts must be positive")
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.Conv1d(channels, mid_channels, 1, bias=False),
            ) for d in (1, 2, 4)
        ])
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(3 * mid_channels), nn.GELU(),
            nn.Conv1d(3 * mid_channels, channels, 1),
        )

    def forward(self, x):
        return x + self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


class ShortGatedTCN(nn.Module):
    """CSI temporal adapter on [B,C,F,T], preserving frequency positions.

    Kernels span 3/5 time samples. A zero scalar gate starts at the backbone's
    exact identity; the nonzero random branch allows the gate to learn first.
    """
    def __init__(self, channels=64, mid_channels=32):
        super().__init__()
        if channels < 1 or mid_channels < 1:
            raise ValueError('Channel counts must be positive')
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (1, 3), padding=(0, d),
                          dilation=(1, d), groups=channels, bias=False),
                nn.Conv2d(channels, mid_channels, 1, bias=False),
            ) for d in (1, 2)
        ])
        self.fusion = nn.Sequential(
            nn.GroupNorm(1, 2*mid_channels), nn.GELU(),
            nn.Conv2d(2*mid_channels, channels, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] < 2:
            raise ValueError('ShortGatedTCN requires [B,C,F,T] with T>=2')
        correction = self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))
        return x + self.alpha * correction


class ShortResidualTCN(nn.Module):
    """Identity-initialized early CSI adapter without a scalar gradient gate.

    The output projection starts at zero. It receives gradients immediately;
    temporal trunk gradients start after that projection becomes nonzero.
    This is an experimental alternative, not a claim of improved accuracy.
    """
    def __init__(self, channels=64, mid_channels=32):
        super().__init__()
        if channels < 1 or mid_channels < 1:
            raise ValueError('Channel counts must be positive')
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (1, 3), padding=(0, d),
                          dilation=(1, d), groups=channels, bias=False),
                nn.Conv2d(channels, mid_channels, 1, bias=False),
            ) for d in (1, 2)
        ])
        self.fusion = nn.Sequential(
            nn.GroupNorm(1, 2*mid_channels), nn.GELU(),
            nn.Conv2d(2*mid_channels, channels, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] < 2:
            raise ValueError('ShortResidualTCN requires [B,C,F,T] with T>=2')
        return x + self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))

"""ST-HPE using the original backbone and decoder, with optional upgrades.

4D input produces a single pose. 5D [B,W,C,F,T] input produces W poses;
W denotes pose frames and T denotes CSI samples inside each frame.
"""
import time
import torch
from torch import nn
from .HPE_no_denoiser import OriginalHPE
from .HPE_Wipose import HPEWiPoseModel
from .tcn_block import MultiScaleDilatedTCN


class STHPE(nn.Module):
    def __init__(self, dataset="mmfi", use_tcn=True, direct_3d=True):
        super().__init__()
        if dataset not in ("mmfi", "wipose"):
            raise ValueError("dataset must be mmfi or wipose")
        self.dataset = dataset
        self.joints = 17 if dataset == "mmfi" else 18
        self.direct_3d = direct_3d
        self.backbone = OriginalHPE() if dataset == "mmfi" else HPEWiPoseModel()
        channels = 128 if dataset == "mmfi" else 256
        self.tcn = MultiScaleDilatedTCN(channels) if use_tcn else nn.Identity()
        if direct_3d:
            old = self.backbone.regression.fc3
            self.backbone.regression.fc3 = nn.Linear(old.in_features, self.joints * 3)

    def forward(self, x):
        start = time.perf_counter()
        if x.ndim not in (4, 5):
            raise ValueError("Expected [B,C,F,T] or [B,W,C,F,T]")
        sequence_shape = x.shape[:2] if x.ndim == 5 else None
        if sequence_shape is not None:
            x = x.flatten(0, 1)
        pool = torch.nn.functional.avg_pool2d
        x = pool(self.backbone.skunit1(x), (2, 2))
        x = self.backbone.skunit2(x)
        if self.dataset == "wipose":
            x = self.backbone.skunit3(pool(x, (2, 2)))
        # Reduce only frequency, retain CSI time. Broadcast temporal residual
        # over frequencies so that the original decoder dimensions are retained.
        temporal = x.mean(dim=2)
        x = x + (self.tcn(temporal) - temporal).unsqueeze(2)
        if self.dataset == "mmfi":
            x = pool(x, (2, 2))
        pose = self.backbone.regression(x).reshape(
            x.shape[0], self.joints, 3 if self.direct_3d else 2)
        if sequence_shape is not None:
            pose = pose.reshape(*sequence_shape, self.joints, pose.shape[-1])
        return pose, time.perf_counter() - start
